"""世界模型最终演示：VAE + MDN-RNN + 线性控制器 + CMA-ES。

这个文件回答一个问题：
    在已经训练好 VAE 和 MDN-RNN 之后，CMA-ES 到底在优化什么？

答案：
    CMA-ES 优化的不是某一帧的动作 a，而是线性控制器的参数 theta。
    控制器：a_t = squash(W_c @ [z_t, h_t] + b_c)
    z_t 是当前帧的 VAE 编码（取 mu），h_t 是当前 RNN 隐状态。
    论文里 W_c 是 3x288，b_c 是 3，一共 867 个参数。

    f(theta) = 用这个控制器在真实环境里跑一整局，累计 reward 的总和。
    CMA-ES 找 theta，让 f(theta) 最大。

注意：
    - CarRacing 的 MDN-RNN 只预测 z，不预测 reward；所以控制器在
      真实环境里评估，用环境给的 reward。论文里也只有 C 能看 reward。
    - 这里用的是当前 z_t，不是 RNN 预测的 z_{t+1}。h_t 已经包含了对未来的预期；
      预测的 z_{t+1} 用于“梦境” rollout / 可视化，不是当前动作的输入。
    - 为了单机可跑，默认评估预算比论文小很多；论文是 population=64，
      每个候选 16 个 seed。

运行：
    # 训练控制器（headless，比较慢）
    python main.py --mode train --generations 5 --fitness-steps 300

    # 用训练好的控制器自动玩（打开窗口）
    python main.py --mode demo

    # 不开窗口，只评估分数
    python main.py --mode evaluate --episodes 5
"""

import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from car_racing_demo import ENV_ID
from cma import cma_es
from mdn_rnn import MDNRNN
from vae import VAE, preprocess_image


PROJECT_DIR = Path(__file__).resolve().parent
LATENT_DIM = 32
HIDDEN_SIZE = 256
ACTION_DIM = 3
FEATURE_DIM = LATENT_DIM + HIDDEN_SIZE
THETA_SIZE = ACTION_DIM * FEATURE_DIM + ACTION_DIM  # 3*288 + 3 = 867


def parse_args():
    parser = argparse.ArgumentParser(description="世界模型自动玩 CarRacing")
    parser.add_argument(
        "--mode",
        choices=["train", "demo", "evaluate"],
        default="demo",
        help="train=CMA-ES 训练控制器；demo=打开窗口玩；evaluate=只评估",
    )
    parser.add_argument(
        "--vae-checkpoint",
        type=str,
        default="data/vae_carracing.pth",
        help="VAE 参数",
    )
    parser.add_argument(
        "--rnn-checkpoint",
        type=str,
        default="data/mdn_rnn.pth",
        help="MDN-RNN 参数",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="data/controller.npz",
        help="控制器参数文件",
    )
    parser.add_argument("--episodes", type=int, default=1, help="demo/evaluate 跑几局")
    parser.add_argument("--max-steps", type=int, default=1000, help="每局最多多少步")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--no-render", action="store_true", help="demo 时不打开窗口")
    parser.add_argument("--generations", type=int, default=5, help="CMA-ES 迭代代数")
    parser.add_argument("--sigma0", type=float, default=0.1, help="CMA-ES 初始步长")
    parser.add_argument("--fitness-steps", type=int, default=300, help="每个候选评估多少步")
    parser.add_argument("--fitness-episodes", type=int, default=1, help="每个候选评估几条赛道")
    parser.add_argument("--demo-after-train", action="store_true", help="训练完立刻 demo")
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes 必须大于 0")
    if args.max_steps <= 0:
        parser.error("--max-steps 必须大于 0")
    if args.generations <= 0:
        parser.error("--generations 必须大于 0")
    if args.sigma0 <= 0:
        parser.error("--sigma0 必须大于 0")
    if args.fitness_steps <= 0:
        parser.error("--fitness-steps 必须大于 0")
    if args.fitness_episodes <= 0:
        parser.error("--fitness-episodes 必须大于 0")

    return args


def resolve_path(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path


def load_models(args, device):
    vae_path = resolve_path(args.vae_checkpoint)
    rnn_path = resolve_path(args.rnn_checkpoint)

    if not vae_path.exists():
        raise FileNotFoundError(f"找不到 VAE 参数：{vae_path}")
    if not rnn_path.exists():
        raise FileNotFoundError(f"找不到 MDN-RNN 参数：{rnn_path}")

    vae = VAE(LATENT_DIM).to(device)
    vae.load_state_dict(torch.load(vae_path, map_location=device))
    vae.eval()

    rnn = MDNRNN(LATENT_DIM, ACTION_DIM).to(device)
    rnn.load_state_dict(torch.load(rnn_path, map_location=device))
    rnn.eval()

    print(f"VAE 参数 : {vae_path}")
    print(f"RNN 参数 : {rnn_path}")
    return vae, rnn


class Controller:
    """线性控制器：a = squash(W @ [z, h] + b)。"""

    def __init__(self, theta):
        theta = np.asarray(theta, dtype=np.float32)
        if theta.size != THETA_SIZE:
            raise ValueError(f"控制器参数应为 {THETA_SIZE} 维，实际是 {theta.size}")

        self.W = theta[:ACTION_DIM * FEATURE_DIM].reshape(ACTION_DIM, FEATURE_DIM)
        self.b = theta[ACTION_DIM * FEATURE_DIM:]

    def act(self, z, h):
        feature = np.concatenate([z, h]).astype(np.float32)
        raw = self.W @ feature + self.b

        # steering -> [-1,1]
        steer = np.tanh(raw[0])
        # gas / brake -> [0,1]
        gas = (np.tanh(raw[1]) + 1.0) / 2.0
        brake = (np.tanh(raw[2]) + 1.0) / 2.0

        return np.array([steer, gas, brake], dtype=np.float32)


def encode_observation(vae, obs, device):
    """当前帧 -> z_t（取 VAE 的 mu）。"""
    image = preprocess_image(obs)  # 96x96 -> 64x64
    x = image.astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        mu, _ = vae.encode(x)

    return mu[0].cpu().numpy()


def run_episode(env, vae, rnn, controller, seed, max_steps, device):
    """用控制器跑一局，返回累计 reward 和步数。"""
    obs, info = env.reset(seed=seed)

    hidden = None
    total_reward = 0.0

    for step in range(max_steps):
        z = encode_observation(vae, obs, device)

        if hidden is None:
            h = np.zeros(HIDDEN_SIZE, dtype=np.float32)
        else:
            h = hidden[0][0, 0].detach().cpu().numpy()

        action = controller.act(z, h)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # 用“当前 z + 实际动作”更新 RNN 隐状态
        x = np.concatenate([z, action]).astype(np.float32)
        x = torch.from_numpy(x).reshape(1, 1, -1).to(device)

        with torch.no_grad():
            _, hidden = rnn.lstm(x, hidden)

        if terminated or truncated:
            break

    return total_reward, step + 1


def fitness(theta, env, vae, rnn, seed, episodes, max_steps, device):
    """CMA-ES 的适应度：同一组 seed 上，控制器参数 theta 的平均累计 reward。"""
    controller = Controller(theta)

    total = 0.0
    for i in range(episodes):
        score, steps = run_episode(
            env,
            vae,
            rnn,
            controller,
            seed=seed + i,
            max_steps=max_steps,
            device=device,
        )
        total += score

    return total / episodes


def make_initial_theta():
    """初始控制器：不转向，轻踩油门，少刹车。"""
    theta = np.zeros(THETA_SIZE, dtype=np.float32)

    # b = [steer, gas, brake]
    # 设置一个“先往前开”的偏置，避免初始时油门和刹车各 0.5。
    theta[-3 + 1] = 1.0   # gas bias
    theta[-3 + 2] = -1.0  # brake bias

    return theta


def train_controller(args, vae, rnn, device):
    controller_path = resolve_path(args.controller)
    controller_path.parent.mkdir(parents=True, exist_ok=True)

    env = gym.make(ENV_ID, render_mode=None, continuous=True)

    def evaluate_theta(theta):
        return fitness(
            theta,
            env,
            vae,
            rnn,
            seed=args.seed,
            episodes=args.fitness_episodes,
            max_steps=args.fitness_steps,
            device=device,
        )

    x0 = make_initial_theta()
    print()
    print("=" * 72)
    print("开始 CMA-ES 训练控制器")
    print("=" * 72)
    print(f"控制器参数量 : {THETA_SIZE}")
    print(f"初始 sigma   : {args.sigma0}")
    print(f"迭代代数     : {args.generations}")
    print(f"每候选步数   : {args.fitness_steps}")
    print(f"每候选赛道数 : {args.fitness_episodes}")
    print()
    print(f"初始 f(theta) = {evaluate_theta(x0):.2f}")
    print()

    t0 = time.time()
    best_theta, best_sigma, _ = cma_es(
        evaluate_theta,
        x0,
        args.sigma0,
        maxiter=args.generations,
        seed=args.seed,
        verbose=True,
    )
    elapsed = time.time() - t0

    final_fitness = evaluate_theta(best_theta)
    env.close()

    np.savez_compressed(
        controller_path,
        theta=best_theta,
        sigma=best_sigma,
        generations=args.generations,
        fitness_steps=args.fitness_steps,
        fitness_episodes=args.fitness_episodes,
    )

    print()
    print(f"训练完成，用时 {elapsed:.1f}s")
    print(f"控制器已保存：{controller_path}")
    print(f"最终 f(theta) = {final_fitness:.2f}")

    return best_theta


def load_controller(path):
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"找不到控制器：{path}\n"
            "请先运行：python main.py --mode train"
        )
    data = np.load(path)
    return data["theta"]


def play(args, vae, rnn, render):
    controller_theta = load_controller(args.controller)
    controller = Controller(controller_theta)

    device = next(vae.parameters()).device
    env = gym.make(ENV_ID, render_mode="human" if render else None, continuous=True)

    scores = []
    for episode in range(args.episodes):
        score, steps = run_episode(
            env,
            vae,
            rnn,
            controller,
            seed=args.seed + episode,
            max_steps=args.max_steps,
            device=device,
        )
        scores.append(score)
        print(f"episode {episode:2d} | seed={args.seed + episode:4d} | "
              f"steps={steps:4d} | reward={score:8.2f}")

    env.close()
    print()
    print(f"平均 reward : {np.mean(scores):.2f}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备 : {device}")

    vae, rnn = load_models(args, device)

    if args.mode == "train":
        train_controller(args, vae, rnn, device)
        if args.demo_after_train:
            play(args, vae, rnn, render=True)
    elif args.mode == "demo":
        play(args, vae, rnn, render=not args.no_render)
    else:
        play(args, vae, rnn, render=False)


if __name__ == "__main__":
    main()
