"""为 MDN-RNN 生成连续时间序列数据。

和 data_gen_vae.py 的区别：
    - VAE 数据可以稀疏采样（每 20 步一张），只要画面多样就行；
    - RNN 数据必须每一步都保存，因为要学 P(z_{t+1} | a_t, z_t, h_t)。

这份数据每局只玩很少的帧：
    1. 随机赛道种子；
    2. 先 warmup 若干步，跳过开局镜头拉远的动画；
    3. 之后连续玩 frames_per_episode 帧，每一帧都保存；
    4. 换下一个随机种子，继续下一局。

默认 frames_per_episode = seq_len + 1 = 11：
    10 个输入步 + 1 个目标帧，刚好构成一个 10 步的 RNN 窗口。
    最后第 11 帧的 action 是 lane 策略在这一帧会选的动作，但没有下一步，
    训练时不会用到它。

保存的字段：
    images       (N, 96, 96, 3)  每一帧原始 observation
    actions      (N, 3)          每一帧实际执行的动作；每局最后一帧的 action 不使用
    rewards      (N,)            动作得到的 reward；每局最后一帧填 0
    dones        (N,)            环境是否在这一步结束（terminated or truncated）
    episode_ids  (N,)            第几局；同一局的所有帧共享一个 id
    steps        (N,)            局内帧编号，从 0 开始
    seeds        (N,)            这一局使用的赛道随机种子

注意：
    - 这份数据是给 RNN 用的，不要和 data/car_racing_frames_*.npz 混用；
    - 当前每局只有 11 帧，dones 通常全是 False；done 主要是给以后
      需要 done 预测或 Controller 的场景留的字段；
    - 如果想一局玩更久，把 --frames-per-episode 调大，RNN 训练时可以
      从一局里切出多个窗口。

运行：
    conda activate world_model
    python data_gen_rnn.py

    # 想多玩几帧
    python data_gen_rnn.py --frames-per-episode 51
"""

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np

from car_racing_demo import ENV_ID, lane_action


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="生成 MDN-RNN 连续序列数据")
    parser.add_argument("--num-frames", type=int, default=10000, help="总共保存多少帧")
    parser.add_argument("--seed", type=int, default=0, help="基础随机种子")
    parser.add_argument("--warmup", type=int, default=60, help="每局开始前先跑多少步")
    parser.add_argument("--seq-len", type=int, default=10, help="RNN 窗口长度 T")
    parser.add_argument(
        "--frames-per-episode",
        type=int,
        default=0,
        help="每局连续保存多少帧；0 表示 seq_len + 1",
    )
    parser.add_argument(
        "--policy",
        choices=["lane", "random"],
        default="lane",
        help="采集时使用的策略",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/rnn_sequences.npz",
        help="输出 npz 文件",
    )
    parser.add_argument("--quiet", action="store_true", help="少打印一点进度")
    args = parser.parse_args()

    if args.num_frames <= 0:
        parser.error("--num-frames 必须大于 0")
    if args.warmup < 0:
        parser.error("--warmup 不能是负数")
    if args.seq_len <= 0:
        parser.error("--seq-len 必须大于 0")

    if args.frames_per_episode == 0:
        args.frames_per_episode = args.seq_len + 1

    if args.frames_per_episode < args.seq_len + 1:
        parser.error("--frames-per-episode 至少要是 --seq-len + 1")

    return args


def resolve_path(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path


def policy_action(policy, obs, rng):
    """返回当前帧要执行的动作。"""
    if policy == "lane":
        return lane_action(obs)

    # random 策略：动作空间是 [-1,1] × [0,1] × [0,1]
    steer = rng.uniform(-1.0, 1.0)
    gas = rng.uniform(0.0, 1.0)
    brake = rng.uniform(0.0, 1.0)
    return np.array([steer, gas, brake], dtype=np.float32)


def collect_sequences(env, num_frames, base_seed, warmup, frames_per_episode,
                      policy, quiet=False):
    """按局采集连续帧，直到凑够 num_frames 帧左右。"""
    images = []
    actions = []
    rewards = []
    dones = []
    episode_ids = []
    steps = []
    seeds = []

    rng = np.random.default_rng(base_seed)
    attempt = 0          # 尝试过的随机种子数
    episode_id = 0       # 成功采集的局编号
    discarded = 0

    while len(images) < num_frames:
        seed = base_seed + attempt
        attempt += 1

        obs, info = env.reset(seed=seed)

        # 先热身：不保存这些帧
        ended = False
        for _ in range(warmup):
            action = policy_action(policy, obs, rng)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                ended = True
                break

        if ended:
            discarded += 1
            continue

        # 一局连续采集 frames_per_episode 帧
        episode_images = []
        episode_actions = []
        episode_rewards = []
        episode_dones = []

        ok = True
        for step in range(frames_per_episode):
            action = policy_action(policy, obs, rng)

            if step < frames_per_episode - 1:
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                episode_images.append(obs.copy())
                episode_actions.append(action)
                episode_rewards.append(reward)
                episode_dones.append(done)

                obs = next_obs

                if done:
                    # 这一局还没凑够 frames_per_episode 帧，丢掉不完整的数据
                    ok = False
                    break
            else:
                # 最后一帧：它是上一步动作的结果，没有下一步，也没有 action 可执行。
                # 这里仍然保存一个 action，是为了保持数组等长；训练时不会用到。
                episode_images.append(obs.copy())
                episode_actions.append(action)
                episode_rewards.append(0.0)
                episode_dones.append(False)

        if not ok or len(episode_images) != frames_per_episode:
            discarded += 1
            continue

        images.extend(episode_images)
        actions.extend(episode_actions)
        rewards.extend(episode_rewards)
        dones.extend(episode_dones)

        episode_ids.extend([episode_id] * frames_per_episode)
        steps.extend(range(frames_per_episode))
        seeds.extend([seed] * frames_per_episode)

        if not quiet and (episode_id % 50 == 0 or len(images) >= num_frames):
            print(
                f"episode {episode_id:4d} | seed={seed:4d} | "
                f"已保存={len(images):5d}/{num_frames}"
            )

        episode_id += 1

        # 如果 lane 一直失败，避免死循环
        if attempt > 4 * (num_frames // frames_per_episode + 10):
            raise RuntimeError("连续失败太多次，请检查策略或环境")

    print()
    print(f"成功局数 : {episode_id}")
    print(f"丢弃局数 : {discarded}")
    print(f"总帧数   : {len(images)}")

    return (
        np.stack(images),
        np.stack(actions),
        np.array(rewards, dtype=np.float32),
        np.array(dones, dtype=bool),
        np.array(episode_ids, dtype=np.int32),
        np.array(steps, dtype=np.int32),
        np.array(seeds, dtype=np.int32),
    )


def main():
    args = parse_args()
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("生成 MDN-RNN 连续序列数据")
    print("=" * 72)
    print(f"策略           : {args.policy}")
    print(f"每局热身       : {args.warmup} 步")
    print(f"每局保存       : {args.frames_per_episode} 帧")
    print(f"RNN 窗口 T     : {args.seq_len}")
    print(f"目标帧数       : {args.num_frames}")
    print(f"输出文件       : {output_path}")
    print()

    env = gym.make(ENV_ID, render_mode=None, continuous=True)
    images, actions, rewards, dones, episode_ids, steps, seeds = collect_sequences(
        env,
        num_frames=args.num_frames,
        base_seed=args.seed,
        warmup=args.warmup,
        frames_per_episode=args.frames_per_episode,
        policy=args.policy,
        quiet=args.quiet,
    )
    env.close()

    np.savez_compressed(
        output_path,
        images=images,
        actions=actions,
        rewards=rewards,
        dones=dones,
        episode_ids=episode_ids,
        steps=steps,
        seeds=seeds,
        frames_per_episode=np.array(args.frames_per_episode),
        seq_len=np.array(args.seq_len),
        warmup=np.array(args.warmup),
        policy=np.array(args.policy),
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print()
    print("=" * 72)
    print("完成")
    print("=" * 72)
    print(f"images      : {images.shape}, dtype={images.dtype}")
    print(f"actions     : {actions.shape}, dtype={actions.dtype}")
    print(f"rewards     : {rewards.shape}, dtype={rewards.dtype}")
    print(f"dones       : {dones.shape}, dtype={dones.dtype}")
    print(f"episode_ids : {episode_ids.shape}，共 {len(np.unique(episode_ids))} 局")
    print(f"steps       : {steps.shape}")
    print(f"文件大小    : {size_mb:.2f} MB")
    print()
    print("每一局是连续帧；最后一帧的 action/reward 只是占位，训练时用不到。")


if __name__ == "__main__":
    main()
