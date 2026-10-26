"""CarRacing-v3 小游戏演示（Gymnasium + Box2D）。

这个文件只做一件事：把世界模型论文里 CarRacing 的输入、输出、动作和奖励
原原本本地展示出来，为后面的 VAE、MDN-RNN 和 CMA-ES 做准备。

运行前先激活环境（环境已经建好）：
    conda activate world_model

常用命令：
    # 默认：打开窗口，用简单的手写车道跟随策略跑一个 episode
    python car_racing_demo.py

    # 看随机动作
    python car_racing_demo.py --policy random --seed 1

    # 自己用方向键开（需要窗口）
    python car_racing_demo.py --policy keyboard

    # 不开窗口，只打印每一步的数据
    python car_racing_demo.py --policy random --no-render

    # 把 observation 存成 GIF，方便反复看
    python car_racing_demo.py --policy lane --gif lane.gif

策略说明：
    idle      什么都不做
    straight  一直踩油门
    random    每一帧从动作空间里随机采样
    lane      手写的“看灰色赛道”启发式，仅用于演示，不是世界模型的一部分
    keyboard  方向键驾驶：←/→ 转向，↑ 油门，↓ 刹车，ESC 退出

几个容易混淆的点：
    1. 论文用的是 CarRacing-v0，这里用 Gymnasium 的 CarRacing-v3。核心规则一样：
       96x96 RGB 图像、3 维连续动作、每帧 -0.1、每经过一个新 tile 加 1000/N。
    2. 论文的 VAE 输入是 64x64x3，所以后面训练 VAE 前要把 96x96 resize 到 64x64。
    3. observation 最下面约 12 行是黑色 HUD（显示累计分数），它不是赛道状态。
       论文 Figure 10 里的 VAE 输入截图没有这条 HUD，所以复现时建议先裁掉，
       例如 obs[:84, :, :]，再 resize 到 64x64。否则分数文字也会被编码进 z。
"""

import argparse
import time

import gymnasium as gym
import numpy as np


ENV_ID = "CarRacing-v3"


def parse_args():
    parser = argparse.ArgumentParser(description="CarRacing-v3 小游戏演示")
    parser.add_argument(
        "--policy",
        choices=["idle", "straight", "random", "lane", "keyboard"],
        default="lane",
        help="用哪种策略开车（默认 lane）",
    )
    parser.add_argument("--seed", type=int, default=0, help="赛道随机种子")
    parser.add_argument("--steps", type=int, default=1000, help="最多跑多少步")
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="不打开 pygame 窗口（只计算，速度快）",
    )
    parser.add_argument(
        "--gif",
        type=str,
        default=None,
        help="把 observation 保存成 GIF，例如 --gif lane.gif",
    )
    args = parser.parse_args()

    if args.policy == "keyboard" and args.no_render:
        parser.error("--policy keyboard 必须打开窗口，不能和 --no-render 一起用")

    return args


def describe_environment():
    """打印环境的基本信息，不跑 episode。"""
    env = gym.make(ENV_ID, continuous=True)
    env_discrete = gym.make(ENV_ID, continuous=False)

    print("=" * 72)
    print("CarRacing-v3 环境说明")
    print("=" * 72)

    print("\n[观测 observation]")
    print(f"  observation_space = {env.observation_space}")
    print(f"  形状 = {env.observation_space.shape}，dtype = {env.observation_space.dtype}")
    print("  含义：96x96 的 RGB 俯视图，车固定在画面下方中间，赛道随车移动。")
    print("  注意：最下面约 12 行是黑色 HUD，显示累计 reward，不是赛道本身。")
    print("        论文 Figure 10 的 VAE 输入没有这条 HUD，复现时建议裁掉 obs[:84] 再 resize。")

    print("\n[动作 action]")
    print(f"  continuous=True  -> action_space = {env.action_space}")
    print("    action[0] = steering，方向盘：-1 左，0 直行，+1 右")
    print("    action[1] = gas，油门：0 不加速，1 全油门")
    print("    action[2] = brake，刹车：0 不刹车，1 全刹车")
    print("    low  = [-1, 0, 0]")
    print("    high = [ 1, 1, 1]")
    print(f"\n  continuous=False -> action_space = {env_discrete.action_space}")
    print("    0 = do nothing，1 = steer right，2 = steer left，3 = gas，4 = brake")
    print("    论文用的是连续动作，所以后面 CMA-ES 控制器输出 3 个实数。")

    print("\n[奖励 reward]")
    print("  每走一帧：-0.1")
    print("  每第一次经过一个新的赛道 tile：+1000 / N")
    print("    N 是这条赛道的 tile 总数，所以跑完一圈大约能拿到 1000 分。")
    print("  开出 playfield 很远：这一帧 -100，并且 episode 立即结束。")
    print("  跑完一圈（或达到 1000 步上限）：episode 结束。")
    print("  注意：撞出 playfield 时，step() 返回 -100，但画面 HUD 里的累计分数")
    print("        不会减这 100。所以 CMA 的 f 必须累加 step() 的返回值，不能读 env.reward。")
    print("  论文的达标线：100 次随机赛道平均 reward >= 900。")

    print("\n[episode 长度]")
    print(f"  TimeLimit 默认最大步数 = {env.spec.max_episode_steps}")
    print("  50 FPS，所以 1000 步大约是 20 秒。")

    env.close()
    env_discrete.close()
    print()


def policy_action(policy, obs, rng):
    """根据策略返回一个 3 维连续动作。"""
    if policy == "idle":
        # 不踩油门也不打方向
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    if policy == "straight":
        # 一直踩油门，方向保持 0：用来观察“不转弯会怎样”
        return np.array([0.0, 0.5, 0.0], dtype=np.float32)

    if policy == "random":
        # 从合法范围里均匀采样
        steer = rng.uniform(-1.0, 1.0)
        gas = rng.uniform(0.0, 1.0)
        brake = rng.uniform(0.0, 1.0)
        return np.array([steer, gas, brake], dtype=np.float32)

    if policy == "lane":
        return lane_action(obs)

    raise ValueError(f"未知策略：{policy}")


def lane_action(obs):
    """手写的车道跟随策略。

    思路非常简单：
      1. 赛道是灰色的，草地是绿色的，所以灰色像素就是可行驶区域；
      2. 在画面中上部取几行，算出每行灰色区域的中心 x；
      3. 如果中心在画面右边，就向右打方向；在左边就向左打；
      4. 油门固定踩一点点。

    这只是一个用来演示动作含义的 baseline，不是世界模型的一部分。
    """
    image = obs.astype(np.int16)  # uint8 相减会溢出，先转成 int16
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    # 灰色赛道：三个通道接近，亮度中等。
    # 这个阈值只对 CarRacing 默认配色有效；domain_randomize=True 时颜色会变。
    road = (np.abs(r - g) < 18) & (np.abs(g - b) < 18) & (r > 55) & (r < 165)

    # 只看车前方的一段（第 30~58 行）。车在画面下方，行号越小看得越远。
    centers = []
    for y in range(30, 60, 2):
        xs = np.nonzero(road[y])[0]
        if len(xs) > 6:
            centers.append(xs.mean())

    if len(centers) == 0:
        # 前方看不到赛道：不转向，轻轻给油
        return np.array([0.0, 0.2, 0.0], dtype=np.float32)

    center = np.median(centers)
    # 画面宽 96，中心是 48。把偏差归一化到 [-1, 1]。
    error = (center - 48.0) / 48.0
    steer = float(np.clip(1.2 * error, -1.0, 1.0))

    return np.array([steer, 0.35, 0.0], dtype=np.float32)


def keyboard_action():
    """读取键盘，返回动作；按 ESC 或关闭窗口时返回 None。"""
    import pygame

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return None

    keys = pygame.key.get_pressed()
    if keys[pygame.K_ESCAPE]:
        return None

    steer = 0.0
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        steer -= 1.0
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        steer += 1.0

    gas = 1.0 if (keys[pygame.K_UP] or keys[pygame.K_w]) else 0.0
    brake = 1.0 if (keys[pygame.K_DOWN] or keys[pygame.K_s]) else 0.0

    return np.array([steer, gas, brake], dtype=np.float32)


def save_gif(path, frames):
    """把 observation 序列保存成 GIF（每 2 步取一帧，近似实时速度）。"""
    import imageio.v3 as iio

    if len(frames) < 2:
        print("帧数太少，跳过 GIF 保存")
        return

    iio.imwrite(path, np.stack(frames), duration=40, loop=0)
    print(f"GIF 已保存：{path}")


def run(args):
    render_mode = "human" if not args.no_render else None
    env = gym.make(ENV_ID, render_mode=render_mode, continuous=True)
    obs, info = env.reset(seed=args.seed)

    rng = np.random.default_rng(args.seed)
    frames = []
    total_reward = 0.0
    steps_done = 0
    reason = f"达到 --steps={args.steps}"

    print(f"reset 完成：obs.shape={obs.shape}, obs.dtype={obs.dtype}")
    print(f"reset 后环境内部累计 reward = {env.unwrapped.reward:.2f}")
    print("（reset 内部会先走一步来生成第一帧，所以出生点附近的 tile 奖励已经算进去了）")
    print()
    print("  step |    steer    gas  brake |   step_reward   total_reward | terminated truncated")
    print("  -----+-------------------------+-------------------------------+-------------------")

    for step in range(args.steps):
        if args.policy == "keyboard":
            action = keyboard_action()
            if action is None:
                reason = "用户退出"
                break
        else:
            action = policy_action(args.policy, obs, rng)

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps_done = step + 1

        if args.gif is not None and step % 2 == 0:
            frames.append(obs.copy())

        # 前 10 步逐步打印，后面每 100 步打印一次
        if step < 10 or (step + 1) % 100 == 0:
            print(
                f"  {step:4d} | {action[0]:+8.3f} {action[1]:6.3f} {action[2]:6.3f}"
                f" | {reward:13.3f} {total_reward:14.3f}"
                f" | {str(terminated):10s} {str(truncated):9s}"
            )

        if terminated or truncated:
            if terminated and info.get("lap_finished") is True:
                reason = "跑完一圈"
            elif terminated and info.get("lap_finished") is False:
                reason = "开出 playfield（-100，回合结束）"
            elif terminated:
                reason = "terminated"
            else:
                reason = "达到 1000 步上限（TimeLimit）"
            break

    if not args.no_render:
        time.sleep(0.5)  # 让最后一帧多停留一会儿

    env_reward = env.unwrapped.reward
    tiles = env.unwrapped.tile_visited_count
    total_tiles = len(env.unwrapped.track)
    env.close()

    print()
    print("=" * 72)
    print("Episode 结束")
    print("=" * 72)
    print(f"  策略          : {args.policy}")
    print(f"  seed          : {args.seed}")
    print(f"  实际步数      : {steps_done}")
    print(f"  真实 return    : {total_reward:.2f}  （所有 step reward 的和，CMA 用这个）")
    print(f"  HUD 里的分数   : {env_reward:.2f}  （环境内部计数，撞出界时不含 -100）")
    if abs(env_reward - total_reward) > 1e-6:
        print("  注意：两者不一致。撞出 playfield 的那一帧 step reward = -100，")
        print("        但 env.reward 只扣了 -0.1，所以不能拿 HUD 分数当适应度。")
    print(f"  赛道 tile     : {tiles}/{total_tiles}")
    print(f"  结束原因      : {reason}")
    print(f"  info          : {info}")
    print()
    print("论文里 CMA-ES 的适应度 f 就是这里的“真实 return”：")
    print("    f(controller) = 一个 episode 里所有 reward 的和")
    print("因为每次 reset 赛道都不一样，论文让每个候选解跑 16 个 seed 再取平均，")
    print("不能只用一个 seed 的分数来评价控制器。")

    if args.gif is not None:
        save_gif(args.gif, frames)


def main():
    args = parse_args()
    describe_environment()
    run(args)


if __name__ == "__main__":
    main()
