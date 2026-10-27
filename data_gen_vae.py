"""生成 CarRacing 的 VAE 训练图片（只给 VAE 用，不用于 MDN-RNN）。

这份数据是“稀疏采样”：每隔 --gap 步保存一张，画面更分散、更多样，
适合逐帧训练 VAE。MDN-RNN 需要连续帧和每一步的动作，请用
data_gen_rnn.py 单独采集。

做这件事的原因：
    VAE 要学会重建赛车画面，所以训练集里必须有足够多的赛道、弯道和车身姿态。
    如果只截取连续帧，相邻图片几乎一样，数据集会很冗余；所以这里让 lane 策略
    先开一段时间，再每隔 --gap 步保存一张 observation。

生成方式：
    1. 用 --seed 作为基础种子，第 i 个 episode 使用 seed = base_seed + i；
    2. 每个 episode 先跑 --warmup 步（跳过开局镜头拉远的动画）；
    3. 之后每跑 --gap 步保存一张 96x96 的原始 observation；
    4. 一个 episode 结束后自动换下一个随机赛道，直到凑够 --num-images 张。

输出：
    data/raw_frames/0000_seed0000_step0080.png ...   （原始 96x96 图片）
    data/car_racing_frames_<num_images>.npz           （打包后的数组，方便读取）

重复运行会先清空 data/raw_frames/，并删除旧的 data/car_racing_frames_*.npz，
所以可以放心重跑。

加载方式：
    import numpy as np
    data = np.load("data/car_racing_frames_10000.npz")
    images = data["images"]      # (10000, 96, 96, 3), uint8, 0..255
    seeds = data["seeds"]
    steps = data["steps"]

注意：
    这里保存的是“原始” observation，所以底部 12 行 HUD 还在。
    训练 VAE 时再决定是否裁掉 HUD，并 resize 到论文的 64x64。
"""

import argparse
import shutil
from pathlib import Path

import gymnasium as gym
import imageio.v3 as iio
import numpy as np

from car_racing_demo import ENV_ID, lane_action


def parse_args():
    parser = argparse.ArgumentParser(description="生成 CarRacing VAE 训练数据")
    parser.add_argument("--num-images", type=int, default=1000, help="总共保存多少张图")
    parser.add_argument("--seed", type=int, default=0, help="基础随机种子")
    parser.add_argument("--gap", type=int, default=20, help="每多少步保存一张图")
    parser.add_argument("--warmup", type=int, default=60, help="每个 episode 先跑多少步再开始截图")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="输出目录（相对路径以本文件所在目录为基准）",
    )
    parser.add_argument("--quiet", action="store_true", help="不打印每个 episode 的进度")
    args = parser.parse_args()

    if args.num_images <= 0:
        parser.error("--num-images 必须大于 0")
    if args.gap <= 0:
        parser.error("--gap 必须大于 0")
    if args.warmup < 0:
        parser.error("--warmup 不能是负数")

    return args


def collect_images(env, num_images, base_seed, gap, warmup, verbose=True):
    """用 lane 策略开车，按固定间隔收集图片。"""
    images = []
    seeds = []
    steps = []
    episode = 0
    total_env_steps = 0

    while len(images) < num_images:
        seed = base_seed + episode
        obs, info = env.reset(seed=seed)

        terminated = False
        truncated = False
        step = 0

        # 先热身：让镜头拉近、车真正开起来，避免保存开局那种缩得很远的画面
        for _ in range(warmup):
            obs, reward, terminated, truncated, info = env.step(lane_action(obs))
            step += 1
            total_env_steps += 1
            if terminated or truncated:
                break

        # 之后每开 gap 步保存一张
        while not (terminated or truncated) and len(images) < num_images:
            for _ in range(gap):
                obs, reward, terminated, truncated, info = env.step(lane_action(obs))
                step += 1
                total_env_steps += 1
                if terminated or truncated:
                    break

            if terminated or truncated:
                break

            images.append(obs.copy())
            seeds.append(seed)
            steps.append(step)

        episode += 1
        if verbose:
            print(
                f"episode {episode - 1:3d} | seed={seed:4d} | "
                f"本局步数={step:4d} | 已收集={len(images):4d}/{num_images}"
            )

    return np.stack(images), np.array(seeds), np.array(steps), total_env_steps


def save_dataset(output_dir, images, seeds, steps, num_images, gap, warmup, base_seed):
    """保存原始 PNG，并打包成 npz。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = output_dir / "raw_frames"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir()

    for i in range(len(images)):
        filename = f"{i:04d}_seed{seeds[i]:04d}_step{steps[i]:04d}.png"
        iio.imwrite(raw_dir / filename, images[i])

    # 只删除上一次生成的同名数据，避免误删别的文件
    for old_npz in output_dir.glob("car_racing_frames_*.npz"):
        old_npz.unlink()

    npz_path = output_dir / f"car_racing_frames_{num_images}.npz"
    np.savez_compressed(
        npz_path,
        images=images,
        seeds=seeds,
        steps=steps,
        num_images=np.array(num_images),
        gap=np.array(gap),
        warmup=np.array(warmup),
        base_seed=np.array(base_seed),
    )

    return raw_dir, npz_path


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir

    print("=" * 72)
    print("生成 CarRacing VAE 训练数据")
    print("=" * 72)
    print(f"基础种子   : {args.seed}")
    print(f"图片数量   : {args.num_images}")
    print(f"截图间隔   : 每 {args.gap} 步一张")
    print(f"每局热身   : {args.warmup} 步")
    print(f"输出目录   : {output_dir}")
    print()

    env = gym.make(ENV_ID, render_mode=None, continuous=True)
    images, seeds, steps, total_env_steps = collect_images(
        env,
        num_images=args.num_images,
        base_seed=args.seed,
        gap=args.gap,
        warmup=args.warmup,
        verbose=not args.quiet,
    )
    env.close()

    raw_dir, npz_path = save_dataset(
        output_dir,
        images,
        seeds,
        steps,
        num_images=args.num_images,
        gap=args.gap,
        warmup=args.warmup,
        base_seed=args.seed,
    )

    print()
    print("=" * 72)
    print("完成")
    print("=" * 72)
    print(f"原始图片 : {raw_dir}")
    print(f"打包数据 : {npz_path}")
    print(f"images   : shape={images.shape}, dtype={images.dtype}, "
          f"min={images.min()}, max={images.max()}")
    print(f"seeds    : {seeds.min()} .. {seeds.max()}（共 {len(np.unique(seeds))} 条赛道）")
    print(f"环境步数 : {total_env_steps}")
    print()
    print("加载示例：")
    print('    import numpy as np')
    print(f'    data = np.load(r"{npz_path}")')
    print('    images = data["images"]   # (N, 96, 96, 3), uint8')


if __name__ == "__main__":
    main()
