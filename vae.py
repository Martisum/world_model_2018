"""CarRacing 的卷积 VAE（论文里的 V Model）。

这个文件只做最基础的 VAE：
    - 编码器：4 层卷积，把 64x64x3 图片变成 mu 和 logvar（各 32 维）
    - 解码器：4 层反卷积，把 32 维 z 还原成 64x64x3 图片
    - 损失：重建误差（L2 / MSE）+ KL 散度
    - 训练：普通的 mini-batch + Adam

没有使用 BatchNorm、学习率调度、混合精度、beta-VAE、数据增强等技巧。
目的先把 VAE 跑通，看清楚每一层在做什么。

数据：
    data/car_racing_frames_1000.npz
    data_gen.py 保存的是原始 96x96 图片，底部 12 行是 HUD。
    这里在读取时裁掉 HUD，并 resize 到论文使用的 64x64：
        image[0:84, 6:90, :]  -> 84x84  （去掉底部 12 行，左右各裁 6 像素）
        resize 到 64x64

    注意：这里没有把小车扣掉。小车本身就是 observation 的一部分，
    论文 Figure 10 的 VAE 输入和重建图里也有小车。先保留它，看看效果；
    如果以后发现小车占据了太多 latent 容量，再考虑遮掉。

运行：
    conda activate world_model
    python vae.py

    # 改训练轮数、batch size、学习率
    python vae.py --epochs 30 --batch-size 64 --lr 0.001

    # 只看看数据能不能读、模型能不能跑
    python vae.py --epochs 1

输出：
    data/vae_carracing.pth         模型参数
    data/vae_reconstructions.png   原图（上排）和重建图（下排）
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


PROJECT_DIR = Path(__file__).resolve().parent
LATENT_DIM = 32
IMAGE_SIZE = 64


def parse_args():
    parser = argparse.ArgumentParser(description="在 CarRacing 图片上训练一个基础 VAE")
    parser.add_argument(
        "--data",
        type=str,
        default=str(PROJECT_DIR / "data" / "car_racing_frames_1000.npz"),
        help="data_gen.py 生成的 npz 文件",
    )
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=64, help="每个 batch 的图片数")
    parser.add_argument("--lr", type=float, default=0.001, help="Adam 学习率")
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM, help="latent z 的维度")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--save-every", type=int, default=5, help="每多少轮保存一次重建图")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="data/vae_carracing.pth",
        help="模型参数保存路径",
    )
    parser.add_argument(
        "--recon",
        type=str,
        default="data/vae_reconstructions.png",
        help="重建对比图保存路径",
    )
    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs 必须大于 0")
    if args.batch_size <= 0:
        parser.error("--batch-size 必须大于 0")
    if args.lr <= 0:
        parser.error("--lr 必须大于 0")
    if args.latent_dim <= 0:
        parser.error("--latent-dim 必须大于 0")
    if args.save_every <= 0:
        parser.error("--save-every 必须大于 0")

    return args


def resolve_path(path):
    """相对路径统一相对于 vae.py 所在目录，避免受当前工作目录影响。"""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path


def preprocess_image(image, image_size=IMAGE_SIZE):
    """把一张原始 96x96 图片裁成 84x84，再缩放到 64x64。"""
    # 底部 12 行是黑色 HUD，左右各裁 6 像素让剩下的 96x84 变成正方形。
    image = image[0:84, 6:90, :]

    pil_image = Image.fromarray(image)
    pil_image = pil_image.resize(
        (image_size, image_size),
        Image.Resampling.BILINEAR,
    )
    return np.asarray(pil_image)


def load_images(npz_path, image_size=IMAGE_SIZE):
    """读取 npz，返回 uint8 图片数组，形状 (N, 64, 64, 3)。"""
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(
            f"找不到数据文件：{npz_path}\n"
            "请先运行：python data_gen.py"
        )

    with np.load(npz_path) as data:
        raw_images = data["images"]

    print(f"读取数据 : {npz_path}")
    print(f"原始图片 : shape={raw_images.shape}, dtype={raw_images.dtype}")

    processed = []
    for image in raw_images:
        processed.append(preprocess_image(image, image_size))

    images = np.stack(processed)
    print(f"预处理后 : shape={images.shape}, dtype={images.dtype}")
    print(f"           （已裁掉底部 HUD，并 resize 到 {image_size}x{image_size}）")
    return images


class CarRacingDataset(Dataset):
    """把 uint8 的 HWC 图片转换成 VAE 需要的 float32 CHW 张量。"""

    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]

        # 0..255 -> 0..1
        image = image.astype(np.float32) / 255.0

        # HWC -> CHW，PyTorch 卷积层要求通道在前
        image = np.transpose(image, (2, 0, 1))

        return torch.from_numpy(image)


class Encoder(nn.Module):
    """4 层卷积编码器：64x64x3 -> mu, logvar（各 latent_dim 维）。"""

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()

        # 每一层都使用 kernel=4, stride=2，所以尺寸大约减半：
        # 64 -> 31 -> 14 -> 6 -> 2
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2)

        # 最后一层卷积输出 2x2x256，拉平后是 1024 个数
        self.fc_mu = nn.Linear(in_features=2 * 2 * 256, out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=2 * 2 * 256, out_features=latent_dim)

    def forward(self, x):
        # x: (batch, 3, 64, 64)
        x = F.relu(self.conv1(x))  # (batch, 32, 31, 31)
        x = F.relu(self.conv2(x))  # (batch, 64, 14, 14)
        x = F.relu(self.conv3(x))  # (batch, 128, 6, 6)
        x = F.relu(self.conv4(x))  # (batch, 256, 2, 2)

        x = x.reshape(x.size(0), -1)  # (batch, 1024)

        mu = self.fc_mu(x)            # (batch, latent_dim)
        logvar = self.fc_logvar(x)    # (batch, latent_dim)
        return mu, logvar


class Decoder(nn.Module):
    """4 层反卷积解码器：latent_dim -> 64x64x3。"""

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()

        # 先把 z 映射回 2x2x256，再逐步反卷积放大
        self.fc = nn.Linear(in_features=latent_dim, out_features=2 * 2 * 256)

        # 尺寸变化：2 -> 5 -> 13 -> 30 -> 64
        # 第一层用 padding=1，是为了严格对齐论文 Figure 22 里的 5x5。
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=5,
            stride=2,
            padding=1,
        )
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=5,
            stride=2,
        )
        self.deconv3 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=6,
            stride=2,
        )
        self.deconv4 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=3,
            kernel_size=6,
            stride=2,
        )

    def forward(self, z):
        # z: (batch, latent_dim)
        x = F.relu(self.fc(z))                  # (batch, 1024)
        x = x.reshape(x.size(0), 256, 2, 2)     # (batch, 256, 2, 2)

        x = F.relu(self.deconv1(x))             # (batch, 128, 5, 5)
        x = F.relu(self.deconv2(x))             # (batch, 64, 13, 13)
        x = F.relu(self.deconv3(x))             # (batch, 32, 30, 30)

        # 最后一层用 sigmoid，把输出压到 0..1，和输入图片的范围一致
        x = torch.sigmoid(self.deconv4(x))      # (batch, 3, 64, 64)
        return x


class VAE(nn.Module):
    """把编码器和解码器拼在一起。"""

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def encode(self, x):
        return self.encoder(x)

    def reparameterize(self, mu, logvar):
        """从 N(mu, exp(logvar)) 里采样一个 z。"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar):
    """返回 (总损失, 重建损失, KL 损失)，都是每个样本的平均值。

    重建损失：每个像素的平方误差，先对一张图的所有像素求和，再对 batch 平均。
    KL 损失：KL(N(mu, sigma^2) || N(0, 1)) 的标准闭式解。
    """
    batch_size = x.size(0)

    recon_loss = F.mse_loss(recon, x, reduction="sum") / batch_size
    kl_loss = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp()) / batch_size

    total_loss = recon_loss + kl_loss
    return total_loss, recon_loss, kl_loss


def tensor_to_image(tensor):
    """把 (3, 64, 64) 的 float 张量转成 (64, 64, 3) 的 uint8 图片。"""
    image = tensor.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = np.clip(image, 0.0, 1.0)
    image = (image * 255.0).astype(np.uint8)
    return image


def save_reconstructions(model, images, device, path, n=8):
    """保存上排原图、下排重建图的对比图。"""
    model.eval()

    # 从数据集里均匀取 n 张，避免每次都看同样位置的图
    indices = np.linspace(0, len(images) - 1, n).astype(int)
    selected = images[indices].astype(np.float32) / 255.0

    # (n, 64, 64, 3) -> (n, 3, 64, 64)
    batch = torch.from_numpy(selected).permute(0, 3, 1, 2).to(device)

    with torch.no_grad():
        recon, _, _ = model(batch)

    originals = []
    recons = []
    for i in range(n):
        originals.append(tensor_to_image(batch[i]))
        recons.append(tensor_to_image(recon[i]))

    top = np.concatenate(originals, axis=1)
    bottom = np.concatenate(recons, axis=1)

    # 中间加一条白线，方便区分原图和重建图
    separator = np.full((4, top.shape[1], 3), 255, dtype=np.uint8)
    grid = np.concatenate([top, separator, bottom], axis=0)

    Image.fromarray(grid).save(path)
    print(f"重建图已保存：{path}")


def train(args):
    # 固定随机种子，保证每次训练结果大致可复现
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备 : {device}")
    if device.type == "cuda":
        print(f"GPU      : {torch.cuda.get_device_name(0)}")

    data_path = resolve_path(args.data)
    images = load_images(data_path)

    dataset = CarRacingDataset(images)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Windows 下用 0 最稳
    )

    model = VAE(args.latent_dim).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"latent z : {args.latent_dim} 维")
    print(f"参数量   : {num_params:,}")
    print()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    checkpoint_path = resolve_path(args.checkpoint)
    recon_path = resolve_path(args.recon)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    recon_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()

        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0

        for batch in dataloader:
            x = batch.to(device)

            optimizer.zero_grad()

            recon, mu, logvar = model(x)
            loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar)

            loss.backward()
            optimizer.step()

            # loss 是每个样本的平均值，乘回 batch 大小再累加
            total_loss += loss.item() * x.size(0)
            total_recon += recon_loss.item() * x.size(0)
            total_kl += kl_loss.item() * x.size(0)

        num_images = len(dataset)
        avg_loss = total_loss / num_images
        avg_recon = total_recon / num_images
        avg_kl = total_kl / num_images

        print(
            f"epoch {epoch + 1:3d}/{args.epochs} | "
            f"loss={avg_loss:9.3f} | "
            f"recon={avg_recon:9.3f} | "
            f"kl={avg_kl:7.3f}"
        )

        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            save_reconstructions(model, images, device, recon_path)

    torch.save(model.state_dict(), checkpoint_path)
    print()
    print(f"训练完成，模型已保存：{checkpoint_path}")
    print(f"重建对比图：{recon_path}")


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
