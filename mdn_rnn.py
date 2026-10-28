"""基础 MDN-RNN：在 VAE 的 latent 空间里预测下一帧。

数据来自 data_gen_rnn.py：
    images / actions / rewards / dones / episode_ids / steps / seeds

每一局是连续帧，每一步都有真实动作 a_t。流程：
    1. 用训练好的 VAE 编码器把图片编码成 z（这里取 mu，不采样）；
    2. 把 z 序列和动作序列按时间切成窗口；
    3. LSTM 读入 [z_t, a_t]，输出混合高斯分布 P(z_{t+1} | a_t, z_t, h_t)；
    4. 取混合高斯的均值作为预测的 z_{t+1}；
    5. 把预测的 z 交给 VAE 解码器，得到预测图像。

这是一个“基础版”：
    - 输入是 [z_t, a_t]，和论文的 P(z_{t+1} | a_t, z_t, h_t) 对齐；
    - 没有预测 done / reward；
    - 没有做多步自回归 rollout，只做 teacher forcing 的下一步预测。

运行：
    conda activate world_model
    python mdn_rnn.py

    # 改训练轮数、序列长度
    python mdn_rnn.py --epochs 30 --seq-len 10

输出：
    data/mdn_rnn.pth              MDN-RNN 参数
    data/mdn_rnn_prediction.png   上排真实未来帧，下排预测解码帧
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from vae import VAE, load_images, tensor_to_image


PROJECT_DIR = Path(__file__).resolve().parent
LATENT_DIM = 32
HIDDEN_SIZE = 256
NUM_MIXTURES = 5
SEQ_LEN = 10


def parse_args():
    parser = argparse.ArgumentParser(description="训练一个基础 MDN-RNN")
    parser.add_argument(
        "--data",
        type=str,
        default=str(PROJECT_DIR / "data" / "rnn_sequences.npz"),
        help="data_gen_rnn.py 生成的连续序列数据",
    )
    parser.add_argument(
        "--vae-checkpoint",
        type=str,
        default="data/vae_carracing.pth",
        help="训练好的 VAE 参数",
    )
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="每个 batch 的序列数")
    parser.add_argument("--lr", type=float, default=0.001, help="Adam 学习率")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN, help="每条序列的时间步数")
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE, help="LSTM 隐藏维度")
    parser.add_argument("--num-mixtures", type=int, default=NUM_MIXTURES, help="高斯混合个数")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="data/mdn_rnn.pth",
        help="MDN-RNN 参数保存路径",
    )
    parser.add_argument(
        "--prediction",
        type=str,
        default="data/mdn_rnn_prediction.png",
        help="预测对比图保存路径",
    )
    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs 必须大于 0")
    if args.batch_size <= 0:
        parser.error("--batch-size 必须大于 0")
    if args.lr <= 0:
        parser.error("--lr 必须大于 0")
    if args.seq_len <= 0:
        parser.error("--seq-len 必须大于 0")
    if args.hidden_size <= 0:
        parser.error("--hidden-size 必须大于 0")
    if args.num_mixtures <= 0:
        parser.error("--num-mixtures 必须大于 0")

    return args


def resolve_path(path):
    """相对路径统一相对于 mdn_rnn.py 所在目录。"""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path


def load_vae(path, device):
    """加载训练好的 VAE，并切到推理模式。"""
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到 VAE 参数：{path}")

    model = VAE(LATENT_DIM).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"VAE 参数 : {path}")
    return model


def encode_images(vae, images, device, batch_size=128):
    """把 (N,64,64,3) 的 uint8 图片编码成 (N,32) 的 z。

    这里直接取编码器的 mu，不做采样。对 MDN-RNN 来说，
    用确定性的 mu 作为序列输入更稳定。
    """
    vae.eval()
    all_z = []

    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            x = batch.astype(np.float32) / 255.0
            x = torch.from_numpy(x).permute(0, 3, 1, 2).to(device)

            mu, _ = vae.encode(x)
            all_z.append(mu.cpu())

    z = torch.cat(all_z, dim=0).numpy()
    print(f"编码完成 : z.shape={z.shape}")
    return z


def build_sequences(z, actions, episode_ids, steps, seq_len):
    """按 episode 分组，切成 (seq_len + 1) 帧的窗口。

    返回：
        z_sequences      : (S, seq_len+1, 32)
        action_sequences : (S, seq_len, 3)  只取输入步的动作
        index_sequences  : (S, seq_len+1)   对应的原始图片下标
    """
    z_sequences = []
    action_sequences = []
    index_sequences = []

    for episode_id in np.unique(episode_ids):
        indices = np.where(episode_ids == episode_id)[0]
        indices = indices[np.argsort(steps[indices])]

        # 窗口不重叠，步长就是 seq_len
        for start in range(0, len(indices) - seq_len, seq_len):
            window = indices[start:start + seq_len + 1]
            z_sequences.append(z[window])
            action_sequences.append(actions[window[:-1]])
            index_sequences.append(window)

    if len(z_sequences) == 0:
        raise ValueError("没有足够的连续帧，无法构造序列；请减小 --seq-len")

    return (
        np.stack(z_sequences),
        np.stack(action_sequences),
        np.stack(index_sequences),
    )


class SequenceDataset(Dataset):
    """一个样本 = 一个窗口；DataLoader 的 shuffle 只打乱窗口顺序。

    输入：
        z_t      (seq_len, 32)
        a_t      (seq_len, 3)
    目标：
        z_{t+1}  (seq_len, 32)
    """

    def __init__(self, z_sequences, action_sequences):
        self.z_sequences = z_sequences
        self.action_sequences = action_sequences

    def __len__(self):
        return len(self.z_sequences)

    def __getitem__(self, index):
        z_sequence = self.z_sequences[index]       # (T+1, 32)
        action_sequence = self.action_sequences[index]  # (T, 3)

        input_z = z_sequence[:-1]                  # (T, 32)
        target_z = z_sequence[1:]                  # (T, 32)

        return (
            torch.from_numpy(input_z),
            torch.from_numpy(action_sequence),
            torch.from_numpy(target_z),
        )


class MDNRNN(nn.Module):
    """LSTM + 混合高斯输出头。

    LSTM 每个时间步读入 [z_t, a_t]，输出 h_t；
    再用一个线性层把 h_t 变成 K 个高斯分量的参数：
        pi_logits : (B, T, K)
        mu        : (B, T, K, D)
        sigma     : (B, T, K, D)
    """

    def __init__(self, latent_dim=LATENT_DIM, action_dim=3,
                 hidden_size=HIDDEN_SIZE, num_mixtures=NUM_MIXTURES):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.num_mixtures = num_mixtures

        self.lstm = nn.LSTM(
            input_size=latent_dim + action_dim,
            hidden_size=hidden_size,
            batch_first=True,
        )

        # 每个时间步输出：K 个权重 + K 组 mu + K 组 sigma
        output_size = num_mixtures * (1 + 2 * latent_dim)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, z_sequence, action_sequence, hidden=None):
        # z_sequence: (B, T, D)
        # action_sequence: (B, T, A)
        #   B = 一次并行处理的序列条数
        #   T = 每条序列的时间步数
        x = torch.cat([z_sequence, action_sequence], dim=-1)  # (B,T,D+A)

        lstm_out, hidden = self.lstm(x, hidden)               # (B,T,H)
        params = self.fc(lstm_out)                            # (B,T,K*(1+2D))

        batch_size, seq_len, _ = params.shape
        k = self.num_mixtures
        d = self.latent_dim

        pi_logits = params[:, :, :k]                          # (B,T,K)

        mu_start = k
        mu_end = k + k * d
        mu = params[:, :, mu_start:mu_end]
        mu = mu.reshape(batch_size, seq_len, k, d)            # (B,T,K,D)

        sigma_start = mu_end
        sigma_raw = params[:, :, sigma_start:]
        sigma_raw = sigma_raw.reshape(batch_size, seq_len, k, d)
        # softplus 保证 sigma > 0，比直接 exp 更稳定
        sigma = F.softplus(sigma_raw) + 1e-4                  # (B,T,K,D)

        return pi_logits, mu, sigma, hidden


def mdn_loss(pi_logits, mu, sigma, target):
    """混合高斯的负对数似然（NLL）。

    target: (B, T, D)
    mu/sigma: (B, T, K, D)

    注意：连续分布的对数密度可正可负，所以 NLL 也可能是负数。
    训练时只需要看它是否在下降，不要把它当成概率。
    """
    log_pi = F.log_softmax(pi_logits, dim=-1)  # (B,T,K)

    target = target.unsqueeze(2)               # (B,T,1,D)

    # 每个高斯分量的对数密度
    log_prob = -0.5 * ((target - mu) / sigma) ** 2
    log_prob = log_prob - torch.log(sigma)
    log_prob = log_prob - 0.5 * math.log(2.0 * math.pi)
    log_prob = log_prob.sum(dim=-1)            # (B,T,K)，对 D 求和

    # log Σ_k π_k N(z; μ_k, σ_k)
    log_mix = log_pi + log_prob                # (B,T,K)
    nll = -torch.logsumexp(log_mix, dim=-1)    # (B,T)

    return nll.mean()


def mixture_mean(pi_logits, mu):
    """把混合高斯压成一个确定性的 z：Σ_k π_k μ_k。

    这里取均值是为了得到稳定的可视化结果；如果想看“可能的下一帧”，
    也可以按 π 选一个分量，再从 N(mu_k, sigma_k²) 采样。
    """
    pi = F.softmax(pi_logits, dim=-1)          # (B,T,K)
    pi = pi.unsqueeze(-1)                      # (B,T,K,1)
    return (pi * mu).sum(dim=2)                # (B,T,D)


def save_prediction(model, vae, z_sequences, action_sequences,
                    index_sequences, images, device, path):
    """用一条验证序列做下一步预测，并保存真实帧 vs 预测帧。"""
    model.eval()
    vae.eval()

    # 只取一条验证序列，输入 z_0..z_{T-1} 和 a_0..a_{T-1}
    z_sequence = torch.from_numpy(z_sequences[0:1]).to(device)  # (1,T+1,D)
    action_sequence = torch.from_numpy(action_sequences[0:1]).to(device)

    input_z = z_sequence[:, :-1]                                # (1,T,D)
    target_z = z_sequence[:, 1:]                                # (1,T,D)

    with torch.no_grad():
        # 注意：这里的 mu 是 MDN 输出头的分量均值 (1,T,K,D)，
        # 不是 VAE 编码器的 mu (B,D)。
        pi_logits, mu, _, _ = model(input_z, action_sequence)
        predicted_z = mixture_mean(pi_logits, mu)               # (1,T,D)

        # 预测的 z -> 解码器 -> 预测图像
        predicted_images = vae.decoder(predicted_z[0])          # (T,3,64,64)

    # 真实未来帧：原始图片（已经裁过 HUD、resize 到 64x64）
    frame_indices = index_sequences[0][1:]
    actual_images = images[frame_indices]                       # (T,64,64,3)

    # 上排真实，下排预测
    top = np.concatenate(list(actual_images), axis=1)
    bottom = np.concatenate(
        [tensor_to_image(predicted_images[i]) for i in range(len(predicted_images))],
        axis=1,
    )
    separator = np.full((4, top.shape[1], 3), 255, dtype=np.uint8)
    grid = np.concatenate([top, separator, bottom], axis=0)

    Image.fromarray(grid).save(path)

    latent_mse = F.mse_loss(predicted_z, target_z).item()
    print(f"预测图已保存：{path}")
    print(f"latent MSE（预测 z vs 真实 z）：{latent_mse:.6f}")


def train(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备 : {device}")
    if device.type == "cuda":
        print(f"GPU      : {torch.cuda.get_device_name(0)}")

    data_path = resolve_path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"找不到数据文件：{data_path}\n"
            "请先运行：python data_gen_rnn.py"
        )

    # 读取动作和 episode 信息
    with np.load(data_path) as data:
        actions = data["actions"]
        episode_ids = data["episode_ids"]
        steps = data["steps"]

    print(f"数据文件 : {data_path}")
    print(f"帧数     : {len(episode_ids)}")
    print(f"局数     : {len(np.unique(episode_ids))}")
    print(f"动作维度 : {actions.shape[1]}")

    images = load_images(data_path)
    vae = load_vae(args.vae_checkpoint, device)
    z = encode_images(vae, images, device)

    # 用最后一局做验证，其他局训练
    val_episode = int(np.max(episode_ids))
    train_mask = episode_ids != val_episode
    val_mask = episode_ids == val_episode

    train_z, train_actions, _ = build_sequences(
        z[train_mask],
        actions[train_mask],
        episode_ids[train_mask],
        steps[train_mask],
        args.seq_len,
    )
    val_z, val_actions, val_indices = build_sequences(
        z[val_mask],
        actions[val_mask],
        episode_ids[val_mask],
        steps[val_mask],
        args.seq_len,
    )

    print(f"训练序列 : {train_z.shape}")
    print(f"验证序列 : {val_z.shape}（held-out episode={val_episode}）")
    print()

    dataset = SequenceDataset(train_z, train_actions)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,  # 打乱的是序列窗口，不会打乱窗口内部的 T 个时间步
        num_workers=0,
    )

    model = MDNRNN(
        latent_dim=LATENT_DIM,
        action_dim=actions.shape[1],
        hidden_size=args.hidden_size,
        num_mixtures=args.num_mixtures,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"LSTM 隐藏维度 : {args.hidden_size}")
    print(f"高斯分量个数 : {args.num_mixtures}")
    print(f"参数量       : {num_params:,}")
    print()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for input_z, input_actions, target_z in dataloader:
            input_z = input_z.to(device)
            input_actions = input_actions.to(device)
            target_z = target_z.to(device)

            optimizer.zero_grad()

            pi_logits, mu, sigma, _ = model(input_z, input_actions)
            loss = mdn_loss(pi_logits, mu, sigma, target_z)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * input_z.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"epoch {epoch + 1:3d}/{args.epochs} | nll={avg_loss:.4f}")

    checkpoint_path = resolve_path(args.checkpoint)
    prediction_path = resolve_path(args.prediction)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), checkpoint_path)
    print()
    print(f"训练完成，MDN-RNN 已保存：{checkpoint_path}")

    save_prediction(
        model,
        vae,
        val_z,
        val_actions,
        val_indices,
        images,
        device,
        prediction_path,
    )


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
