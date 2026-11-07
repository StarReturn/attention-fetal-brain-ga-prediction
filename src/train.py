"""
单平面训练脚本
=================================================================
用法示例(在 src/ 目录下):
  python train.py --data-root ../data/images --mode global --plane coronal --n-slices 1
  python train.py --data-root ../data/images --mode dual   --plane coronal --n-slices 3

训练协议(论文 Materials and methods 原设置, 未改动):
  - 优化器 Adam, lr = 1e-4, batch = 50, 共 2000 iterations (注意是
    iteration 不是 epoch; 518 个训练样本 x 2000 iter / 50 batch ≈ 193 epoch)
  - 损失: MSE。dual 模式下全局/局部两支 MSE 之和
  - 划分: 70% train / 10% val / 20% test, 随机划分(论文未公开具体划分
    与随机种子, 本复现固定 numpy seed=42 以保证自身可复现)

本复现新增(论文未提及、但对收敛至关重要, 见 README"改动说明"):
  - 回归目标 z-score 标准化(train 集统计), 评测时反变换回天数。
    不做归一化时 MSE 初值 ~4 万, 2000 iter 内无法收敛 —— 这是 GPT 版
    复现 R² 为负的根因之二。
  - 评测: model.eval() + center crop + torch.no_grad()

指标: R^2 / MAE(天) / Lin's CCC, 全部在反变换后的原始天数上计算。
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import FetalMRIDataset, load_labels
from model import DualBranchNet


def ccc(y, p):
    """Lin's concordance correlation coefficient(论文使用的的一致性指标)。"""
    y, p = np.asarray(y, float), np.asarray(p, float)
    sy, sp = y.std(), p.std()
    return 2 * np.corrcoef(y, p)[0, 1] * sy * sp / (sy**2 + sp**2 + (y.mean() - p.mean()) ** 2 + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="../data/images", help="images/ 数据目录")
    ap.add_argument("--plane", default="coronal", choices=["axial", "coronal", "sagittal"])
    ap.add_argument("--n-slices", type=int, default=1)
    ap.add_argument("--mode", default="global", choices=["global", "dual"],
                    help="global=仅全局分支(对照); dual=双分支attention(论文方法)")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--sigma", type=float, default=96.0, help="attention 高斯先验 sigma(px)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="预测 CSV 输出路径")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- 数据与划分 ----------
    labels = load_labels(args.data_root)
    cases = [int(c) for c in labels["patient_id"].tolist()
             if os.path.isdir(os.path.join(args.data_root, str(c)))]
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(cases))
    n = len(cases)
    tr = [cases[i] for i in perm[: int(0.7 * n)]]
    va = [cases[i] for i in perm[int(0.7 * n) : int(0.8 * n)]]   # 本快速复现未用早停
    te = [cases[i] for i in perm[int(0.8 * n) :]]
    print(f"[data] {len(cases)} cases | train {len(tr)} / val {len(va)} / test {len(te)}")

    dl_tr = DataLoader(FetalMRIDataset(args.data_root, tr, labels, args.plane, args.n_slices, True),
                       batch_size=args.batch, shuffle=True, num_workers=8, drop_last=True,
                       pin_memory=True, persistent_workers=True)
    dl_te = DataLoader(FetalMRIDataset(args.data_root, te, labels, args.plane, args.n_slices, False),
                       batch_size=args.batch, num_workers=4)

    # ---------- 目标归一化(关键!) ----------
    y_tr = np.array([float(labels.loc[c, "ga_days"]) for c in tr])
    y_mean, y_std = y_tr.mean(), y_tr.std()
    print(f"[target] GA days mean={y_mean:.1f} std={y_std:.1f}")

    model = DualBranchNet(args.n_slices, gaussian_sigma=args.sigma).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---------- 训练循环(iteration 计数, 论文 2000 iterations) ----------
    it, log = 0, [0.0, 0]
    while it < args.iters:
        for x, y in dl_tr:
            if it >= args.iters:
                break
            x = x.to(dev, non_blocking=True)
            yn = ((y - y_mean) / y_std).float().to(dev, non_blocking=True)  # z-score 目标
            g_out, l_out, _ = model(x)
            loss = F.mse_loss(g_out.flatten(), yn)          # 全局分支 MSE
            if args.mode == "dual":
                loss = loss + F.mse_loss(l_out.flatten(), yn)  # +局部分支 MSE
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            log[0] += loss.item()
            log[1] += 1
            it += 1
            if it % 100 == 0:
                print(f"[{it}/{args.iters}] mse(norm)={log[0]/log[1]:.4f}", flush=True)
                log = [0.0, 0]

    # ---------- 评测(反变换回天数再算指标) ----------
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for x, y in dl_te:
            g_out, l_out, _ = model(x.to(dev))
            # 论文: 最终预测 = 全局/局部分支输出的平均(dual); global 模式仅全局
            out = (g_out + l_out) / 2 if args.mode == "dual" else g_out
            preds += (out.flatten() * y_std + y_mean).cpu().tolist()
            gts += y.tolist()
    preds, gts = np.array(preds), np.array(gts)
    r2 = 1 - ((gts - preds) ** 2).sum() / ((gts - gts.mean()) ** 2).sum()
    mae = np.abs(gts - preds).mean()
    print(f"\n[RESULT] plane={args.plane} n={args.n_slices} mode={args.mode} "
          f"R2={r2:.3f} MAE={mae:.1f} days CCC={ccc(gts, preds):.3f}")

    out = args.out or f"pred_{args.plane}_n{args.n_slices}_{args.mode}.csv"
    pd.DataFrame({"true": gts, "pred": preds}).to_csv(out, index=False)
    print("saved", out)


if __name__ == "__main__":
    main()
