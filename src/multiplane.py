"""
多平面融合训练脚本(论文最优模型)
=================================================================
论文方法: "trained a multi-plane model by minimizing the total MSE loss
involving axial, coronal, and sagittal planes ... After convergence,
prediction outputs from each plane are then averaged."

即: 三个平面各一套完整的双分支网络 -> 联合最小化三平面 MSE 之和 ->
推理时三平面预测取平均。论文最优结果 R2=0.945 / MAE=6.7d / CCC=0.970。

用法:
  python multiplane.py --data-root ../data/images --n-slices 3
  python multiplane.py --n-slices 3 --seed 43   # 多种子测方差

显存技巧: "total MSE = 三平面 MSE 之和"的梯度 == 逐平面分别 backward
的梯度之和(autograd 线性性), 因此采用逐平面 forward+backward 的梯度
累积实现, 显存峰值 = 单平面, 而优化轨迹与联合 backward 完全等价。
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

PLANES = ["axial", "coronal", "sagittal"]


def ccc(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    sy, sp = y.std(), p.std()
    return 2 * np.corrcoef(y, p)[0, 1] * sy * sp / (sy**2 + sp**2 + (y.mean() - p.mean()) ** 2 + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="../data/images")
    ap.add_argument("--n-slices", type=int, default=3)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--sigma", type=float, default=96.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda"

    # ---------- 数据: 与 train.py 相同的划分(seed 相同则划分一致) ----------
    labels = load_labels(args.data_root)
    cases = [int(c) for c in labels["patient_id"].tolist()
             if os.path.isdir(os.path.join(args.data_root, str(c)))]
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(cases))
    n = len(cases)
    tr = [cases[i] for i in perm[: int(0.7 * n)]]
    te = [cases[i] for i in perm[int(0.8 * n) :]]
    print(f"[data] {len(cases)} cases | train {len(tr)} / test {len(te)}", flush=True)

    # 每平面独立的 DataLoader(独立 shuffle, 逐 iteration 对齐取 batch)
    dls_tr, dls_te = [], []
    for pl in PLANES:
        dls_tr.append(DataLoader(FetalMRIDataset(args.data_root, tr, labels, pl, args.n_slices, True),
                                 batch_size=args.batch, shuffle=True, num_workers=6,
                                 drop_last=True, pin_memory=True, persistent_workers=True))
        dls_te.append(DataLoader(FetalMRIDataset(args.data_root, te, labels, pl, args.n_slices, False),
                                 batch_size=args.batch, num_workers=3))

    y_tr = np.array([float(labels.loc[c, "ga_days"]) for c in tr])
    y_mean, y_std = y_tr.mean(), y_tr.std()

    nets = [DualBranchNet(args.n_slices, gaussian_sigma=args.sigma).to(dev) for _ in PLANES]
    opt = torch.optim.Adam([p for net in nets for p in net.parameters()], lr=args.lr)

    # ---------- 联合训练: 梯度累积等价于 total MSE backward ----------
    it, log = 0, [0.0, 0]
    while it < args.iters:
        for batches in zip(*dls_tr):
            if it >= args.iters:
                break
            for net, (x, y) in zip(nets, batches):
                x = x.to(dev, non_blocking=True)
                yn = ((y - y_mean) / y_std).float().to(dev, non_blocking=True)
                g_out, l_out, _ = net(x)
                loss = F.mse_loss(g_out.flatten(), yn) + F.mse_loss(l_out.flatten(), yn)
                loss.backward()          # 梯度累积, 不 step
                log[0] += loss.item()
                log[1] += 1
            opt.step()                   # 三平面梯度都累积完后再更新
            opt.zero_grad(set_to_none=True)
            it += 1
            if it % 100 == 0:
                print(f"[{it}/{args.iters}] total-mse(norm)={log[0]/log[1]:.4f}", flush=True)
                log = [0.0, 0]

    # ---------- 评测: 三平面(全局+局部分支均值)再取平面均值 ----------
    preds, gts = np.zeros(len(te)), np.array([float(labels.loc[c, "ga_days"]) for c in te])
    with torch.no_grad():
        offs = 0
        for batches in zip(*dls_te):
            bs = len(batches[0][1])
            acc = np.zeros(bs)
            for net, (x, _) in zip(nets, batches):
                g_out, l_out, _ = net(x.to(dev))
                out = ((g_out + l_out) / 2).flatten() * y_std + y_mean  # 分支平均+反归一化
                acc += out.cpu().numpy()
            preds[offs : offs + bs] = acc / len(PLANES)                 # 平面平均
            offs += bs
    r2 = 1 - ((gts - preds) ** 2).sum() / ((gts - gts.mean()) ** 2).sum()
    mae = np.abs(gts - preds).mean()
    print(f"\n[RESULT] multiplane n={args.n_slices} R2={r2:.3f} MAE={mae:.1f} days "
          f"CCC={ccc(gts, preds):.3f}", flush=True)
    out = f"pred_multiplane_n{args.n_slices}_s{args.seed}.csv"
    pd.DataFrame({"true": gts, "pred": preds}).to_csv(out, index=False)
    print("saved", out, flush=True)
    for i, pl in enumerate(PLANES):   # 保存三平面权重(共 ~600MB)
        torch.save(nets[i].state_dict(), f"net_{pl}_n{args.n_slices}_s{args.seed}.pt")


if __name__ == "__main__":
    main()
