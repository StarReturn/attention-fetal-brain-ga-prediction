"""
论文 Fig.1 风格可视化: (a) pred-vs-true 回归散点; (b) 改良 Bland-Altman 图
=================================================================
用途: 验证论文报告的"指纹现象"—— 34 周前预测偏高 / 34 周后逐渐偏低
(论文 Discussion: "model appears to underestimate gestational age
beyond 34 weeks", 归因于晚孕期样本稀少 + 脑回化增速放缓)。

(b) 图红线 = 局部均值曲线(滑动窗口 15 点, 对齐论文 local piecewise
regression 的 window=15 / 95% overlap 设置), 虚线竖线 = 34 周分界。

用法: python plot_ba.py pred_multiplane_n3_s42.csv   # 输出同名 _fig.png
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

csv = sys.argv[1] if len(sys.argv) > 1 else "pred_multiplane_n3_s42.csv"
out = csv.replace(".csv", "_fig.png")
df = pd.read_csv(csv).sort_values("true").reset_index(drop=True)
y = df["true"].values / 7.0   # 天 -> 周
p = df["pred"].values / 7.0
r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
mae = np.abs(y - p).mean() * 7


def piecewise(x, v, win=15, step=1):
    """局部均值(滑动窗口 win, 步长 step; 论文 95% 重叠 ~ step 取 1)。"""
    xs, vs = [], []
    for i in range(0, max(len(v) - win + 1, 1), step):
        xs.append(x[i : i + win].mean())
        vs.append(v[i : i + win].mean())
    return np.array(xs), np.array(vs)


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# (a) 回归散点: 对角虚线=理想, 蓝线=最小二乘拟合
ax = axes[0]
ax.scatter(y, p, s=14, alpha=0.6, color="#1f77b4")
lim = [min(y.min(), p.min()) - 1, max(y.max(), p.max()) + 1]
ax.plot(lim, lim, "k--", lw=1, label="ideal")
k, b = np.polyfit(y, p, 1)
xx = np.linspace(*lim, 50)
ax.plot(xx, k * xx + b, color="dodgerblue", lw=1.5, label=f"fit (R2={r2:.3f})")
ax.set_xlabel("true GA (weeks)")
ax.set_ylabel("predicted GA (weeks)")
ax.set_title(f"(a) Regression  MAE={mae:.1f} d")
ax.legend()

# (b) 改良 Bland-Altman: x=真实孕周, y=预测误差(周)
ax = axes[1]
diff = p - y
ax.scatter(y, diff, s=14, alpha=0.6, color="#2ca02c")
bx, bv = piecewise(y, diff)
ax.plot(bx, bv, color="crimson", lw=2, label="local mean (win=15)")
ax.axhline(0, color="k", ls="--", lw=1)
ax.axvline(34, color="gray", ls=":", lw=1)
ax.text(34.1, ax.get_ylim()[1] * 0.85, "34 wk", color="gray")
ax.set_xlabel("true GA (weeks)")
ax.set_ylabel("predicted - true (weeks)")
ax.set_title("(b) Modified Bland-Altman")
ax.legend()
m34 = diff[y > 34].mean() if (y > 34).any() else float("nan")
m34b = diff[y <= 34].mean()
ax.text(0.02, 0.03, f"mean diff <=34wk: {m34b:+.2f} wk | >34wk: {m34:+.2f} wk",
        transform=ax.transAxes, fontsize=9, bbox=dict(fc="white", alpha=0.8, ec="none"))
plt.tight_layout()
plt.savefig(out, dpi=150)
print("saved", out, f"R2={r2:.3f} MAE={mae:.2f}d  diff<=34wk={m34b:+.2f}wk diff>34wk={m34:+.2f}wk")
