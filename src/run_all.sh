#!/bin/bash
# =====================================================================
# 批处理: 论文 Table 1 消融补全 + 多平面 1-slice + 多 seed 方差实验
# 顺序执行 9 个训练 run(单卡 RTX 4090 约 100 分钟), 全部日志落盘。
# 用法: bash run_all.sh   (在 src/ 目录下, data-root 按需修改)
# =====================================================================
cd "$(dirname "$0")"
D=${DATA_ROOT:-../data/images}

# --- Table 1 消融: 双分支 attention, 各平面 x 切片数 ---
python3 train.py --data-root $D --mode dual --plane axial    --n-slices 1 > run_d_a1.log 2>&1
python3 train.py --data-root $D --mode dual --plane coronal  --n-slices 1 > run_d_c1.log 2>&1
python3 train.py --data-root $D --mode dual --plane sagittal --n-slices 1 > run_d_s1.log 2>&1
python3 train.py --data-root $D --mode dual --plane axial    --n-slices 3 > run_d_a3.log 2>&1
python3 train.py --data-root $D --mode dual --plane sagittal --n-slices 3 > run_d_s3.log 2>&1
python3 train.py --data-root $D --mode dual --plane coronal  --n-slices 5 > run_d_c5.log 2>&1
# (coronal 3-slice 已单独跑过, 见 run_d_c3.log)

# --- 多平面 1-slice (论文: R2=0.935 / MAE=7.3d) ---
python3 multiplane.py --data-root $D --n-slices 1 > run_mp1.log 2>&1

# --- 多平面 3-slice 换 seed 43/44, 与 seed 42 组成 3-seed 方差统计 ---
python3 multiplane.py --data-root $D --n-slices 3 --seed 43 > run_mp3_s43.log 2>&1
python3 multiplane.py --data-root $D --n-slices 3 --seed 44 > run_mp3_s44.log 2>&1

echo ALL-DONE
