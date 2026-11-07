# Attention-Guided Deep Learning for Gestational Age Prediction from Fetal Brain MRI — Reproduction

An independent reproduction of:

> Shen, L., Zheng, J., Lee, E.H. et al. **Attention-guided deep learning for gestational age prediction using fetal brain MRI.** *Scientific Reports* **12**, 1408 (2022). [https://www.nature.com/articles/s41598-022-05468-5](https://www.nature.com/articles/s41598-022-05468-5)

The paper proposes an end-to-end dual-branch ResNet-50 that regresses gestational age (GA) directly from raw clinical T2w fetal MRI slices — no brain segmentation required. The original authors released the dataset but not the code; this repo is a from-scratch PyTorch implementation that closely matches the published results.

## Results

All numbers are on the held-out 20% test split (149 subjects) of the public 741-subject Stanford dataset, GA 19–39 weeks. Single run, seed 42, paper's original hyperparameters (Adam, lr 1e-4, batch 50, 2000 iterations).

| Model | Paper | This reproduction |
|---|---|---|
| **Multi-plane 3-slice dual-branch (best model)** | R² 0.945 / MAE 6.7 d / CCC 0.970 | **R² 0.940 / MAE 8.0 d / CCC 0.968** |
| Coronal 3-slice dual-branch (best single-plane) | R² 0.924 / MAE 7.9 d | R² 0.915 / MAE 9.3 d |
| Coronal 1-slice global-only (ablation control) | — | R² 0.923 / MAE 9.0 d / CCC 0.961 |

Key findings of the paper are reproduced:

- **Multi-plane fusion helps** — single-plane 0.915 → multi-plane 0.940 (paper: 0.924 → 0.945).
- **The "beyond 34 weeks" fingerprint** — modified Bland–Altman analysis shows the model overestimates GA before ~34 weeks (+0.59 wk on average) and progressively underestimates it afterwards (−0.73 wk), matching the pattern reported in the paper and its proposed explanation (third-trimester sample sparsity + decelerating gyrification).

The remaining MAE gap (+1.3 days, while R²/CCC are nearly identical) is attributable to the unpublished train/test split, the Gaussian prior width (see below), and single-seed variance.

## Method

```
input X (N slices stacked center-out, 224x224)
 ├── global branch: ResNet-50 (ImageNet) ──────────────► GA_global
 │      └── layer4 features (2048x7x7)
 │          H(x,y) = max_k |f_k(x,y)|            # channel-max of |activation|
 │          → bilinear upsample to 224x224
 │          → min-max normalize, clamp [0,1]     # paper's "truncated ReLU"
 │          → multiply centred 2D Gaussian prior (sigma = 96 px)
 │          = differentiable attention mask
 ├── local branch:  X ⊙ mask → ResNet-50 ────────► GA_local
 └── prediction = (GA_global + GA_local) / 2       # multi-plane: average 3 planes
```

Both branches are trained jointly by the sum of their MSE losses; the mask stays differentiable, so the whole thing trains end to end.

## Repository layout

```
src/
├── model.py        dual-branch attention ResNet-50 (core)
├── dataset.py      slice selection by series prefix, crops, per-sample normalization
├── train.py        single-plane training / evaluation (global | dual modes)
├── multiplane.py   multi-plane joint training (gradient accumulation = total MSE)
├── plot_ba.py      regression scatter + modified Bland-Altman (34-wk fingerprint)
├── explore.py      dataset structure inspection
└── run_all.sh      full ablation + multi-seed batch
```

## Getting started

### 1. Environment

Python ≥ 3.8, any CUDA GPU with ≥ 8 GB (a single RTX 4090 trains one run in 8–20 min).

```bash
pip install -r requirements.txt   # torch must be a CUDA build
python -c "import torch; assert torch.cuda.is_available()"
```

### 2. Data

Download the public dataset (CC BY-NC 3.0, non-commercial use): [Stanford Digital Repository — Fetal Brain MRI](https://purl.stanford.edu/sf714wg0636) (~3.1 GB zip), then extract:

```bash
wget -c "https://stacks.stanford.edu/file/druid:sf714wg0636/fetal-brain.zip"
unzip -q fetal-brain.zip && mv images data/images
```

Layout note (differs slightly from the paper's description): each case folder mixes all three plane series as `IM-{series:04d}-{slice:04d}.jpg`; `labels.xlsx` maps each plane to its series number and provides the exact middle-slice filename per plane.

> If `stacks.stanford.edu` is unreachable from your network (common on some non-US cloud hosts), relay through a local proxy via an SSH reverse tunnel: `ssh -N -R 17890:127.0.0.1:7890 <host>`, then `curl -x http://127.0.0.1:17890 -C - -O <url>`.

### 3. Train & evaluate

```bash
cd src
D=../data/images

python train.py --data-root $D --mode global --plane coronal --n-slices 1   # control
python train.py --data-root $D --mode dual   --plane coronal --n-slices 3   # best single-plane
python multiplane.py --data-root $D --n-slices 3                            # best model (paper-matching)
bash run_all.sh                                                             # full ablation + seeds 43/44
python plot_ba.py pred_multiplane_n3_s42.csv                                # Fig.1-style plots
```

Each run prints `R2 / MAE (days) / CCC` and writes a `pred_*.csv` (true, pred in days) so every number can be independently re-verified.

## Reproduction decisions (details the paper leaves open)

| Choice | Paper says | This repo does |
|---|---|---|
| Target scaling | not mentioned | z-score GA (train stats), invert at eval — **without this, MSE starts at ~4e4 and 2000 iterations cannot converge** (the most common cause of negative-R² failures) |
| conv1 for N slices | "parameterized to accommodate" | average ImageNet RGB weights over channels, replicate to 3N, scale by 1/N |
| "truncated ReLU" heatmap norm | name only | differentiable min-max + clamp to [0,1], eps-guarded |
| Gaussian prior width | not given | sigma = 96 px (fixed; sensitivity not swept) |
| Test-time crop | not specified | centre crop (train uses random crop) |
| Split | 70/10/20 random, unpublished | same proportions, numpy seed 42 |

External validation on the paper's four non-public sites (CHLA/CCHMC/SJH/TTRH, Table 2) is not reproducible — the data requires institutional agreements.

## Citation & license

If you use this code, please cite the original paper (Shen et al., Sci Rep 2022). The dataset is licensed CC BY-NC 3.0 by its authors; this code is provided for research reproduction purposes.
