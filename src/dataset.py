"""
数据集: Stanford 胎儿脑 MRI 孕周回归数据集
=================================================================
数据来源(公开, CC BY-NC 3.0):
  Stanford Digital Repository  https://purl.stanford.edu/sf714wg0636
  下载文件 fetal-brain.zip (~3.1 GB), 解压后即本脚本期望的目录结构。

实际目录结构(与论文描述略有出入, 以探查结果为准):
  images/                       # 解压根目录
    labels.xlsx                 # 741 行标签表
    <patient_id>/               # 1 ~ 741, 每例一个文件夹
      IM-{series:04d}-{slice:04d}.jpg
      # 注意: 每例的三个平面序列(axial/coronal/sagittal)的切片文件
      # 混放在同一文件夹里, 靠文件名中的 series 号区分!

labels.xlsx 列含义:
  patient_id              病例号(=文件夹名), 741 例, GA 133~273 天
  ga_days / ga_wks        孕周真值(天 / 周), 真值由一孕早期超声推算
  ax_series / cor_series / sag_series    三个平面对应的序列号
  ax_midslice / cor_midslice / sag_midslice
                          三个平面的中间切片文件名(论文: "middle slice
                          contains the largest fetal brain area")

预处理(严格按论文 Materials and methods):
  1. 取中间切片(1-slice)或中间±1 张(3-slice)等
  2. 训练: 随机裁剪 224x224 / 测试: 中心裁剪 224x224
     (论文 "randomly cropped to 224x224"; 测试裁剪方式论文未说明,
      按惯例取中心裁剪, 避免随机性引入评测噪声)
  3. 逐样本标准化 (x - mean) / std —— 论文原话 "normalized using sample
     mean and standard deviation", 是每张图自己的均值方差, 不是
     ImageNet 统计量! (GPT 版复现易错点)
  4. 灰度复制到 3 通道(匹配 ImageNet 预训练 conv1), N 张切片堆成 3N 通道
"""

import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

PLANE_PREFIX = {"axial": "ax", "coronal": "cor", "sagittal": "sag"}


def load_labels(root):
    """读 labels.xlsx 并做类型规整, 以 patient_id 为索引返回 DataFrame。"""
    df = pd.read_excel(os.path.join(root, "labels.xlsx"))
    df.columns = [str(c).strip() for c in df.columns]
    df["patient_id"] = df["patient_id"].astype(int)
    df["ga_days"] = df["ga_days"].astype(float)
    for p in ("ax", "cor", "sag"):
        df[f"{p}_series"] = df[f"{p}_series"].astype(int)
    return df.set_index("patient_id", drop=False)


class FetalMRIDataset(Dataset):
    """按平面 + 切片数加载样本。

    参数:
      root      images/ 目录
      cases     参与的病例号列表(已按 train/val/test 划分好)
      labels    load_labels() 的返回值
      plane     "axial" | "coronal" | "sagittal"
      n_slices  1 / 3 / 5, 以 midslice 为中心取邻近切片
      train     True: 随机裁剪(数据增强); False: 中心裁剪(确定式评测)
    """

    def __init__(self, root, cases, labels, plane="coronal", n_slices=1, train=True, img_size=224):
        self.root = root
        self.cases = list(cases)
        self.labels = labels
        self.prefix = PLANE_PREFIX[plane]
        self.n_slices = n_slices
        self.train = train
        self.img_size = img_size

    def __len__(self):
        return len(self.cases)

    def _plane_files(self, case):
        """列出该病例在指定平面上的全部切片(按文件名排序=按层位排序)。

        同一文件夹混放三个序列, 用 labels 里的 series 号构造前缀
        "IM-{series:04d}-" 过滤出目标平面的文件; midslice 直接取
        labels 提供的精确文件名(若缺失则退化为文件列表中位)。
        """
        row = self.labels.loc[case]
        pre = f"IM-{int(row[f'{self.prefix}_series']):04d}-"
        files = sorted(f for f in os.listdir(os.path.join(self.root, str(case))) if f.startswith(pre))
        if not files:
            raise FileNotFoundError(f"no files with prefix {pre} in case {case}")
        mid_name = str(row[f"{self.prefix}_midslice"])
        mid = files.index(mid_name) if mid_name in files else len(files) // 2
        return files, mid

    def _load(self, case, fname):
        """读单张 JPG -> 裁剪 224x224 -> 逐样本标准化, 返回 float32 数组。"""
        x = np.asarray(Image.open(os.path.join(self.root, str(case), fname)).convert("L"), dtype=np.float32)
        s = self.img_size
        h, w = x.shape
        if min(h, w) < s:
            raise ValueError(f"image {h}x{w} smaller than {s}")
        if self.train:  # 随机位置裁剪 = 论文的随机裁剪增强
            top, left = np.random.randint(0, h - s + 1), np.random.randint(0, w - s + 1)
        else:           # 评测固定中心裁剪, 保证结果可复现
            top, left = (h - s) // 2, (w - s) // 2
        x = x[top : top + s, left : left + s]
        x = (x - x.mean()) / (x.std() + 1e-6)   # 逐样本标准化(论文原话)
        return x

    def __getitem__(self, i):
        case = self.cases[i]
        files, mid = self._plane_files(case)
        # 以 midslice 为中心取 n_slices 张; 边界处钳位(序列不够长时重复端层)
        half = self.n_slices // 2
        idx = [min(len(files) - 1, max(0, mid + d)) for d in range(-half, half + 1)]
        stack = np.stack([self._load(case, files[j]) for j in idx])       # (N,224,224)
        # 每张灰度切片复制成 3 通道后拼接 -> (3N, 224, 224)
        x = torch.from_numpy(stack).unsqueeze(1).repeat(1, 3, 1, 1).flatten(0, 1)
        y = float(self.labels.loc[case, "ga_days"])   # 回归目标: GA 天数
        return x, y
