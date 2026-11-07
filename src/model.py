"""
模型定义：双分支 attention 引导的 ResNet-50 孕周回归网络
=================================================================
复现自: Shen et al., "Attention-guided deep learning for gestational age
        prediction using fetal brain MRI", Scientific Reports 12:1408 (2022).

核心思想（论文 Fig 2）:
  1. 全局分支 (global branch): 整张子宫 MRI 切片 -> ResNet-50 -> 回归 GA
  2. 从全局分支最后一个卷积层 (layer4, 2048 通道) 提取 attention 热图:
       H(x,y) = max_k |f_k(x,y)|          # 逐像素取通道绝对值最大激活
       上采样到输入分辨率 -> 归一化到 [0,1]  # 论文称 truncated ReLU
       乘以中心 2D 高斯先验               # 论文: "脑通常位于图像中心"
  3. 局部分支 (local branch): 热图 mask 逐元素乘原图 -> 抑制母体背景
     (胎盘/子宫壁/羊水) -> ResNet-50 -> 回归 GA
  4. 最终预测 = 两分支输出的平均; 两分支 MSE 共同最小化, 端到端训练
     (mask 归一化全程可微, 不需要任何手工脑分割)

复现时论文未明确、由本实现自行确定的细节（已在 README 详述）:
  - 3-slice 输入时 conv1 预训练权重的通道扩展方式: RGB 均值复制到 3N 通道
  - "truncated ReLU" 实现为 min-max 归一化 + clamp(0,1), 全程可微
  - 高斯先验 sigma = 96 px (论文未给值, 经验取图像短边的 ~0.43)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


def _make_backbone(n_slices: int):
    """构建单个 ResNet-50 回归骨干。

    关键点:
    1. 必须加载 ImageNet 预训练权重 (weights=ResNet50_Weights.IMAGENET1K_V1)。
       本数据集仅 741 例, 从零训练必然失败 (GPT 版复现 R² 为负的根因之一)。
    2. 输入通道: 每张切片是灰度图复制成 3 通道, N 张切片堆叠 -> 3N 通道。
       N=1 时直接用原始 RGB 权重; N>1 时把 RGB 权重在通道维求均值后复制 N
       份 (再除以 N 保持响应幅度不变), 这是论文 "parameterized to
       accommodate different numbers of image slices" 的一种合理实现,
       论文未给出具体做法。
    3. fc 头替换为单输出线性回归, 小 std 初始化避免初始预测离均值太远。
    """
    net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    if n_slices != 1:
        w = net.conv1.weight.data                      # (64, 3, 7, 7)
        w = w.mean(dim=1, keepdim=True).repeat(1, 3 * n_slices, 1, 1) / n_slices
        conv1 = nn.Conv2d(3 * n_slices, 64, kernel_size=7, stride=2, padding=3, bias=False)
        conv1.weight.data = w
        net.conv1 = conv1
    net.fc = nn.Linear(net.fc.in_features, 1)
    nn.init.normal_(net.fc.weight, std=0.01)
    nn.init.zeros_(net.fc.bias)
    return net


class _Features(nn.Module):
    """ResNet-50 截断版: backbone 到 layer4 为止。

    forward 同时返回:
      h      -- layer4 输出特征图 (B, 2048, 7, 7), 供全局分支提取 attention
      logits -- 池化 + fc 后的回归输出 (B, 1)
    """

    def __init__(self, n_slices: int):
        super().__init__()
        net = _make_backbone(n_slices)
        self.conv1, self.bn1, self.relu, self.maxpool = net.conv1, net.bn1, net.relu, net.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = (
            net.layer1, net.layer2, net.layer3, net.layer4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = net.fc

    def forward(self, x):
        h = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        h = self.layer4(self.layer3(self.layer2(self.layer1(h))))
        logits = self.fc(self.pool(h).flatten(1))
        return h, logits


class DualBranchNet(nn.Module):
    """论文的双分支结构: 全局分支 + attention 引导的局部分支。

    gaussian_sigma: 中心高斯先验的标准差(像素)。论文只说 "multiply a 2D
    Gaussian mask", 未给 sigma; 本复现取 96px, 敏感性可扫参(64/96/128)。
    """

    def __init__(self, n_slices: int = 1, img_size: int = 224, gaussian_sigma: float = 96.0):
        super().__init__()
        self.global_net = _Features(n_slices)   # 看整图
        self.local_net = _Features(n_slices)    # 看 attention 加权后的脑区图
        # 中心高斯先验: G(x,y) = exp(-((x-c)^2+(y-c)^2)/(2*sigma^2)), 注册为
        # buffer(不参与训练), 推理时同样生效
        ax = torch.arange(img_size).float() - img_size / 2 + 0.5
        yy, xx = torch.meshgrid(ax, ax, indexing="ij")
        g = torch.exp(-(xx**2 + yy**2) / (2 * gaussian_sigma**2))
        self.register_buffer("gaussian_prior", g.view(1, 1, img_size, img_size))

    def attention_mask(self, feat):
        """从全局分支特征图生成可微 attention mask, 论文 Methods 节公式。

        feat: (B, 2048, 7, 7) -> mask: (B, 1, 224, 224), 取值 [0, 1]

        步骤(全部可微, 保证端到端训练):
          1. 逐像素通道最大绝对激活  H = max_k |f_k|     (论文公式)
          2. 双线性上采样到输入分辨率 224x224
          3. 逐样本 min-max 归一化 + clamp(0,1)          (论文 "truncated ReLU
             to normalize the heatmap to [0,1]" 的实现; 分母加 eps 防除零)
          4. 乘中心高斯先验, 强化中心区域、抑制边缘噪声
        """
        h = feat.abs().amax(dim=1, keepdim=True)
        h = F.interpolate(h, size=self.gaussian_prior.shape[-2:], mode="bilinear",
                          align_corners=False)
        b = h.shape[0]
        mn = h.flatten(1).min(dim=1).values.view(b, 1, 1, 1)
        mx = h.flatten(1).max(dim=1).values.view(b, 1, 1, 1)
        h = torch.clamp((h - mn) / (mx - mn + 1e-6), 0.0, 1.0)
        return h * self.gaussian_prior

    def forward(self, x):
        g_feat, g_out = self.global_net(x)      # 全局分支前向
        mask = self.attention_mask(g_feat)      # 由全局特征生成 attention
        _, l_out = self.local_net(x * mask)     # 局部分支: mask 加权后的图像
        return g_out, l_out, mask               # mask 返回值用于可视化/调试
