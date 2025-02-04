import math
import timm
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
class BinaryClassificationMobileNetV3Large(nn.Module):
    def __init__(self,out_size):
        super(BinaryClassificationMobileNetV3Large, self,).__init__()

        # 加载预训练的MobileNetV3 Large模型
        mobilenet = models.mobilenet_v3_large(pretrained=True)

        # 获取MobileNetV3的特征提取部分（骨干网络）
        self.features = mobilenet.features

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(960, 256),  # MobileNetV3 Large最后一层特征的维度为960
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, out_size),
            nn.Softmax()  # 添加 softmax 操作
        )

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])  # 全局平均池化
        x = self.classifier(x)
        return x