import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthSeparableConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1):
        super().__init__()
        padding = kernel_size // 2  # 保持特征图尺寸不变
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=input_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=input_channels,  # 深度卷积，每个通道一个卷积核
                bias=False
            ),
            nn.BatchNorm2d(input_channels),
            nn.ReLU6(inplace=True)
        )
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=1,  # 提升通道
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU6()
        )

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class MBConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, expansion_ratio):
        super().__init__()
        # 扩展特征图通道
        self.expansion_channels = int(input_channels * expansion_ratio)
        self.expand_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=self.expansion_channels,
                kernel_size=1,  # 1*1卷积扩展通道数
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(self.expansion_channels),
            nn.ReLU6()
        )

        # 深度可分离卷积
        self.depthwise_conv = DepthSeparableConv2d(
            input_channels=self.expansion_channels,
            output_channels=self.expansion_channels,
            kernel_size=kernel_size,
            stride=stride,
        )

        # 压缩通道
        self.project_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=self.expansion_channels,
                out_channels=output_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU6()
        )

        # 残差连接
        self.use_residual_layer = (input_channels == output_channels) and stride == 1

    def forward(self, x):
        identity_layer = x
        x = self.expand_layer(x)
        x = self.depthwise_conv(x)
        x = self.project_layer(x)
        if self.use_residual_layer:  # 输入和特征图统一尺寸则残差连接
            x = identity_layer + x
        return x


class EfficientNetB3(nn.Module):
    def __init__(
            self,
            input_channels=3,
            num_classes=1000,
            alpha=1.4,  # EfficientNet-B3 的宽度因子
            beta=1.8,  # 深度因子
            gamma=1.4,  # 分辨率因子
            stem_out_channels=32,
            stem_kernel_size=3,
            stem_stride=2,
            stem_padding=1,
            layers_kernels=[1, 2, 3, 4, 3, 3, 1],
            layers_channels=[16, 24, 40, 80, 112, 192, 320],
            layers_stride=[1, 2, 2, 2, 1, 2, 1],
            layers_expansion_ratio=[1, 6, 6, 6, 6, 6, 6],
    ):
        super().__init__()

        # 1. stem 初始卷积
        self.stem_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=int(stem_out_channels * alpha),
                kernel_size=stem_kernel_size,
                stride=stem_stride,
                padding=stem_padding,
                bias=False
            ),
            nn.BatchNorm2d(int(stem_out_channels * alpha)),
            nn.ReLU6()
        )

        # 2. 定义 MBConv 模块堆叠，采用复合缩放（Compound Scaling）
        self.blocks = nn.Sequential(
            MBConvBlock(int(32 * alpha), int(16 * alpha), 3, 1, 1),
            MBConvBlock(int(16 * alpha), int(24 * alpha), 3, 2, 6),
            MBConvBlock(int(24 * alpha), int(40 * alpha), 5, 2, 6),
            MBConvBlock(int(40 * alpha), int(80 * alpha), 3, 2, 6),
            MBConvBlock(int(80 * alpha), int(112 * alpha), 5, 1, 6),
            MBConvBlock(int(112 * alpha), int(192 * alpha), 5, 2, 6),
            MBConvBlock(int(192 * alpha), int(320 * alpha), 3, 1, 6)
        )

        # 3. 分类前的卷积，进行特征整合
        self.final_conv_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=int(320 * alpha),
                out_channels=int(1280 * alpha),
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(int(1280 * alpha)),
            nn.ReLU6()
        )

        # 4. 分类层
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # -> [B,C,1,1]
            nn.Flatten(start_dim=1, end_dim=3),  # 除 batch 全部展平为向量
            nn.Linear(int(1280 * alpha), num_classes)
        )

    def forward(self, x):
        x = self.stem_layer(x)
        x = self.blocks(x)
        x = self.final_conv_layer(x)
        x = self.classifier(x)
        return x


# 测试模型
if __name__ == '__main__':
    model = EfficientNetB3(num_classes=1000, alpha=1.4, beta=1.8, gamma=1.4)
    input_tensor = torch.randn(1, 3, 300, 300)  # 输入尺寸适应EfficientNet-B3
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")
