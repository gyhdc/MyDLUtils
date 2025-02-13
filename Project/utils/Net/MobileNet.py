import torch
import torch.nn as nn
import torchvision
class DepthSeparableConv2d(nn.Module):
    '''
    深度可分离卷积，包含深度卷积 + 逐点卷积。
    - 深度卷积：对每个输入通道进行一次卷积操作，不改变通道数（通过 groups 参数实现）。
    - 逐点卷积：采用 1x1 卷积提升通道数并融合通道信息。
    '''
    def __init__(self, input_channels, output_channels, kernel_size, stride=1):
        super().__init__()
        padding = kernel_size // 2  # 保持特征图尺寸不变
        
        # 深度卷积：每个卷积核大小为 KxK，且 groups=input_channels 意味着每个通道单独卷积
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
            nn.ReLU(inplace=True)
        )
        # 逐点卷积：使用1x1卷积提升通道数和融合通道特征
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=1,
                stride=1,
                bias=False
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

class MobileNetV1(nn.Module):
    '''
        MobileNetV1 网络结构，包含初始传统卷积层、深度可分离卷积层和分类器。
        参数：
            input_channels: 输入通道数，默认为 3。
            num_classes: 分类数，默认为 1000。
            gamma: 通道缩放系数，用于轻量化网络，默认为 1。
            initial_channels: 初始卷积核的输出通道数，默认为 32。
            initial_kernel_size: 初始卷积核的大小，默认为 3。
            initial_stride: 初始卷积核的步长，默认为 2。
            DSC_outchannels_and_strides_cfg: 深度可分离卷积核的输出通道数和步长配置，
    '''
    def __init__(
            self,
            input_channels=3,
            num_classes=1000,
            gamma=1,
            initial_channels=32,
            initial_kernel_size=3,
            initial_stride=2,
            DSC_outchannels_and_strides_cfg= [
                (64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2),
                (512, 1), (512, 1), (512, 1), (512, 1), (512, 1), (1024, 2), (1024, 1)
            ],
            DSC_kernel_size=3,  # 深度可分离卷积核大小
            classifier_hidden_size=1024
    ):
        super().__init__()
        self.gamma = gamma  # 通道缩放系数，用于轻量化网络
        self.num_classes = num_classes  # 保存分类数
        
        self.initial_channels = initial_channels
        self.DSC_outchannels_and_strides_cfg = DSC_outchannels_and_strides_cfg
        self.DSC_kernel_size = DSC_kernel_size
        
        # 初始传统卷积层
        self.initial_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=int(initial_channels * self.gamma),  # 缩放后的通道数
                kernel_size=initial_kernel_size,
                stride=initial_stride,
                padding=initial_kernel_size // 2,
                bias=False
            ),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(inplace=True)
        )
        
        # 构建多层深度可分离卷积层
        self.DSC_layers = self._make_layers()
        
        # 分类器：全局平均池化 + Flatten + 全连接层
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 输出形状 [B, C, 1, 1]
            nn.Flatten(),                  # 转换为 [B, C]
            nn.Linear(classifier_hidden_size, self.num_classes)
        )
        
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.DSC_layers(x)
        x = self.classifier(x)
        return x
    
    def _make_layers(self):
        '''
        构建深度可分离卷积层，每一层根据配置逐步构建，输入和输出通道自动衔接。
        '''
        layers = []
        # 初始卷积后的输出通道数
        scaled_in = int(self.initial_channels * self.gamma)
        # 遍历配置列表，配置为 (输出通道, stride)
        for out_channels, stride in self.DSC_outchannels_and_strides_cfg:
            scaled_out = int(out_channels * self.gamma)
            layers.append(
                DepthSeparableConv2d(
                    input_channels=scaled_in,
                    output_channels=scaled_out,
                    kernel_size=self.DSC_kernel_size,
                    stride=stride
                )#尺寸根据stride决定，stride大于2则下采样
            )
            scaled_in = scaled_out  # 更新下一层的输入通道数
        return nn.Sequential(*layers)

# 测试网络结构
if __name__ == "__main__":
    model = MobileNetV1(num_classes=1000)
    pre=torchvision.models.mobilenet_v2(pretrained=False)
    import sys
    sys.path.append("..")
    from metrics import ModelMeasurer
    m1=ModelMeasurer(model)
    m2=ModelMeasurer(pre)
    m1.simply_check_model(input_shape=(4, 3, 224, 224))
    m2.simply_check_model(input_shape=(4, 3, 224, 224))