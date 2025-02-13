import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthSeparableConv2d(nn.Module):
    '''
    深度可分离卷积，包含深度卷积 + 逐点卷积。
    - 深度卷积：对每个输入通道进行一次卷积操作，不改变输出通道数（通过 groups 参数实现）。
    - 逐点卷积：采用 1x1 卷积提升通道数并融合通道信息。
    '''
    def __init__(
            self,
            input_channels,
            output_channels,
            kernel_size,
            stride=1
    ):
        super().__init__()
        padding = kernel_size//2 # 保持特征图尺寸不变
        self.depthwise_conv=nn.Sequential(# 深度卷积，逐通道卷积
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=input_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=input_channels, # 深度卷积，每个通道一个卷积核
                bias=False
            ),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True)
        )
        self.pointwise_conv=nn.Sequential(# 逐点卷积，恢复通道数
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x=self.depthwise_conv(x)
        x=self.pointwise_conv(x)
        return x
class InvertedResidualBlock(nn.Module):
    '''
        扩张：先1*1卷积扩张中间通道数，
        深度可分离卷积层：再深度可分离卷积
        收缩 ：最后1*1卷积 恢复目标输出通道数
    '''
    def __init__(
            self,
            input_channels,
            output_channels,
            stride,
            expansion_ratio
    ):
        super().__init__()
        self.DSC_kernel_size=3 # 深度可分离卷积，默认3*3卷积大小
        # 1. 1*1卷积核拓展中间输出通道数，提升非线性
        self.expansion_ratio = expansion_ratio
        self.expansion_channels=int(input_channels*expansion_ratio)#中间层通道数的扩张
        self.expand_layer=nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=self.expansion_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(self.expansion_channels),
            nn.ReLU(inplace=True)
        )

        # 2.深度可分离卷积层，进一步提取特征
        self.DSC_layer=nn.Sequential(
            DepthSeparableConv2d(
                input_channels=self.expansion_channels,#不改变输出通道数，且逐通道进行
                output_channels=self.expansion_channels,
                kernel_size=self.DSC_kernel_size,
                stride=1,
            ),
            nn.BatchNorm2d(self.expansion_channels),
            nn.ReLU(inplace=True)
        )

        # 3.压缩，1*1卷积将扩张通道压回输出通道数
        self.project_layer=nn.Sequential(
            nn.Conv2d(
                in_channels=self.expansion_channels,
                out_channels=output_channels,# 压缩回目标通道数
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

        # 4.残差连接，如果输入输出通道数相同，且步长为1，则直接相加，否则采用1*1卷积核进行升维或降维
        self.use_residual_conv=(input_channels==output_channels) and (stride==1)
        self.project_input_channels=nn.Sequential(# 调整输入通道数进行残差链接
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(output_channels),
        )
        
    def forward(self,x):
        identity_layer=x
        x=self.expand_layer(x) # 提升通道数
        x=self.DSC_layer(x) # 深度可分离卷积层，进一步提取特征
        x=self.project_layer(x) #压缩回目标通道数
        if self.use_residual_conv:# 如果输入通道上书输出通道数相同
            x=x+identity_layer
        else:
            x=self.project_input_channels(identity_layer)+x
        return x
class MobileNetV2(nn.Module):
    def __init__(
            self,
            num_classes=1000,
            input_channels=3,
            expansion_ratio=6,
            layers=[1,2,3,4,3,3,1],
            layers_channels=[16,24,32,64,96,160,320],
            layers_stride=[1,2,2,2,1,2,1],
            layers_expansion_ratio=[1,6,6,6,6,6,6],
            
    ):
        super().__init__()
        self.initial_conv_layer=nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.blocks = nn.Sequential(
            InvertedResidualBlock(32, 16, 1, 1),
            InvertedResidualBlock(16, 24, 2, 6),
            InvertedResidualBlock(24, 32, 2, 6),
            InvertedResidualBlock(32, 64, 2, 6),
            InvertedResidualBlock(64, 96, 1, 6),
            InvertedResidualBlock(96, 160, 2, 6),
            InvertedResidualBlock(160, 320, 1, 6)
        )
        self.final_conv_layer=nn.Sequential(# 分类前的卷积，进行特征整合
            nn.Conv2d(
                in_channels=320,
                out_channels=1280,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True)
        )
        self.classifier=nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),# -> [B,C,1,1]
            nn.Flatten(start_dim=1,end_dim=3),# 除batch全部展平为向量
            nn.Linear(
                1280,
                num_classes
            )
        )
    def forward(self,x):
        x=self.initial_conv_layer(x)
        x=self.blocks(x)
        x=self.final_conv_layer(x)
        x=self.classifier(x)
        return x
if __name__=='__main__':
    import torchvision
    model = MobileNetV2(num_classes=1000).to("cuda")
    pre=torchvision.models.mobilenet_v2(pretrained=False)
    import sys
    sys.path.append("..")
    from metrics import ModelMeasurer
    m1=ModelMeasurer(model)
    m2=ModelMeasurer(pre)
    m1.simply_check_model(input_shape=(4, 3, 224, 224))
    m2.simply_check_model(input_shape=(4, 3, 224, 224))