import math
import timm
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
#核心，用stride控制是否下采样,ksp,311保持尺寸不变
class BasicBlock(nn.Module):
    '''
        Resnet保证网络的统一性，kernel size通常保持不变为3,
        因为多个3×3卷积可以组合成更大的感受野，同时减少参数数量。
        虽然5×5或7×7卷积核可以实现更大的感受野，但它们的参数更多，计算成本更高。
        由stride控制是否下采样 
        输出尺寸: H=(h+2*p-k)/s+1
        用downsample对不匹配的输入的通道数进行1*1卷积核进行匹配

        参数：
            input_channels:输入通道数
            output_channels:输出通道数
            stride:步长，默认为1，不进行下采样
            downsample:下采样层，默认为Identity Block
            kernel_size:卷积核尺寸，默认为3
            padding:卷积核padding，默认为0
    '''
    expansion = 1  # BasicBlock的输出通道数与输入通道数相同
    def __init__(
        self,
        input_channels,
        output_channels,
        stride=1,#为2则下采样
        downsample=None,#默认为Identity Block
        kernel_size=3,
        padding=1,
    ):
        super().__init__()
        self.downsample=downsample#下采样层
        #每个残差层，只有第一个残差块可能下采样，第一个残差块的第一个卷积可能下采样
        self.feature_extractor=nn.Sequential(
            nn.Conv2d(
                input_channels,output_channels,
                kernel_size=kernel_size,
                stride=stride,#根据stride改变输入
                padding=padding,bias=False
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            #不改变通道数和尺寸的卷积，可以进行特征调整且让输出通道数与输入x对齐。
            nn.Conv2d(#正常k,s,p >3 1 1不改变输入的尺寸，
                output_channels,output_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,bias=False
            ),
            nn.BatchNorm2d(output_channels),
        )
    def forward(self,x):
        identity_input=x
        output=self.feature_extractor(x)#经过包括两个卷积的特征提取层
        if self.downsample is not None:
            identity_input=self.downsample(x)
            #若输入和输出的通道数不对齐，用1*1卷积核对输入通道数进行对齐和下采样
        output=output+identity_input #残差连接,H(x) = F(x) + x,output为残差函数，也是这一层需要拟合的输出
        output=F.relu(output)#在线性残差连接后再进行激活
        return output
    
class ResNet(nn.Module):
    def __init__(
        self,
        layers=[2,2,2,2],#每层的残差块个数
        layers_channels=[64,128,256,512],#每层的输出通道数
        layers_stride=[1,2,2,2],#每层残差块的下采样步长，默认为1，不进行下采样
        basic_block=BasicBlock,#采用什么类型的残差块
        num_classes=1000,
        input_channels=3,
        hidden_channels=64
    ):
        super(ResNet, self).__init__()
        self.hidden_channels=hidden_channels
        self.input_channels=input_channels
        self.now_input_channels=input_channels#每一层变化的时候自动更新当前的输出通道数，作为下一层的输入通道数
        self.layers=layers#每层的残差块个数
        self.layers_channels=layers_channels
        self.layers_stride=layers_stride
         
        self.basic_block=basic_block #调用的残差块类型
        self.initial_block=nn.Sequential(
            #ResNet中最初的7×7大卷积层用于对输入图像进行初步的特征提取和降采样，
            #捕捉较大尺度的纹理和边缘信息，为后续的残差块提供丰富的初始特征表示。
            nn.Conv2d(input_channels,hidden_channels,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)#降采样/2
        )
        self.now_input_channels=hidden_channels#initial_block的输出通道数，作为下一层的输入通道数
        self.Layers=self.get_Layers(stride_list=layers_stride)#获取每一层
        self.classifier=nn.Sequential(#分类器
            nn.AdaptiveAvgPool2d((1,1)),#自适应平均池化，输出尺寸为1*1，池化也是一种类似的卷积计算
            nn.Flatten(),#出batchsize维度都展平·
            nn.BatchNorm1d(self.now_input_channels),#通道数已经扩张
            nn.Linear(
                self.now_input_channels,num_classes
            ),#全连接层
        )
    def forward(self,x):
        x=self.initial_block(x)#初始卷积块，用
        x=self.Layers(x)#残差层
        x=self.classifier(x)#全局池化后展平为向量再分类
        return x
    def get_Layers(
            self,
            stride_list=[1,2,2,2]
        ):#获取resnet每层
        Layers=[]
        #每层的stride，2为下采样，可自己根据layers个数进行定义
        for idx,num_blocks in enumerate(self.layers):
            layer=self._make_layer(
                output_channels=self.layers_channels[idx],#每一层对应的输出通道数
                nums_blocks=num_blocks,
                basic_block=self.basic_block,
                stride=stride_list[idx],
            )
            Layers.append(layer)
        return nn.Sequential(*Layers)#序列化
        
    def _make_layer(
            self,
            output_channels,
            nums_blocks,
            basic_block,
            stride=1,#默认不下采样，大于一为下采样，resnet的kernelsize通常不变化
            padding=1,#卷积核padding，默认为1
    
    ):
        downsample=None
        expansion_output_channels=output_channels*basic_block.expansion#扩展后的输出通道数
        if stride != 1 or self.input_channels != output_channels:
            '''
                stride下采样或者输入输出通道数不相同，则需要1*1卷积核对输入的通道数进行匹配
                以达到多个残差块串联
            '''
            downsample=nn.Sequential(
                nn.Conv2d(self.now_input_channels,expansion_output_channels,kernel_size=1,stride=stride,bias=False),
                #1*1的卷积进行通道数对齐，或进行下采样stride为2
                nn.BatchNorm2d(expansion_output_channels),
                #batchnorm自带偏置对输出进行归一化，使得输出的均值为0，方差为1
            )
        layers=[
            basic_block(
                self.now_input_channels,#当前层的输入通道数
                output_channels,#输出通道数
                stride=stride,#下采样步长
                downsample=downsample,#下采样层
            )
        ]#当resnet这一layer需要下采样的时候(stride>1)就让layer的第一个残差块进行下采样，其他残差块恒等
        self.now_input_channels=expansion_output_channels#经过这一层后，下一层的输入通道数可能发生变化
        for _ in range(1,nums_blocks):
            layers.append(#接下来的残差块都不下采样
                basic_block(
                    self.now_input_channels,#当前层的输入通道数
                    output_channels,#输出通道数
                    stride=1,#不进行下采样
                    downsample=None,#不需要下采样层
     
                )
            )
        return nn.Sequential(*layers)
def resnet18(
        num_classes=1000,
        input_channels=3,
        # hidden_channels=64
):
    return ResNet(
        input_channels=input_channels,
        layers=[2,2,2,2],#每层的残差块个数
        layers_channels=[64,128,256,512],#每层的输出通道数
        layers_stride=[1,2,2,2],#每层残差块的下采样步长，默认为1，不进行下采样
        basic_block=BasicBlock,#采用什么类型的残差块
        num_classes=num_classes,
    )
def resnet34(
        num_classes=1000,
        input_channels=3,
):
    return ResNet(
        input_channels=input_channels,
        layers=[3, 4, 6, 3],  # 每层的残差块个数
        layers_channels=[64, 128, 256, 512],  # 每层的输出通道数
        layers_stride=[1, 2, 2, 2],  # 每层残差块的下采样步长
        basic_block=BasicBlock,  # 采用什么类型的残差块
        num_classes=num_classes,
    )

# import sys
# sys.path.append("..")

if __name__ == "__main__":
    from train_val import validate_model,train_model
    from metrics import ModelMeasurer
    model=resnet18()
    # print(model)
    # print(model)
    input_shape = (4, 3, 256, 256)
    repetitions: int = 300
    unit: int = 1000
    measurer=ModelMeasurer(model)
    cost_time=measurer.get_inference_time()
    parameters=measurer.get_parameters_num()
    print(f"在input_shape={input_shape}下，\n重复计算{repetitions},\n平均模型推理耗时为{cost_time}ms")       
    print(f"模型参数为{parameters}")

