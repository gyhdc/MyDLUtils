import math
import timm
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Attention import SelfAttention2D,ChannelAttention,SpatialAttention,CBAM
class AutoCNN(nn.Module):
    '''
        自动计算卷积层和池化层的特征图尺寸，只需要计算输入输出通道数
        参数：
            input_channels:输入通道数
            input_size:输入图片尺寸
            num_classes:分类数目
            hidden_channels_size_1:第一个卷积核的输出通道数
            hidden_channels_size_2:第二个卷积核的输出通道数
            mlp_hidden_size:mlp隐藏层数目
    '''
    def __init__(self, 
                 input_channels=3,
                 input_size=(28,28),
                 num_classes=10, 
                 hidden_channels_size_1=64,
                 hidden_channels_size_2=128,
                 hidden_channels_size_3=256,
                 mlp_hidden_size=512
        ):
        super(AutoCNN, self).__init__()
        self.input_channels = input_channels
        self.input_size = input_size

        self.feature_extractor=nn.Sequential(
            nn.Conv2d(input_channels,hidden_channels_size_1,kernel_size=3,stride=1,padding=1),
            CBAM(hidden_channels_size_1),#通道注意力
            #(h+2*p-k)/s+1
            nn.BatchNorm2d(hidden_channels_size_1),#2d对2维图进行batchnorm
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),#
            nn.Conv2d(hidden_channels_size_1,hidden_channels_size_2,kernel_size=3,stride=1,padding=1),
            SelfAttention2D(hidden_channels_size_2),#自注意力
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2,stride=2),#(14+2*0-2)/2+1=7
            nn.Conv2d(hidden_channels_size_2,hidden_channels_size_3,kernel_size=3,stride=1,padding=1),
            CBAM(hidden_channels_size_3),#CBAM
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.final_H,self.final_W=self.compute_output_size()
        # print(self.final_H,self.final_W)
        self.classifier=nn.Sequential(
            nn.Linear(
                in_features=hidden_channels_size_3*self.final_H*self.final_W,#将图像展平成特征向量
                out_features=mlp_hidden_size
            ),
           
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden_size, num_classes),
           
            nn.Softmax()  # 添加 softmax 操作,十分类得分归一化为概率
        )
    def forward(self,x):
        x=self.feature_extractor(x)#进行特征提取，输出一个特征图，16通道
        x=x.view(x.size(0),-1)#展平一个batch的每一个特征图，view固定batch维度，后面全部展平
        x=self.classifier(x)#对特征向量进行分类
        return x
    def compute_size(self,input_size,kernel_size=3,stride=1,padding=0):
        if not isinstance(kernel_size,int):
            kernel_size=kernel_size[0]
        if not isinstance(stride,int):
            stride=stride[0]
        if not isinstance(padding,int):
            padding=padding[0]
        if isinstance(input_size,int):
            return ((input_size+2*padding-kernel_size)/stride)+1
        h,w=input_size
        H=((h+2*padding-kernel_size)/stride)+1
        W=((w+2*padding-kernel_size)/stride)+1

        return int(H),int(W) 

    def compute_output_size(self):
        '''自动计算网络每层的输出尺寸'''
        H,W=self.input_size
        layers=[layer for layer in self.feature_extractor]
        for layer in layers:
            if isinstance(layer,nn.MaxPool2d) or isinstance(layer,nn.AvgPool2d) or isinstance(layer,nn.Conv2d):
                H,W=self.compute_size((H,W),layer.kernel_size,layer.stride,layer.padding)
        return H,W