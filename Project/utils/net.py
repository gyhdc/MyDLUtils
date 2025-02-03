import math
import timm
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F


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
                 hidden_channels_size_1=128,
                 hidden_channels_size_2=64,
                 mlp_hidden_size=256
        ):
        super(AutoCNN, self).__init__()
        self.input_channels = input_channels
        self.input_size = input_size

        self.feature_extractor=nn.Sequential(
            nn.Conv2d(input_channels,hidden_channels_size_1,kernel_size=3,stride=1,padding=1),
            #(h+2*p-k)/s+1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),#
            nn.Conv2d(hidden_channels_size_1,hidden_channels_size_2,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),#(14+2*0-2)/2+1=7

        )
        self.final_H,self.final_W=self.compute_output_size()
        # print(self.final_H,self.final_W)
        self.classifier=nn.Sequential(
            nn.Linear(
                in_features=hidden_channels_size_2*self.final_H*self.final_W,#将图像展平成特征向量
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

class CustomResNet(nn.Module):
    def __init__(self, num_classes=2,hidden_size=256):
        super(CustomResNet, self).__init__()
        self.num_classes = num_classes
        # 加载预训练的ResNet-50模型
        self.resnet_model = models.resnet50(pretrained=True)
        self.num_features = self.resnet_model.fc.in_features
        self.hidden_size = hidden_size
        
        # 获取ResNet-50的特征提取部分（骨干网络）
        self.backbone = nn.Sequential(*list(self.resnet_model.children())[:-2])
        
        # 自定义前馈神经网络
        self.custom_network = nn.Sequential(
            nn.Linear(self.num_features, self.hidden_size),  # 假设ResNet-50的输出特征维度为2048
            nn.BatchNorm1d(self.hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_size, self.num_classes),  # num_classes是你的任务的类别数量
            nn.Softmax() 
        )
    
    def forward(self, x):
        # 使用ResNet-50的特征提取部分
        features = self.backbone(x)
        
        # 全局平均池化
        pooled_features = nn.functional.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
        
        # 将全局平均池化后的特征传递给自定义前馈神经网络
        output = self.custom_network(pooled_features)
        return output

class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes=2, hidden_size=256):
        
        super(CustomEfficientNet, self).__init__()
        # Load pre-trained EfficientNet-B0 model
        self.effnet = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        self.num_features = self.effnet.num_features
        self.hidden_size = hidden_size
        
        # Custom classifier for binary classification
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, num_classes)
        )
    def forward(self, x):
        # Extract features using EfficientNet
        features = self.effnet(x)
        # Pass the features through the classifier
        output = self.classifier(features)
        return output






class CustomEfficientNetV2(nn.Module):
    def __init__(self, num_classes=2, hidden_size=256):
        super(CustomEfficientNetV2, self).__init__()
        # Load pre-trained EfficientNetV2-Small model
        self.effnetv2 = timm.create_model('tf_efficientnetv2_s', pretrained=True, num_classes=0)  # 使用 EfficientNetV2
        
        self.num_features = self.effnetv2.num_features
        self.hidden_size = hidden_size
        
        # Custom classifier for binary classification
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_size, num_classes),
            nn.Softmax() 
        )

    def forward(self, x):
        # Extract features using EfficientNetV2
        features = self.effnetv2(x)
        
        # Pass the features through the classifier
        output = self.classifier(features)
        
        return output



class CustomEfficientNet_b1(nn.Module):
    def __init__(self, num_classes=2, hidden_size=256):
        super(CustomEfficientNet_b1, self).__init__()
        
        # Load pre-trained EfficientNet-B0 model
        self.effnet = timm.create_model('efficientnet_b1', pretrained=True, num_classes=0)  
        
  
        self.num_features = self.effnet.num_features
        
        # Custom classifier for binary classification
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        # Extract features using EfficientNet
        features = self.effnet(x)
        
        # Pass the features through the classifier
        output = self.classifier(features)
        return output
    
class CustomEfficientNet_b5(nn.Module):
    def __init__(self, num_classes=2, hidden_size=256):
        super(CustomEfficientNet_b5, self).__init__()
        
        # Load pre-trained EfficientNet-B0 model
        self.effnet = timm.create_model('efficientnet_b5', pretrained=True, num_classes=0)  
        
  
        self.num_features = self.effnet.num_features
        
        # Custom classifier for binary classification
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        # Extract features using EfficientNet
        features = self.effnet(x)
        
        # Pass the features through the classifier
        output = self.classifier(features)
        
        return output