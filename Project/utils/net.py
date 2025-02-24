import math
import timm
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F


from .metrics import ModelMeasurer
#自己实现的网络
from .Net.ResNet import ResNet
from .Net.CNN import AutoCNN
from .Net.AlexNet import AlexNet
from .Net.VGGNet import VGGNet
from .Net.GoogLeNet import GoogLeNet
from .Net.Attention import SelfAttention2D,CBAM,SpatialAttention,ChannelAttention
from .Net.MobileNet import MobileNetV1
from .Net.VisionTransformer import VisionTransformer
from .Net.Pretrained import BinaryClassificationMobileNetV3Large
from .Net.EfficientNet import EfficientNetB3
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
if __name__=="__main__":
    model=ResNet()
    measurer=ModelMeasurer(model)
    measurer.get_inference_time()