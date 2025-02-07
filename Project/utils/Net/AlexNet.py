import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
# from Attention import SelfAttention2D,CBAM

class AlexNet(nn.Module):
    def __init__(
            self, 
            num_classes=1000,
            input_channels=3,
            hidden_channels=[96,256,384,384,256],
            classifier_channels=4096
        ):
        super(AlexNet, self).__init__()
        self.input_channels=input_channels
        self.hidden_channels=hidden_channels
        self.current_conv_hidden_channel=0
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels[0], kernel_size=11, stride=4),
            nn.ReLU(inplace=True),#原地relu，节省内存
            nn.MaxPool2d(kernel_size=3, stride=2),#h/2
            #ksp=512,第二层
            nn.Conv2d(hidden_channels[0],hidden_channels[1],kernel_size=5,stride=1,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            #ksp=311,第三层,1卷2激活3池化
            nn.Conv2d(hidden_channels[1],hidden_channels[2],kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            #ksp=311,第四层
            nn.Conv2d(hidden_channels[2],hidden_channels[3],kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            #ksp=311,第五层
            nn.Conv2d(hidden_channels[3],hidden_channels[4],kernel_size=3,stride=1,padding=1),
                     #ksp=311,不改变输出特征图尺寸
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        #一层算子，一层激活，一层池化/batchnorm/dropout
        self.classifier=nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),#除batch维度全部通道池化为一个标量
            nn.Flatten(),#把batch维度都拉平#输出为(batch,channels)
            nn.BatchNorm1d(hidden_channels[-1]),#把batch维度都拉平#输出为(batch,channels)
            nn.Linear(hidden_channels[-1],classifier_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(classifier_channels,classifier_channels),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(classifier_channels),#把batch维度都拉平#输出为(batch,channels)
            nn.Linear(classifier_channels,num_classes)
            #loss是交叉熵则不需要对输出进行softmax
        )
    def forward(self,x):
        x=self.feature_extractor(x)
        x=self.classifier(x)
        return x
if __name__=="__main__":
    import sys
    sys.path.append("..")
    from metrics import ModelMeasurer
    model=AlexNet(
        num_classes=1000,
        input_channels=3,
        hidden_channels=[96, 256, 384, 384, 256]
    )
    measurer=ModelMeasurer(model)
    parameters_num=measurer.get_parameters_num()
    inference_time=measurer.get_inference_time(input_shape=(4, 3, 128, 128))
    # measurer.print_parameters_num_by_layer()
    print(f"参数数量：{parameters_num},推理时间：{inference_time} ms")
    print(model)