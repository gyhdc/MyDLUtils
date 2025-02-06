import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

class VGGNet(nn.Module):
    '''
        VGGNet: VGGNet卷积网络，只采用3*3的kernelsize的卷积
    '''
    def __init__(self, input_channels=3,num_classes=10,config="D",classifier_hidden_size=[4096,1024]):
        #M为最大池化层，数字为该层卷积输出通道数
        
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.config = self.get_config(config)
        print(self.config)
        self.current_input_channels=self.input_channels#经过每层的输出通道数
        self.classifier_hidden_size=classifier_hidden_size#分类器的全连接隐藏层参数
        self.feature_extractor=self._make_layers()
        assert self.current_input_channels==self.config[-2]#确保最后一层输出通道数与config中定义的输出通道数相同
        self.classifier=nn.Sequential(
            #先归一化/dropout再激活
            nn.AdaptiveAvgPool2d((1,1)),#压成[batch_size,channels,1,1]，池化也是一种卷积运算，也会有内部协变量偏移
            nn.Flatten(start_dim=1),#保留batch_size维度，从1开始展平，每个特征图变为一个向量
            nn.BatchNorm1d(self.current_input_channels),#向量的尺寸，整个batch所有特征对全batch进行归一化
            nn.ReLU(inplace=True),
            #mlp分类器
            nn.Linear(self.current_input_channels,self.classifier_hidden_size[0]),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(self.classifier_hidden_size[0],self.classifier_hidden_size[1]),
            nn.BatchNorm1d(self.classifier_hidden_size[1]),
            nn.Dropout(0.25),
            nn.ReLU(inplace=True),
            nn.Linear(self.classifier_hidden_size[1],num_classes)#回归分类数
        )
    def forward(self,x):
        x=self.feature_extractor(x)
        x=self.classifier(x)
        return x
    def _make_layers(self,):
        
        layers=[]
        for output_channels in self.config:
            if output_channels=="M":#最大池化层
                layers.append(nn.MaxPool2d(kernel_size=2,stride=2))#h/2
            else:#卷积层
                layers.extend(
                    [
                        #ksp=311，不改变特征图尺寸
                        nn.Conv2d(self.current_input_channels,output_channels,kernel_size=3,stride=1,padding=1),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(output_channels)#对每个通道进行一个batch的归一化
                    ]
                )
                self.current_input_channels=output_channels#更新下一层的通道数
        return nn.Sequential(*layers)
    def get_config(self,config):
        if isinstance(config,str) :
            self.VGGNet_CONFIGS = {#输出通道数或者池化层
                'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],  # VGG11
                'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],  # VGG13
                'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],  # VGG16
                'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],  # VGG19
                "MINE":[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 1024, 'M'],  # VGG16,最后1024通道
            }
            return self.VGGNet_CONFIGS.get(config)
        else:
            return config
                    
if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from metrics import ModelMeasurer
    model=VGGNet(
        input_channels=3,
        num_classes=1000,
        config="D",
        classifier_hidden_size=[4096,1024]
    )
    # measurer=ModelMeasurer(model)
    
    # measurer.simply_check_model(input_shape=(4, 3, 128, 128))
    print(model)
    # measurer.print_parameters_num_by_layer()