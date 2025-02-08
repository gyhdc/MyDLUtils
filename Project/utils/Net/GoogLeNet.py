import torch
import torch.nn as nn

class BasicConv2d(nn.Module):
    def __init__(self, input_channels,output_channels,**kwargs):
        super().__init__()
        self.feature_extractor=nn.Sequential(
            nn.Conv2d(input_channels,output_channels,bias=False,**kwargs),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.feature_extractor(x)
class Inception(nn.Module):
    #192, [64, [96, 128], [16, 32], 32]
    '''
        Inception模块，输入特征图尺寸不变，输出特征图尺寸不变
        有四个分支，最后在通道维度将特征图拼接，输出通道数等于四个分支通道数的和
    '''
    def __init__(
            self,
            input_channels,
            BasicBlock=BasicConv2d,
            channels_1x1=[64],#第一个分支，1*1卷积的输出通道数
            channels_3x3=[96,128],
            channels_5x5=[16,32],
            channels_pool=[32]
    ):
        super().__init__()
        '''4个分支均不改变输入特征图尺寸，只有通道数不同，用cat在通道维度融合特征'''
        self.branch_1x1=nn.Sequential(#第一个分支是1*1卷积，用于低级特征提取，降维，增加非线性
            BasicBlock(input_channels,channels_1x1[0],kernel_size=1)
        )
        self.branch_3x3=nn.Sequential(#第二个分支是3*3卷积，用于中低级特征提取，增加非线性
            BasicBlock(input_channels,channels_3x3[0],kernel_size=1),#改变通道数,减少通道数
            BasicBlock(channels_3x3[0],channels_3x3[1],kernel_size=3,stride=1,padding=1)#311不改变特征图尺寸
        )
        self.branch_5x5=nn.Sequential(#第三个分支是5*5卷积，用于中高级特征提取，增加非线性
            BasicBlock(input_channels,channels_5x5[0],kernel_size=1),#降低通道数
            BasicBlock(channels_5x5[0],channels_5x5[1],kernel_size=5,stride=1,padding=2)#512不改变特征图尺寸
        )
        self.branch_pool=nn.Sequential(#第四个分支是池化，用于高级特征提取，减少非线性
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),#不改变特征图尺寸
            BasicBlock(input_channels,channels_pool[0],kernel_size=1)#降低通道数
        )
    def forward(self,x):
        branch_1 = self.branch_1x1(x)
        branch_2 = self.branch_3x3(x)
        branch_3 = self.branch_5x5(x)
        branch_4 = self.branch_pool(x)

        return torch.cat(
            [branch_1,branch_2,branch_3,branch_4],
            dim=1 #在通道维度上cat，通道数等于所有分支通道数相加
        )
class InceptionAuxClassifier(nn.Module):
    '''
        在GoogLeNet中间层进行分类，并和全网络的分类结果一起计算loss然后反向传播\n
        相当于将中间层的梯度信息直接提供给loss，缓解梯度消失，加快浅层权重更新
    '''
    def __init__(self,input_channels,num_classes=1000,AAP_shape=(4,4),MLP_hidden_size=[1024,]):
        super().__init__()
        self.input_channels=input_channels
        self.num_classes=num_classes
        self.channels_after_flatten=input_channels*AAP_shape[0]*AAP_shape[1]
        self.MLP_hidden_size=MLP_hidden_size
        self.AAP_shape=AAP_shape#全局平均池化后特征图尺寸
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(self.AAP_shape),  # 全局平均池化 -> (4,4)
            nn.Flatten(),  # 除 batch 维度外展平
            nn.Linear(self.channels_after_flatten, self.MLP_hidden_size[0]),  # 全连接层
            nn.BatchNorm1d(self.MLP_hidden_size[0]),  # BN 归一化，减少过拟合
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),  # 高 Dropout 防止辅助分类器过拟合
            nn.Linear(self.MLP_hidden_size[0], self.num_classes)  # 输出类别
        )
    def forward(self,x):
        return self.classifier(x)
class GoogLeNet(nn.Module):
    '''
        GoogLeNet网络结构
        参数：
            input_channels (int): 输入通道数。默认为3。
            num_classes (int): 类别数。默认为1000。
            aux_classify (bool): 是否进行辅助分类。默认为True。
            init_weights (bool): 是否进行初始化权重。默认为False。
            BasicBlock (nn.Module): 基础卷积块。默认为BasicConv2d。
            initial_conv_channels (list): 初始卷积层的输出通道数。默认
    '''
    def __init__(
            self,
            input_channels=3,
            num_classes=1000,
            aux_classify=True,
            init_weights=False,
            BasicBlock=BasicConv2d,
            initial_conv_channels=[64,64,192],#初始卷积层的输出通道数
            AAP_shape=(4,4),#全局平均池化后特征图尺寸
    ):
        super().__init__()
        self.aux_classify=aux_classify
        self.num_classes=num_classes
        self.input_channels=input_channels
        self.BasicBlock=BasicBlock
        self.inital_conv_channels=initial_conv_channels
        self.AAP_shape=AAP_shape
        self.use_aux_classifier = self.training and self.aux_classify



        # self.current_input_channels=self.input_channels
        self.inital_conv=nn.Sequential(
            self.BasicBlock(self.input_channels,self.inital_conv_channels[0],kernel_size=7,stride=2,padding=3),#h/2
            nn.MaxPool2d(kernel_size=3,stride=2,padding=0),#h/2
            self.BasicBlock(self.inital_conv_channels[0],self.inital_conv_channels[1],kernel_size=1),
            self.BasicBlock(self.inital_conv_channels[1],self.inital_conv_channels[2],kernel_size=3,stride=1,padding=1),#h*1
            nn.MaxPool2d(kernel_size=3,stride=2,padding=0),#h/2

        )
        self.current_input_channels=self.inital_conv_channels[-1]#获取initial_conv后的通道数
        #Stage 3
        self.feature_extractor_3_1=nn.Sequential(
            Inception(# 3a: [192, 64,[ 96, 128], [16, 32], 32]
                input_channels=self.current_input_channels,
                BasicBlock=self.BasicBlock,
                channels_1x1=[64],
                channels_3x3=[96,128],
                channels_5x5=[16,32],
                channels_pool=[32]
            ),#每个inception的输出通道数是四个分支的输出通道数相加，64+128+32+32=256
            Inception(#3b: [256, 128, [128, 192], [32, 96], 64]
                input_channels=256,
                BasicBlock=self.BasicBlock,
                channels_1x1=[128],
                channels_3x3=[128,192],
                channels_5x5=[32,96],
                channels_pool=[64]
            ),#输出通道数128+192+96+64=480
            nn.MaxPool2d(kernel_size=3,stride=2,padding=0,ceil_mode=True)
        )


        #Stage 4
        self.feature_extractor_4_1=nn.Sequential( 
            Inception( #4a: [480, 192,[ 96, 208], [16, 48], 64)]
                input_channels=480,
                BasicBlock=self.BasicBlock,
                channels_1x1=[192],
                channels_3x3=[96,208],
                channels_5x5=[16,48],
                channels_pool=[64]
            ),#s输出通道数192+208+48+64=512
        )
        self.aux_classifier_1=InceptionAuxClassifier(#aux_classifier_1 在4a后分类,独立除分类结果
            input_channels=512,
            num_classes=self.num_classes,
            AAP_shape=self.AAP_shape,
            MLP_hidden_size=[1024,]
        )
        self.feature_extractor_4_2=nn.Sequential(#Stage 4
            Inception(#4b: [512, 160,[112, 224], [24, 64], 64]
                input_channels=512,
                BasicBlock=self.BasicBlock,
                channels_1x1=[160],
                channels_3x3=[112,224],
                channels_5x5=[24,64],
                channels_pool=[64]
            ),#输出通道数160+224+64+64=512
            Inception(#4c: [512, 128,[128, 256], [24, 64], 64]
                input_channels=512,
                BasicBlock=self.BasicBlock,
                channels_1x1=[128],
                channels_3x3=[128,256],
                channels_5x5=[24,64],
                channels_pool=[64]
            ),#输出通道数128+256+64+64=512
            Inception(#4d: [512, 112,[144, 288], [32, 64], 64]
                input_channels=512,
                BasicBlock=self.BasicBlock,
                channels_1x1=[112],
                channels_3x3=[144,288],
                channels_5x5=[32,64],
                channels_pool=[64]
            ),#输出通道数112+288+64+64=528
        )
        self.aux_classifier_2=InceptionAuxClassifier(#aux_classifier_2 在4d后分类,独立除分类结果
            input_channels=528,
            num_classes=self.num_classes,
            AAP_shape=self.AAP_shape,
            MLP_hidden_size=[1024,]
        )
        self.feature_extractor_4_3=nn.Sequential(#Stage 4
            Inception(#4e: [528, 256,[160, 320], [32, 128], 128]
                input_channels=528,
                BasicBlock=self.BasicBlock,
                channels_1x1=[256],
                channels_3x3=[160,320],
                channels_5x5=[32,128],
                channels_pool=[128]
            ),#输出通道数256+320+128+128=832
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0,ceil_mode=True)
        )

        #Stage 5
        self.feature_extractor_5_1=nn.Sequential(
            Inception(#5a: [832, 256,[160, 320], [32, 128], 128]
                input_channels=832,
                BasicBlock=self.BasicBlock,
                channels_1x1=[256],
                channels_3x3=[160,320],
                channels_5x5=[32,128],
                channels_pool=[128]
            ),#输出通道数256+320+128+128=832
            Inception(#5b: [832, 384,[192, 384], [48, 128], 128]
                input_channels=832,
                BasicBlock=self.BasicBlock,
                channels_1x1=[384],
                channels_3x3=[192,384],
                channels_5x5=[48,128],
                channels_pool=[128]
            )#输出通道数384+384+128+128=1024
        )
        
        self.final_classifier=nn.Sequential(
            nn.AdaptiveAvgPool2d(self.AAP_shape),
            nn.Flatten(),#展平为batchsize维度
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(1024*self.AAP_shape[0]*self.AAP_shape[1],self.num_classes)
        )
    def forward(self,x):
        self.use_aux_classifier = self.training and self.aux_classify
        x=self.inital_conv(x)
        x=self.feature_extractor_3_1(x)#第三层输出
        x=self.feature_extractor_4_1(x)#第四层第一部分，以4a结束
        if self.use_aux_classifier is True:
            #4a结束，辅助分类第一次
            aux_classifier_1_output=self.aux_classifier_1(x)
        x=self.feature_extractor_4_2(x)#第四层第二部分，以4d结束
        if self.use_aux_classifier is True:
            #4d结束，辅助分类第二次
            aux_classifier_2_output=self.aux_classifier_2(x)
        x=self.feature_extractor_4_3(x)
        x=self.feature_extractor_5_1(x)
        x=self.final_classifier(x)
        if self.use_aux_classifier is True:
            return (x,aux_classifier_1_output,aux_classifier_2_output)
        else:
            return x

if __name__ == '__main__':
    import sys
    sys.path.append("..")
    from metrics import ModelMeasurer
    model=GoogLeNet(
        input_channels=3,num_classes=10
    )
    m=ModelMeasurer(model)
    m.simply_check_model(input_shape=(4, 3, 64, 64))
    print(model)

        