import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    '''
        通道注意力，根据输入对特征图的每个通道进行动态加权,并对每个通道的所有位置乘本通道的权重
        参数:
        input_channels: 输入通道数
        channels_reduction: 通道缩放因子，默认为16，即将输入通道数缩小为原来的1/16
        independent_weight: 是否需要独立权重，默认为True，即每个通道的权重独立地计算，不需要softmax
    '''
    def __init__(self, input_channels, channels_reduction=16,independent_weight=True):
        super().__init__()
        self.hidden_channels = max(1, input_channels // channels_reduction)
        self.channels_attention_by_avg=nn.Sequential( #用全局平均池化提取每个通道的特征
            nn.AdaptiveAvgPool2d((1, 1)),  # 将每个通道全局平均池化为一个标量（像素）
            nn.Flatten(),  # 展平为 [batch_size, input_channels]
            nn.Linear(input_channels, self.hidden_channels, bias=False),
            nn.BatchNorm1d(self.hidden_channels),
            nn.ReLU(),
            nn.Linear(self.hidden_channels, input_channels, bias=False),
        )
        self.channels_attention_by_max = nn.Sequential(#用全局最大池化提取每个通道的特征
            nn.AdaptiveMaxPool2d((1, 1)),  # 将每个通道全局全局池化为一个标量（像素）
            nn.Flatten(),  # 展平为 [batch_size, input_channels]
            nn.Linear(input_channels, self.hidden_channels, bias=False),
            nn.BatchNorm1d(self.hidden_channels),
            nn.ReLU(),
            nn.Linear(self.hidden_channels, input_channels, bias=False),
        )
        self.activation = nn.Sigmoid() if independent_weight else nn.Softmax(dim=1) 
    def forward(self, x):
        batch_size, channels, h, w = x.size()
        avg_attention = self.channels_attention_by_avg(x)  # [b, c]
        max_attention = self.channels_attention_by_max(x)  # [b, c]
        attention = avg_attention + max_attention#对表示通道的两个特征进行融合
        attention = self.activation(attention) # 如果不需要独立权重，则使用softmax进行归一化
        attention = attention.view(batch_size, channels, 1, 1)  # 升维度 [b, c, 1, 1]
        return x * attention  # 逐元素相乘，对每个通道的所有位置都乘这个通道对应的权重

class SpatialAttention(nn.Module):
    '''
        空间注意力，根据输入对特征图的每个位置进行动态加权,并对每个位置的所有通道乘本位置的权重
        参数:
        spatial_conv_kernel_size: 空间卷积核大小，默认为7，卷积核大小为7x7
    '''
    def __init__(self, spatial_conv_kernel_size=7):
        super().__init__()
        assert spatial_conv_kernel_size % 2 == 1, "Kernel size must be odd."
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(
                in_channels=2,  # 输入通道数，2每个位置对各通道进行特征统计的整合特征图，包括[mean,max]
                out_channels=1,  # 输出通道数，1是因为输出是每个位置的权重
                kernel_size=spatial_conv_kernel_size,  # 卷积核大小
                padding=spatial_conv_kernel_size // 2,  # 填充，使得输出和输入尺寸相同
                bias=False,#位置加权，最后会被sigmod归一化到[0,1]不需要偏置
            ),
            nn.BatchNorm2d(1),#对输出特征图进行batchnorm
            nn.Sigmoid()#独立归一化权重
        )
    def forward(self, x):
        batch_size, channels, h, w = x.size()
        #计算每个位置对应所有通道的特征信息,keepdim=True表示输出的维度与输入相同，除计算的维度，不改变其他的输入维度
        channels_mean_of_position = x.mean(dim=1, keepdim=True)#计算每个位置各通道的平均值，[b,1,h,w]
        channels_max_of_position = x.max(dim=1, keepdim=True)[0]#计算每个位置各通道的最大值，[b,1,h,w]
        channels_feature_map=torch.cat(
            [channels_mean_of_position,channels_max_of_position],
            dim=1  #在第二个维度合并两个特征图[b,2,h,w]
        )
        #使用卷积层+sigmod对各个位置计算权重，[b,1,h,w]
        spatial_attention = self.spatial_attention(channels_feature_map)
        return x*spatial_attention

class CBAM(nn.Module):
    '''
        CBAM模块，结合了通道注意力和空间注意力，对输入进行通道和空间维度的注意力
        参数:
        input_channels: 输入通道数
        channels_reduction: 通道缩放因子，默认为16，即将输入通道数缩小为原来的1/16
        spatial_conv_kernel_size: 空间卷积核大小，默认为7，卷积核大小为7x7
    '''
    def __init__(self, input_channels, channels_reduction=16, spatial_conv_kernel_size=7,gamma=1):
        super().__init__()
        self.gamma=nn.Parameter(torch.ones(1)*gamma)#缩放注意力的因子
        self.channel_attention = ChannelAttention(input_channels, channels_reduction)
        self.spatial_attention = SpatialAttention(spatial_conv_kernel_size)
    def forward(self, x):
        identity_layer=x
        x = self.channel_attention(x)#先通道注意力,对各通道加权
        x = self.spatial_attention(x)#再空间注意力对每个位置加权
        return x*self.gamma + identity_layer

class SelfAttention1D(nn.Module):
    def __init__(self, embedding_dim):
        #[b,N,embedding_dim]，一个序列有多个元素（词）词向量的维度是embedding_dim,N是序列长
        super().__init__()
        self.query_linear=nn.Linear(embedding_dim,embedding_dim)#同等维度进行映射
        self.key_linear=nn.Linear(embedding_dim,embedding_dim)
        self.value_linear=nn.Linear(embedding_dim,embedding_dim)
        self.softmax=nn.Softmax(dim=-1)#其他词j对该词i的分布归一，在最后一个维度归一化（表示i词的所有权重分数）
        self.gamma=nn.Parameter(torch.ones(1))#输出缩放因子
    def forward(self,x):
        b,N,D=x.size()
        delta=1/torch.sqrt(torch.tensor(D))#权重缩放因子，防止权重过大，与词向量维度相关
        # nn.Linear 层会对每个元素(词向量)(即每一行)单独进行全连接操作，而不需要对整个 N 进行展平。
        Q=self.query_linear(x)#[b,N,D] nn.Linear会对N个词向量批量全连接
        K=self.key_linear(x)
        V=self.value_linear(x)
        KT=K.permute(0,2,1)#转置，[b,N,D]=>[b,D,N]
        scores=torch.bmm(Q,KT)*delta #[b,N,D]·[b,D,N]=[b,N,N] 实现每个词与所有词的相似度
        attention=self.softmax(scores)#最终得到词与词的注意力权重
        return torch.bmm(attention,V)*self.gamma + x#[b,N,N]·[b,N,D]=[b,N,D] 得到加权的输出
    


class SelfAttention2D(nn.Module):
    '''
        对[B,C,H,W]图像的每个像素计算与其他像素的相关性，得到权重矩阵
    '''
    def __init__(self, input_channels,channels_reduction=8):
        super().__init__()
        #QK只是用来计算不同位置的注意力权重，不需要完全保留原始信息，低通道数计算 会更快
        #1*1卷积，stride=1不改变特征图尺寸
        self.query_conv=nn.Conv2d(input_channels,input_channels//channels_reduction,kernel_size=1)
        self.key_conv=nn.Conv2d(input_channels,input_channels//channels_reduction,kernel_size=1)
        #V表示原始特征信息，不能 降低维度
        self.value_conv=nn.Conv2d(input_channels,input_channels,kernel_size=1)
        self.gamma=nn.Parameter(torch.zeros(1))#登记网络参数,对最终输出进行动态缩放
        
        self.softmax=nn.Softmax(dim=-1)#对每个像素j对该像素i的权重进行归一化
    def forward(self,x):
        B,C,H,W=x.size()
        delta=1/torch.sqrt(torch.tensor(H*W))#权重缩放因子，防止权重过大，与像素个数相关
        #1*1卷积，将特征图映射到低维度，方便计算
        Q=self.query_conv(x).view(B,-1,H*W)#展平为[b,c,h*w]运用相关性公式
        K=self.key_conv(x).view(B,-1,H*W)
        V=self.value_conv(x).view(B,-1,H*W)
        #Q=>[B,D,N],N=H*W,D为降维后的通道
        #K=>[B,D,N],N=H*W
        #V=>[B,C,N],N=H*W
        #为了实现N个像素可以自己互相计算相似度,需要对齐维度为[B,N,D]·[B,D,N]=[B,N,N]
        QT=Q.permute(0,2,1)#转置除batch的其他维度
        scores = torch.bmm(QT,K)*delta #batch级别的矩阵乘法[B,N,D]·[B,D,N]=[B,N,N]=[B,H*W,H*W]，即每个像素与其他像素的相关性
        attention = self.softmax(scores)# [B,N,N]，非独立相关性
        output=torch.bmm(V,attention).view(B,C,H,W)#[B,C,N]·[B,N,N] =[B,C,N]=[B,C,H*W]->[B,C,H,W]
        return output*self.gamma + x  #本质是一种残差连接,用可学习的gamma系数控制，对权重进行缩放
   
      