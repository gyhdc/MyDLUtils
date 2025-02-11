import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# 1. Patch Embedding 模块
# -----------------------------
class PatchEmbedding(nn.Module):
    """
    将输入图像切分为不重叠的小块（patch），并将每个 patch 映射为一个向量（token）。
    这里使用一个卷积层实现：
      - kernel_size = patch_size
      - stride = patch_size
    这样可以直接将图像划分成若干个 patch。
    参数:
      - image_size: 输入图像的大小，默认为 (224, 224)
      - patch_size: 切分 patch 的大小，默认为 (16, 16)
      - input_channels: 输入图像的通道数，默认为 3
      - embed_dim: 映射后的向量维度，默认为 768
      return:embed_patch [B,num_patchs,embed_dim] => [B,N,D]
    """
    def  __init__(
            self,
            image_size=(224,224),
            patch_size=(16,16),
            input_channels=3,
            embed_dim=768
    ):
        super().__init__()
        
        if isinstance(image_size,int):
            image_size=(image_size,image_size)
        if isinstance(patch_size,int):
            patch_size=(patch_size,patch_size)
        assert image_size[0]%patch_size[0]==0 and image_size[1]%patch_size[1]==0,f"image_size {image_size} should be divisible by patch_size {patch_size}"
        self.patch_size=patch_size
        self.num_patches=(image_size[0]//patch_size[0])*(image_size[1]//patch_size[1])#  总的patch数
        
        self.projector=nn.Conv2d(
            in_channels=input_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )#将每个patch映射到高维空间
    def forward(self,x):
        B,C,H,W=x.size()
        '''
            patch相当于一个词
            num_patchs相当于词的个数
        '''
        #划分patch和把patch映射到高维空间
        #x->[B,embed_dim,H_num_patch,W_num_patch] ->[B,embed_dim,num_patches]
        x=self.projector(x).flatten(start_dim=2,end_dim=3)
        x=x.permute(0,2,1)#->[B,num_patches,embed_dim] <=> [B,N,D]
        return x # [B,N,D]

# -----------------------------
# 2. MLP 模块（Feed Forward Network）
# -----------------------------
class MLP(nn.Module):
    """
        Transformer 中的前馈网络，由两个全连接层构成
        中间加激活函数（GELU）和 Dropout。
    """
    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            dropout_rate=0.1
    ):
        super().__init__()
        self.mlp=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.GELU(),#缓解梯度弥散，神经元死亡
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim,output_dim),
            nn.Dropout(dropout_rate)
        )
    def  forward(self,x ):
        return self.mlp(x)
# -----------------------------
# 3. 多头自注意力模块
# -----------------------------
class MultiHeadAttention(nn.Module):
    """
    多头自注意力模块：
      - 将输入线性映射为 Query、Key 和 Value。
      - 对每个 head 计算点积注意力（带缩放）。
      - 将各个 head 的输出拼接(全连接映射)起来，再做一次线性变换输出。
    """
    def __init__(
            self,
            embed_dim,
            num_heads=12,
            dropout_rate=0.1
    ):
        super().__init__()
        assert embed_dim%num_heads==0,f"embed_dim {embed_dim} should be divisible by num_heads {num_heads}"
        self.num_heads=num_heads
        self.head_dim=embed_dim//num_heads # 将词嵌入维度降维，不同空间进行自注意力

        self.query_linear=nn.Linear(embed_dim,embed_dim)
        self.key_linear=nn.Linear(embed_dim,embed_dim)
        self.value_linear=nn.Linear(embed_dim,embed_dim)

        self.delta=1/(self.head_dim**0.5)
        self.layer_norm=nn.LayerNorm(embed_dim)#序列任务常用layernorm,图像任务常用BN
        self.multihead_attention_embedding=nn.Linear(embed_dim,embed_dim)
        self.dropout=nn.Dropout(dropout_rate)
    def forward(self,x):
        B,N,D=x.size()
        #进行QKV在多个头的映射
        #[B,N,D]-> [B,N,num_heads,head_dim]->[B,num_heads,N,head_dim]
        #num_heads 维度提前，可以批量并行进行子空间的自注意力计算
        Q=self.query_linear(x).view(B,N,self.num_heads,self.head_dim).permute(0,2,1,3)
        K=self.key_linear(x).view(B,N,self.num_heads,self.head_dim).permute(0,2,1,3)
        V=self.value_linear(x).view(B,N,self.num_heads,self.head_dim).permute(0,2,1,3)

        #计算自注意力得到[B,num_heads,N,N]的权重
        KT=K.permute(0,1,3,2) # ->[B,num_heads,head_dim,N]
        scores=torch.matmul(Q,KT)*self.delta#权重 ->[B,num_heads,N,N]
        weights=F.softmax(scores,dim=-1)# 非独立归一化
        weights=self.dropout(weights)

        #整合多个子空间自注意力，计算多头注意力
        #[B,num_heads,N,N]@[B,num_heads,N,head_dim] ->[B,num_heads,N,head_dim]
        attention=torch.matmul(weights,V)# 各子空间自注意力加权结果
        #->[B,N,num_heads,head_dim] ->[B,N,D]
        attention=attention.permute(0,2,1,3).contiguous().view(B,N,D)
        multihead_attention=self.multihead_attention_embedding(attention)#计算多头注意力，整合多个空间自注意力特征
        multihead_attention=self.dropout(multihead_attention)
        return multihead_attention#到encoder再残差连接

# -----------------------------
# 4. Transformer Encoder Block 模块
# -----------------------------
class TransformerEncoderBlock(nn.Module):
    """
    Encoder Block
    Transformer 编码器块包含两个主要子模块：
      1. 多头自注意力（注意力子层），前后配以 LayerNorm 和残差连接；
      2. MLP（前馈网络），同样配以 LayerNorm 和残差连接。
    """
    def __init__(
            self,
            embed_dim,
            num_heads=12,
            mlp_ratio=4,# 隐藏层变大倍数
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
    ):
        super().__init__()
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.mlp_ratio=mlp_ratio

        self.multihead_attention=MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout_rate=attention_dropout_rate
        )
        self.mlp=MLP(
            input_dim=embed_dim,
            hidden_dim=int(embed_dim*mlp_ratio),
            output_dim=embed_dim,
            dropout_rate=dropout_rate
        )

        self.layer_norm_1=nn.LayerNorm(embed_dim)
        self.layer_norm_2=nn.LayerNorm(embed_dim)
    def forward(self,x):
        #先归一化再残差连接pre-norm
        x=self.multihead_attention(self.layer_norm_1(x))+x 
        x=self.mlp(self.layer_norm_2(x))+x
        return x

# -----------------------------
# 5. Vision Transformer 整体结构
# -----------------------------
class VisionTransformer(nn.Module):
    """
    Vision Transformer（ViT）整体结构：
      1. Patch Embedding：将输入图像切分为若干 patch，并线性映射到一个较高的维度。
      2. Positional Embedding：为每个 patch（以及分类 token）添加位置信息。
      3. Transformer Encoder Blocks：堆叠多个编码器块进行全局信息交互。
      4. Classification Head：用一个全连接层对分类 token 输出进行分类。
      参数：
      image_size: 输入图像尺寸，默认为 224。
      patch_size: 输入图像切分 patch 大小，默认为 16。
      input_channels: 输入图像通道数，默认为 3。
      num_classes: 分类数目，默认为 1000。
      embed_dim: 词向量维度，默认为 768。
      depth: 编码器堆叠次数，默认为 12。
    """
    def __init__(
        self,
        image_size=224,
        patch_size=16,# patch尺寸
        input_channels=3,
        num_classes=1000,
        embed_dim=768,# patch嵌入维度
        depth=12,# 编码器堆叠次数
        num_heads=12,# 多头注意力的头数
        mlp_ratio=4,# 隐藏层变大倍数
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # 1. Patch Embedding，将图像切分为 patch 并映射到 embed_dim 维度
        self.patch_embed=PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            input_channels=input_channels,
            embed_dim=embed_dim,
        )#[B,num_patches,embed_dim]<=>[B,N,D]
        self.num_patches=self.patch_embed.num_patches#图像的patch数量->词个数

        # 2. 分类 token（cls token），一个学习得到的向量，后续用于分类任务
        # 加到每个样本的patch嵌入之前，因为不属于图像，所以可以通过自注意力与其他patch交互
        # 得到全局的，可以代表整个图像特征的向量。初始为0
        self.cls_token=nn.Parameter(torch.zeros(1,1,embed_dim))

        # 3. 位置编码，为每个 token 添加位置信息（包括 cls token）
        self.pos_embed=nn.Parameter(torch.zeros(1,self.num_patches+1,embed_dim))# [1,N+1,D]
        self.pos_drop=nn.Dropout(dropout_rate)

        # 4. Transformer Encoder Blocks：堆叠多个 Transformer 编码器块
        self.encoder_blocks=nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout_rate=dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
            )
            for _ in range(depth)#构建depth个编码器块
        ])
        # 编码器最后的归一化层
        self.layer_norm=nn.LayerNorm(embed_dim)#对一个样本归一化

        # 5. 分类头：cls_token 训练为可以代表图像进行分类的向量，将 cls_token 的输出映射到类别数
        self.head=nn.Linear(embed_dim,num_classes)
        # 6. 初始化权重
        self._init_weights()

    def forward(self,x):
        B,C,H,W=x.size()
        D=self.embed_dim # 对应词向量的嵌入维度，此处为patch的嵌入维度
        # 1. 通过 Patch Embedding 得到 patch token 序列
        x=self.patch_embed(x)#[B,num_patches,embed_dim] <=> [B,N,D]

        # 2. 添加 cls token 到 token 序列的最前面（第二维度的0）
        cls_token=self.cls_token.expand(B,-1,-1)#[1,1,embed_dim]->[B,1,embed_dim] 每个样本都有一个cls token
        x=torch.cat([cls_token,x],dim=1)#[B,N+1,D]

        # 3. 加上位置编码，并做 dropout
        x=x+self.pos_embed#[B,N+1,D]
        x=self.pos_drop(x)

        # 4. 堆叠多个 Transformer Encoder Blocks
        for encoder_block in self.encoder_blocks:
            x=encoder_block(x)#[B,N+1,D]
        x=self.layer_norm(x)#[B,N+1,D]

        # 5. 分类头，输出类别 
        cls_output=x[:,0]#取出clas_token全局特征[B,D]   
        return self.head(cls_output)# 对全局特征进行分类


    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed,std=0.02,a=-2,b=2)#截断高斯分布为-2,2之间
        nn.init.trunc_normal_(self.cls_token,std=0.02,a=-2,b=2)
        for module in self.modules():
            # 初始化所有线性层和 LayerNorm 参数
            if isinstance(module,nn.Linear):
                nn.init.trunc_normal_(module.weight,std=0.02,a=-2,b=2)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module,nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
if __name__=='__main__':
    import sys
    sys.path.append("..")
    from metrics import ModelMeasurer
    model=VisionTransformer(
        image_size=224,
        patch_size=16,
        input_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
    )
    measurer=ModelMeasurer(model)
    measurer.simply_check_model(input_shape=(4, 3, 224, 224),inference_repeation=100)