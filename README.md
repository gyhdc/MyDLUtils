### 本人准备研究生复试的计算机视觉基础学习记录项目。
#### 目前较为完整的实现了图像识别的整套训练流程
- 数据集处理和划分
- 训练参数记录和训练日志的记录
- 训练中检查点的设置和训练中模型指标计算比较出的最优模型的存储
- 较为完整的训练过程和指标计算
- 支持AMP混合精度计算，训练过程动态学习率调整，梯度缩放等操作
#### 网络文件相关
网络文件位于**Project/utils/Net**，均是自己实现的经典网络和模块，用于学习参考
- 经典图像分类网络(如ResNet，GoogLeNet，VGGNet，AlexNet等)
- 自己实现的各种注意力机制，如SelfAttention2D，ChannelsAttention，SpatialAttention，CBAM等
- **待续**
#### 环境信息如下
- python==3.9.17
- numpy==1.25.2
- torch==2.0.1+cu117
- tensorboard==2.15.1
