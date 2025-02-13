### 本人准备研究生复试的计算机视觉基础学习记录项目。

#### 目的为学习和实现一个简单的训练流程，学习算法时重点考虑网络的编写，避免训练，可视化过程还有大框架的学习成本和繁琐操作。

#### 目前较为完整的实现了图像分类和目标检测的训练流程

- 数据集处理和划分
- 训练参数记录和训练日志的记录
- 训练中检查点的设置和训练中模型指标计算比较出的最优模型的存储
- 较为完整的训练过程和指标计算
- 支持AMP混合精度计算，训练过程动态学习率调整，梯度缩放等操作

#### 不同任务

- **main.ipynb**为**图像分类**的笔记文件，可一键训练和调参
- **detection.ipynb**为**目标检测**的笔记文件，可一键训练和调参

#### 网络文件相关

网络文件位于`Project/utils/Net`，均是自己实现的经典网络和模块，可以调用和修改，附有注释用于学习参考。

- **经典图像分类网络**
  - `VisionTransformer, ResNet, GoogLeNet, VGGNet, MobileNetV2, EfficientNet`等
- **自己实现的各种注意力机制和模块**
  - `SelfAttention2D, MultiHeadAttention, CBAM`等
  - `InvertedResidualBlock, MBConvBlock, DepthSeparableConvBlock`等
- **待续**

#### 环境信息如下

- python==3.9.17
- numpy==1.25.2
- torch==2.0.0+cu118
- torchvision=0.15.1+cu118
- `pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html`