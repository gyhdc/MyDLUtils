import os
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def visualize_batch(
        images, 
        predictions, 
        save_path=None, 
        category_names=None, 
        score_threshold=0.5, 
        unnormalize_fn=None,
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
        label_offset=1
    ):
    """
    可视化一个 batch 的目标检测结果。输入图片可以是文件路径、PIL.Image 对象或 tensor（形状：(C,H,W)），
    或者一个包含这些元素的列表；同时 predictions 为每张图片对应的预测结果字典，格式如下：
    
        {
            'boxes': Tensor[N, 4] 或 list,  # [x_min, y_min, x_max, y_max]
            'labels': Tensor[N] 或 list,     # 类别标签
            'scores': Tensor[N] 或 list      # 置信度
        }
    
    :param images: 图片列表，支持以下类型：
                   - 文件路径列表 (str)
                   - PIL.Image 对象列表
                   - tensor，形状为 (N, C, H, W)
                   - 或者包含 tensor 的列表
    :param predictions: 与 images 对应的预测结果列表，每个元素为一个字典
    :param save_path: 如果不为 None，则保存结果到该路径，否则直接显示（建议传入类似 "predict/result.jpg"）；
                      如果目录不存在会自动创建
    :param category_names: 类别名称字典，例如 {1: 'Cat', 2: 'Dog'}，用于展示类别名称
    :param score_threshold: 过滤低于该置信度的检测框
    :param unnormalize_fn: 如果传入，对 tensor 图片进行反归一化操作的函数，形如 lambda img: (img * std + mean)
    """
    def unnormalize(img, mean=mean, std=std):
        # img: tensor (C, H, W)
        mean = torch.tensor(mean).view(-1, 1, 1).to(img.device)
        std = torch.tensor(std).view(-1, 1, 1).to(img.device)
        return img * std + mean
    def process_image(img):
        """
        将单张图片转换为 numpy 数组，方便 matplotlib 显示。
        支持：文件路径、PIL.Image、tensor (C,H,W)。
        """
        # 如果是文件路径
        if isinstance(img, str):
            return np.array(Image.open(img).convert('RGB'))
        # 如果是 PIL.Image 对象
        elif isinstance(img, Image.Image):
            return np.array(img.convert('RGB'))
        # 如果是 tensor
        elif torch.is_tensor(img):
            # 若提供了反归一化函数，则应用
            if unnormalize_fn is not None:
                img = unnormalize_fn(img)
            # 将 tensor 移到 cpu 并 detach
            img = img.detach().cpu()
            # 如果是单通道图像，复制为三通道
            if img.dim() == 3:
                if img.shape[0] == 1:
                    img = img.repeat(3, 1, 1)
            # 转置为 (H, W, C)
            img_np = img.permute(1, 2, 0).numpy()
            # 如果数值在 [0,1]范围内则直接显示；如果数值较大则尝试 uint8
            if img_np.max() > 1:
                img_np = img_np.astype(np.uint8)
            return img_np
        else:
            raise ValueError(f"不支持的图片输入类型：{type(img)}")
    
    # 如果 images 是 tensor 且形状为 (N, C, H, W)，转换为列表
    if torch.is_tensor(images) and images.dim() == 4:
        images_list = [images[i] for i in range(images.shape[0])]
    else:
        images_list = images

    # 将所有图片处理成 numpy 数组
    processed_images = [process_image(img) for img in images_list]

    # 检查 predictions 与图片数量是否匹配
    if len(processed_images) != len(predictions):
        raise ValueError("图片数量与预测结果数量不一致！")

    num_images = len(processed_images)
    # 根据 batch 数量设置网格：最多每行显示 4 张图片
    ncols = min(num_images, 4)
    nrows = math.ceil(num_images / ncols)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5))
    # 当只有一张图片时，axes 可能不是数组
    if num_images == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    # 遍历 batch 中的每张图片及其对应的预测结果
    for idx, (img, pred) in enumerate(zip(processed_images, predictions)):
        ax = axes[idx]
        ax.imshow(img)
        ax.axis('off')
        
        # 获取预测结果中的 boxes, labels, scores
        boxes = pred.get('boxes', [])
        labels = pred.get('labels', [])
        scores = pred.get('scores', [1]*len(boxes))#predict传的标签
        # 如果是 tensor，则转换为 list
        if torch.is_tensor(boxes):
            boxes = boxes.tolist()
        if torch.is_tensor(labels):
            labels = labels.tolist()
        if torch.is_tensor(scores):
            scores = scores.tolist()
        
        # 绘制检测框
        for box, label, score in zip(boxes, labels, scores):
            label
            if score < score_threshold:
                continue
            # box 格式为 [x_min, y_min, x_max, y_max]
            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(rect)
            label+=label_offset
            # 获取类别名称
            class_name = (category_names[label] if (category_names and label in category_names)
                          else f'Class {label}')
            ax.text(
                box[0], box[1] - 5,
                f'{class_name}: {score:.2f}',
                color='red',
                fontsize=10,
                weight='bold',
                bbox=dict(facecolor='white', alpha=0.2, edgecolor='none')
            )

    # 隐藏多余子图（如果有）
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    
    # 保存或显示图像
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        print(f"保存批量预测结果到：{save_path}")
    else:
        plt.show()


# ===================== 示例调用 =====================
if __name__ == "__main__":
    # # 示例1：使用图片文件路径列表
    # image_paths = ["test1.jpg", "test2.jpg", "test3.jpg"]
    # # 假设每张图片对应的预测结果
    # predictions_example = [
    #     {
    #         'boxes': torch.tensor([[30, 30, 180, 180]]),
    #         'labels': torch.tensor([1]),
    #         'scores': torch.tensor([0.92])
    #     },
    #     {
    #         'boxes': torch.tensor([[50, 50, 200, 200], [100, 100, 150, 150]]),
    #         'labels': torch.tensor([2, 3]),
    #         'scores': torch.tensor([0.88, 0.75])
    #     },
    #     {
    #         'boxes': torch.tensor([[20, 20, 100, 100]]),
    #         'labels': torch.tensor([1]),
    #         'scores': torch.tensor([0.95])
    #     }
    # ]
    # visualize_batch(
    #     images=image_paths,
    #     predictions=predictions_example,
    #     save_path="predict/batch_result.jpg",
    #     category_names={1: 'Cat', 2: 'Dog', 3: 'Bird'},
    #     score_threshold=0.5
    # )
    
    # 示例2：使用 dataloader 输出的图片
    # 假设 images_tensor 为 tensor，形状 (N, C, H, W)，例如：
    images_tensor = torch.randn(4, 3, 224, 224)  # 仅作示例（实际数据请使用真实图片）
    # 同时构造对应的预测结果
    predictions_tensor = []
    for i in range(4):
        predictions_tensor.append({
            'boxes': torch.tensor([[30, 30, 180, 180]]),
            'labels': torch.tensor([1]),
            'scores': torch.tensor([0.85])
        })
    # 若图片经过归一化（例如均值和标准差归一化），可以定义 unnormalize_fn：
    # 例如：假设均值=[0.485, 0.456, 0.406]，标准差=[0.229, 0.224, 0.225]
    def unnormalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        # img: tensor (C, H, W)
        mean = torch.tensor(mean).view(-1, 1, 1).to(img.device)
        std = torch.tensor(std).view(-1, 1, 1).to(img.device)
        return img * std + mean
    
    visualize_batch(
        images=images_tensor,
        predictions=predictions_tensor,
        save_path=None,  # 直接显示
        category_names={1: 'Cat'},
        score_threshold=0.5,
        unnormalize_fn=unnormalize
    )
