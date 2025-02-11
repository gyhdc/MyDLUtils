import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import VOCDetection
import torchvision.transforms as transforms
from torchmetrics.detection.mean_ap import MeanAveragePrecision
class ImagesAndBoxesTransform:
    def __init__(self, resize_size, mean=None, std=None):
        self.resize_size = (resize_size[1], resize_size[0])  # (height, width)
        self.mean = mean
        self.std = std
        self.resize = transforms.Resize(self.resize_size)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=mean, std=std) if mean is not None else None

    def __call__(self, image, target):
        
        image = self.to_tensor(image)  # (C, H, W)
        original_height, original_width = image.shape[1], image.shape[2]

        image = self.resize(image)  # (C, new_H, new_W)
        new_height, new_width = image.shape[-2], image.shape[-1]


        boxes = []
        labels = []
        for meta in target:#这里 meta['bbox'] 是 [x, y, width, height]
            x, y, w, h = meta['bbox']
            xmin = x
            ymin = y
            xmax = x + w
            ymax = y + h
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(meta['category_id'])

        
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

        # Scale boxes
        if target is not None and "boxes" in target and len(target["boxes"]) > 0:
            boxes = target["boxes"].clone().float()
            scale_factor_w = new_width / original_width
            scale_factor_h = new_height / original_height

            boxes[:, 0::2] *= scale_factor_w  # xmin and xmax
            boxes[:, 1::2] *= scale_factor_h  # ymin and ymax

            # Clamp boxes to image boundaries
            boxes[:, 0::2].clamp_(min=0, max=new_width)
            boxes[:, 1::2].clamp_(min=0, max=new_height)

            target["boxes"] = boxes

        else:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64)
            }

        # Apply normalization
        if self.normalize is not None:
            image = self.normalize(image)

        return image, target


# class ImagesAndBoxesTransform:
#     def __init__(self, resize_size, mean=None, std=None):
        
#         self.resize_size = (resize_size[1], resize_size[0])  # 转换为 (height, width)
#         self.mean = mean
#         self.std = std
#         self.resize = transforms.Resize(self.resize_size)
#         self.to_tensor = transforms.ToTensor()
#         self.normalize = transforms.Normalize(mean=mean, std=std) if mean is not None else None

#     def __call__(self, image, target):
#         """
#         images 图像：一个 PIL.Image 对象或 numpy.ndarray，形状为 (H, W, C)。
#         targets 真实标注：一个字典，包含 boxes和labels  。
#         """
#         # 转换为 Tensor
#         image = self.to_tensor(image)  # (C, H, W)
#         original_height, original_width = image.shape[1], image.shape[2]

#         # Resize 图像
#         image = self.resize(image)  # (C, new_H, new_W)
#         new_height, new_width = image.shape[-2], image.shape[-1]
#         boxes=[]
#         labels = []
#         for meta in target:
#             boxes.append(meta['bbox'])#这里 meta['bbox'] 是 [x, y, width, height]
#             labels.append(meta['category_id'])
#         target={
#             "boxes": torch.tensor(boxes, dtype=torch.float32),
#             "labels": torch.tensor(labels, dtype=torch.int8)
#         }
#         # 调整边界框坐标
#         if target is not None and "boxes" in target and len(target["boxes"]) > 0:
#             boxes = target["boxes"].clone().float()  # 确保为浮点型 Tensor
#             # 计算缩放因子
#             scale_factor_w = new_width / original_width
#             scale_factor_h = new_height / original_height
#             # 调整坐标
#             boxes[:, 0::2] *= scale_factor_w  # x 坐标 (xmin, xmax)
#             boxes[:, 1::2] *= scale_factor_h  # y 坐标 (ymin, ymax)
#             # 截断到图像范围内
#             boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=new_width)
#             boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=new_height)
#             target["boxes"] = boxes
#         else:
#             # 处理空目标
#             target = {"boxes": torch.zeros((0, 4), dtype=torch.float32), "labels": torch.zeros(0, dtype=torch.int64)}

#         # 归一化图像
#         if self.normalize is not None:
#             image = self.normalize(image)

#         return image, target

class COCO:
    '''
        [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 150, 200, 300],
                "area": 60000,
                "iscrowd": 0,
                "segmentation": [[100, 150, 300, 150, 300, 450, 100, 450]]
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 2,
                "bbox": [400, 200, 100, 200],
                "area": 20000,
                "iscrowd": 0,
                "segmentation": [[400, 200, 500, 200, 500, 400, 400, 400]]
            }
        ]
    '''
    COCO_CLASSES=COCO_CLASSES = ['background',
 'person',
 'bicycle',
 'car',
 'motorcycle',
 'airplane',
 'bus',
 'train',
 'truck',
 'boat',
 'traffic light',
 'fire hydrant',
 'NULL_12',
 'stop sign',
 'parking meter',
 'bench',
 'bird',
 'cat',
 'dog',
 'horse',
 'sheep',
 'cow',
 'elephant',
 'bear',
 'zebra',
 'giraffe',
 'NULL_26',
 'backpack',
 'umbrella',
 'NULL_29',
 'handbag',
 'tie',
 'suitcase',
 'frisbee',
 'skis',
 'snowboard',
 'sports ball',
 'kite',
 'baseball bat',
 'baseball glove',
 'skateboard',
 'surfboard',
 'tennis racket',
 'bottle',
 'wine glass',
 'NULL_45',
 'cup',
 'fork',
 'knife',
 'spoon',
 'bowl',
 'banana',
 'apple',
 'sandwich',
 'orange',
 'broccoli',
 'carrot',
 'hot dog',
 'pizza',
 'donut',
 'cake',
 'chair',
 'couch',
 'potted plant',
 'bed',
 'dining table',
 'NULL_66',
 'toilet',
 'NULL_68',
 'tv',
 'laptop',
 'NULL_71',
 'mouse',
 'remote',
 'keyboard',
 'cell phone',
 'microwave',
 'oven',
 'toaster',
 'sink',
 'refrigerator',
 'book',
 'clock',
 'NULL_83',
 'vase',
 'NULL_85',
 'scissors',
 'NULL_87',
 'teddy bear',
 'NULL_89',
 'hair drier',
 'toothbrush']

    @classmethod
    def collate_fn(self,batch):
        """
        自定义 collate_fn 用于 COCO 数据集，转换每个样本的 target 格式以符合 torchvision Faster R-CNN 的要求。
        
        每个 batch 中的样本格式为 (image, target)，其中：
        - image: tensor 或 PIL.Image 对象（建议转换为 tensor，形状为 (C, H, W)）
        - target: list，每个元素为一个字典，包含 keys: ['segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id']
        
        转换过程：
        - 对于每个 target（一个列表），将所有标注中的 `"bbox"` 转换为 `[x_min, y_min, x_max, y_max]` 格式，
            并将 `"category_id"` 改为 `"labels"`，同时保留 `"area"`, `"iscrowd"`, `"image_id"` 等。
        
        返回：
        - images: list，每个元素为一张图片
        - targets: list，每个元素为字典，格式如下：
                {\n
                    'boxes': Tensor[N, 4],\n
                    'labels': Tensor[N],\n
                    'area': Tensor[N],\n
                    'iscrowd': Tensor[N],\n
                    'image_id': Tensor[1]\n
                }
        """
        images, targets = zip(*batch)  # 分别获取图片和对应的 targets
        new_targets = []
        for anns in targets:
            # print(anns)
            boxes = []
            labels = []
            area = []
            iscrowd = []
            image_id = None
            # anns 是一个列表，每个元素是一个标注字典
            if isinstance(anns, dict):
                target_dict = anns
            else:
                for ann in anns:
                    # print(ann)
                    # COCO 的 bbox 格式为 [x, y, width, height]，转换为 [x_min, y_min, x_max, y_max]
                    x, y, w, h = ann['bbox']
                    boxes.append([x, y, x + w, y + h])
                    labels.append(ann['category_id'])
                    area.append(ann['area'])
                    iscrowd.append(ann['iscrowd'])
                    # image_id 在同一图片内相同，取第一个即可
                    if image_id is None:
                        image_id = ann['image_id']
                if image_id is None:
                    image_id = -1  # 如果该图片没有标注，则设置默认 image_id
                # 构建新的 target 字典
                target_dict = {
                    'boxes': torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
                    'labels': torch.as_tensor(labels, dtype=torch.int8) if labels else torch.tensor([], dtype=torch.int8),
                    'area': torch.as_tensor(area, dtype=torch.float32) if area else torch.tensor([], dtype=torch.float32),
                    'iscrowd': torch.as_tensor(iscrowd, dtype=torch.int8) if iscrowd else torch.tensor([], dtype=torch.int64),
                    'image_id': torch.tensor([image_id])
                }
            new_targets.append(target_dict)
        
        # 返回格式为 (list[images], list[target_dict])
        images=torch.stack(images,dim=0)
        '''
            images:[B,C,H,W]
            targets:{
                boxes:Tensor[N,4],
                labels:Tensor[N],
                area:Tensor[N],
                iscrowd:Tensor[N],
                image_id:Tensor[1]
            }
        '''
        return images, new_targets

    
class PascalVOC:
# VOC 20 个目标类别（注意 VOC 中类别名称为小写）
    VOC_CLASSES = (
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
        'dog', 'horse', 'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor'
    )
    @classmethod
    def label_to_int(self,label_name):
        """将类别名称转换为数字标签"""
        return self.VOC_CLASSES.index(label_name)
    @classmethod
    def label_to_int(self,label_name):
        """将类别名称转换为数字标签"""
        return self.VOC_CLASSES.index(label_name)
    @classmethod
    def collate_fn(self,batch):
        """
        自定义 batch 处理函数。
        从 VOCDetection 数据集中读取的数据格式为 (image, target)，
        其中 target 为一个字典，包含标注信息，本函数解析出目标的 bbox 和类别。
        """
        images = []
        targets = []
        for sample in batch:
            image, target = sample
            images.append(image)
            annotation = target["annotation"]
            boxes = []
            labels = []
            objs = annotation["object"]
            # 如果只有一个目标，VOC 返回的不是列表
            if not isinstance(objs, list):
                objs = [objs]
            for obj in objs:
                bndbox = obj["bndbox"]
                xmin = float(bndbox["xmin"])
                ymin = float(bndbox["ymin"])
                xmax = float(bndbox["xmax"])
                ymax = float(bndbox["ymax"])
                boxes.append([xmin, ymin, xmax, ymax])
                label_name = obj["name"]
                labels.append(self.label_to_int(label_name))
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            targets.append({"boxes": boxes, "labels": labels})
            images = torch.stack(images, dim=0)
            '''
            images:[B,C,H,W]
            targets:{
                boxes:Tensor[N,4],
                labels:Tensor[N],
                }
            '''
        return images, targets