import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import VOCDetection
import torchvision.transforms as transforms
from torchmetrics.detection.mean_ap import MeanAveragePrecision
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
        return images, targets