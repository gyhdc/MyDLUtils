import numpy as np
import torch
import tqdm

from torch.utils.tensorboard import SummaryWriter


'''
• Scalars：显示训练损失和准确率的变化曲线。
• Graphs：可以查看模型的计算图。
• Images：如果记录了图像数据，可以在这里查看。
• Distributions 和 Histograms：可以查看模型参数的分布

'''
class TensorboardRecorder:
    '''
        Tensorboard 记录器
        参数：
            log_dir: 日志目录，默认为 runs/。
    '''
    def __init__(self, log_dir=None):
        """
        如果 log_dir 为 None，则使用默认日志路径 runs/。
        """
        self.writer = SummaryWriter(log_dir=log_dir)
    def get_proper_tag(self, tag : str):
        '''第一个字符大写，其余不变'''
        tag=tag.replace("_lst","")
        # capitalized_s = tag[0].upper() + tag[1:]
        return tag

    def logs_scalars(self, scalars: dict, prefix: str = ""):
        '''
            scalars:{"Accuracy":[acc1,aacc2...]}
            参数：
                scalars: 字典，键为指标名称，值为一个列表，表示每个指标的记录值。
                prefix: 前缀，用于区分不同指标。
        '''
        keys=list(scalars.keys())
        epochs_num=len(scalars[keys[0]])
        for key,logs in scalars.items():
            for epoch in range(epochs_num):
                key=self.get_proper_tag(key)
                tag="_".join([prefix,key])
                
                self.writer.add_scalar(tag, logs[epoch], epoch)

    
    def log_model_graph(self, model, input_example):
        """
        记录模型结构图，需要提供一个 input_example 用于构建图。
        """
        self.writer.add_graph(model, input_example)
    
    def log_histograms(self, model, epoch: int):
        """
        对模型所有参数添加直方图记录，便于观察参数分布变化。
        """
        for name, param in model.named_parameters():
            self.writer.add_histogram(name, param, epoch)
    
    def log_text(self, tag: str, text: str, epoch: int):
        """
        添加文本记录，比如保存模型结构、超参数信息等。
        """
        self.writer.add_text(tag, text, epoch)
    
    def close(self):
        self.writer.close()
def get_parameters_num(model):
    '''
        获取模型参数量
        model: 模型
        return: 模型参数量
    '''
    total_params = sum(p.numel() for p in model.parameters())
    return total_params
def get_inference_time(
        model, 
        device, 
        input_shape=(4, 3, 256, 256), 
        repetitions=300,
        unit=1000
        ):
    '''
        获取平均模型推理时间
        model: 模型
        device: 设备
        input_shape: 输入形状
        repetitions: 测试次数
        unit: 时间单位，默认为毫秒
        return: 平均推理时间
    '''
    # 准备模型和输入数据
    model = model.to(device)
    dummy_input = torch.rand(*input_shape).to(device)
    model.eval()
    
    # 预热 GPU
    # print('Warm up ...\n')
    with torch.no_grad():
        for _ in tqdm.tqdm(range(100),desc='Warm up ....'):
            _ = model(dummy_input)
    # 同步等待所有 GPU 任务完成
    torch.cuda.synchronize()
    # 初始化时间容器
    timings =[]
    # print('Testing ...\n')
    with torch.no_grad():
        for rep in tqdm.tqdm(range(repetitions),desc='Testing ...'):
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # 同步等待 GPU 任务完成
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)  # 从 starter 到 ender 之间用时，单位为毫秒
            timings.append( curr_time)
    avg = sum(timings) / repetitions/unit
    return avg

def weighted_average_metrics(acc, ap, precision , recall, loss, weights=[1.1, 1.2, 1,1, 1.05]):
    '''
        获取加权指标选择模型
        acc: 准确率
        ap: 平均准确率
        precision: 精确率
        recall: 召回率
        loss: 损失率
        return: 加权平均指标
    '''
    loss=1-loss
    weighted_acc = acc * weights[0]
    weighted_ap = ap * weights[1]
    weighted_precision=precision*weights[2]
    weighted_recall = recall * weights[3]
    weighted_loss = loss * weights[4]

    weighted_average = (weighted_acc + weighted_ap + weighted_precision + weighted_recall + weighted_loss) / sum(weights)

    return weighted_average