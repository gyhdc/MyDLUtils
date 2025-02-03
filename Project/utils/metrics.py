import numpy as np
import torch
import tqdm

def get_parameters_num(model):
    '''
        获取模型参数量
        model: 模型
        return: 模型参数量
    '''
    total_params = sum(p.numel() for p in model.parameters())
    return total_params
def measure_inference_time(model, device, input_shape=(1, 3, 256, 256), repetitions=300,unit=1000):
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
    print('Warm up ...\n')
    with torch.no_grad():
        for _ in tqdm.tqdm(range(100),desc='Warm up'):
            _ = model(dummy_input)
    # 同步等待所有 GPU 任务完成
    torch.cuda.synchronize()
    # 初始化时间容器
    timings =[]
    print('Testing ...\n')
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