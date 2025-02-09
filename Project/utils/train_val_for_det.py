import torch
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.detection.mean_ap import MeanAveragePrecision

#############################################
# 辅助函数
#############################################

def weighted_average_detection_metrics(mAP, loss, weights=[1.2, 1.05]):
    """
    根据 mAP 和 loss（经过 1-loss 转换）计算加权平均指标。
    weights[0] 对 mAP 进行加权，weights[1] 对 (1 - loss) 加权。
    """
    adjusted_loss = 1 - loss  # 越小的loss希望转化为越大的指标
    weighted_mAP = mAP * weights[0]
    weighted_loss = adjusted_loss * weights[1]
    return (weighted_mAP + weighted_loss) / sum(weights)

def float_equal(a, b, epsilon=1e-4):
    return abs(a - b) < epsilon

#############################################
# 训练单周期
#############################################

def train_one_epoch_detection(
    model, 
    criterion, 
    optimizer, 
    train_loader, 
    device,
    scaler=None,
    multi_loss_weight=None  # 若模型输出多个loss，可传入各个loss的权重列表
):
    """
    单个训练周期的目标检测模型训练函数
    参数:
        model: 待训练的检测模型
        criterion: 损失函数，接收 (outputs, targets)
        optimizer: 优化器
        train_loader: 训练数据加载器，返回 (images, targets)
        device: 设备
        scaler: 混合精度训练的GradScaler对象（可选）
        multi_loss_weight: 多个loss的权重列表（可选）
    返回:
        平均训练loss
    """
    model.train()
    running_loss = 0.0
    use_amp = scaler is not None

    for images, targets in train_loader:
        # 将图像和目标转移到指定设备
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            outputs = model(images)
            # 如果输出为多个loss，则用 multi_loss_weight 组合，否则直接计算 loss
            if isinstance(outputs, (list, tuple)):
                if multi_loss_weight is not None:
                    if len(outputs) != len(multi_loss_weight):
                        raise ValueError("outputs的数量与multi_loss_weight长度不匹配！")
                    loss = sum(criterion(out, targets) * w for out, w in zip(outputs, multi_loss_weight))
                else:
                    loss = sum(criterion(out, targets) for out in outputs)
            else:
                loss = criterion(outputs, targets)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)

#############################################
# 模型验证
#############################################

def validate_model_detection(
    model, 
    val_loader, 
    device, 
    only_val=False, 
    AMP=True, 
    criterion=None, 
    multi_loss_weight=None
):
    """
    在验证集上评估目标检测模型，返回 mAP 和 loss 等指标。
    参数:
        model: 待评估的检测模型
        val_loader: 验证数据加载器，返回 (images, targets)
        device: 设备
        only_val: 若为 True，则在进度条中显示验证进程
        AMP: 是否启用混合精度
        criterion: 损失函数（可选，用于计算验证loss）
        multi_loss_weight: 多loss的权重列表（可选）
    返回:
        metrics: 包含 "mAP" 和 "Loss" 的字典（若未计算loss，则 Loss 为 None）
    """
    model.eval()
    running_loss = 0.0
    use_amp = AMP
    # 使用 torchmetrics 计算检测 mAP
    metric = MeanAveragePrecision()
    if only_val:
        val_loader = tqdm(val_loader, desc="模型验证中:")

    with torch.no_grad():
        for images, targets in val_loader:
            # images = [img.to(device) for img in images]
            images = torch.stack(images, dim=0).to(device)#一batch的图像[B,C,H,W]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with autocast(enabled=use_amp):
                outputs = model(images)
                if criterion is not None:
                    if isinstance(outputs, (list, tuple)):
                        if multi_loss_weight is not None:
                            if len(outputs) != len(multi_loss_weight):
                                raise ValueError("outputs的数量与multi_loss_weight长度不匹配！")
                            loss = sum(criterion(out, targets) * w for out, w in zip(outputs, multi_loss_weight))
                        else:
                            loss = sum(criterion(out, targets) for out in outputs)
                        running_loss += loss.item()
            # 将预测结果转换到 CPU，满足 torchmetrics 的要求
            outputs_cpu = [{k: v.cpu() for k, v in out.items()} for out in outputs]
            targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]
            metric.update(outputs_cpu, targets_cpu)
    metric_result = metric.compute()
    mAP = metric_result["map"].item() if isinstance(metric_result["map"], torch.Tensor) else metric_result["map"]
    avg_loss = running_loss / len(val_loader) if criterion is not None else None

    metrics = {
        "mAP": mAP,
        "Loss": avg_loss,
        # 可根据需要添加其他指标，如 map_50, map_75 等
    }
    return metrics

#############################################
# 更新最佳模型
#############################################

def update_best_model_detection(bestMod, current_metrics, epoch_loss, model, epoch, metrics_weights):
    """
    根据加权指标更新 bestMod 对象（最佳模型记录器）。
    参数:
        bestMod: 最佳模型记录器对象，应包含属性如 bestMod.mAP, bestMod.loss，并提供 update() 和 update_checkpoint() 方法。
        current_metrics: 当前评估指标字典，至少包含 "mAP"
        epoch_loss: 当前训练周期的训练loss
        model: 当前模型
        epoch: 当前周期数
        metrics_weights: 用于计算加权指标的权重列表（例如 [1.2, 1.05]）
    """
    # 这里采用 mAP 和训练loss（转换为 1 - loss）计算加权指标
    current_weighted = weighted_average_detection_metrics(
        mAP=current_metrics["mAP"], 
        loss=epoch_loss, 
        weights=metrics_weights
    )
    try:
        best_weighted = weighted_average_detection_metrics(
            mAP=bestMod.mAP, 
            loss=bestMod.loss, 
            weights=metrics_weights
        )
    except Exception:
        best_weighted = 0

    if current_weighted >= best_weighted:
        # 若加权指标相等，则比较 loss
        if float_equal(current_weighted, best_weighted):
            if epoch_loss < bestMod.loss:
                bestMod.update(
                    mAP=current_metrics["mAP"],
                    model=model,
                    loss=epoch_loss,
                    epoch=epoch,
                )
        else:
            bestMod.update(
                mAP=current_metrics["mAP"],
                model=model,
                loss=epoch_loss,
                epoch=epoch,
            )

#############################################
# 整体训练流程
#############################################

def train_model(
    model, 
    criterion, 
    optimizer, 
    train_loader, 
    val_loader,
    bestMod,
    train_logs, 
    config, 
    metrics_weights=[1.2, 1.05],
    num_epochs=10,
    checkpoint_interval=0.25,
    show_progress_interval=2,
    AMP=True,
    multi_loss_weight=None,
    lr_scheduler_step=0.6,
):
    """
    训练目标检测模型，并在每个 epoch 后于验证集上评估，同时更新最佳模型和训练日志。
    参数:
        model: 待训练的检测模型
        criterion: 损失函数
        optimizer: 优化器
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        bestMod: 用于记录最佳模型的对象
        train_logs: 用于记录训练日志的对象，需提供 update() 方法
        config: 配置对象，应包含 config.epochs 和 config.device
        metrics_weights: 用于加权指标计算的权重列表
        num_epochs: 训练周期数（此处会以 config.epochs 为准）
        checkpoint_interval: 检查点保存间隔（若小于1，则视为占总周期比例）
        show_progress_interval: 日志显示周期（若小于1，则视为占总周期比例）
        AMP: 是否使用混合精度
        multi_loss_weight: 多个loss的权重（可选）
        lr_scheduler_step: 学习率调度因子（若不为0则启用 ReduceLROnPlateau）
    返回:
        bestMod: 更新后的最佳模型对象
        train_logs: 更新后的训练日志记录对象
    """
    num_epochs = config.epochs
    device = config.device

    # 初始化日志记录列表
    loss_history = []
    mAP_history = []
    validation_loss_history = []

    scaler = GradScaler() if AMP else None

    if checkpoint_interval < 1:
        final_checkpoint_interval = int(checkpoint_interval * num_epochs)
    else:
        final_checkpoint_interval = int(checkpoint_interval)
    if show_progress_interval < 1:
        final_show_interval = int(show_progress_interval * num_epochs)
    else:
        final_show_interval = int(show_progress_interval)
    if final_checkpoint_interval == 0:
        final_checkpoint_interval = 1
    if final_show_interval == 0:
        final_show_interval = 1

    lr_scheduler = None
    if lr_scheduler_step != 0:
        lr_scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=lr_scheduler_step, 
            patience=final_checkpoint_interval, verbose=True
        )

    for epoch in tqdm(range(num_epochs), desc="训练检测模型的 Epoch"):
        # 训练阶段
        epoch_loss = train_one_epoch_detection(
            model, 
            criterion,
            optimizer, 
            train_loader, 
            device,
            scaler=scaler,
            multi_loss_weight=multi_loss_weight
        )
        # 验证阶段
        metrics = validate_model_detection(
            model, 
            val_loader, 
            device,
            AMP=AMP,
            criterion=criterion,
            multi_loss_weight=multi_loss_weight
        )
        if lr_scheduler is not None:
            # 优先使用验证loss（若可用），否则使用训练loss
            current_val_loss = metrics["Loss"] if metrics["Loss"] is not None else epoch_loss
            lr_scheduler.step(current_val_loss)

        loss_history.append(epoch_loss)
        mAP_history.append(metrics["mAP"])
        validation_loss_history.append(metrics["Loss"] if metrics["Loss"] is not None else 0)

        # 更新最佳模型
        update_best_model_detection(bestMod, metrics, epoch_loss, model, epoch, metrics_weights=metrics_weights)
        if epoch % final_checkpoint_interval == 0:
            bestMod.update_checkpoint(model, epoch)

        if epoch % final_show_interval == 0:
            val_loss_str = f'{metrics["Loss"]:.4f}' if metrics["Loss"] is not None else '0.0000'
            print(
                f'Epoch {epoch+1}/{num_epochs}, train_Loss: {epoch_loss:.4f}, '
                f'val_Loss: {val_loss_str}, mAP: {metrics["mAP"]:.4f}'
            )
            print(f"当前最好的模型: {str(bestMod)}")

        # 更新训练日志记录对象（train_logs 需实现 update 方法）
        train_logs.update(
            loss_lst=loss_history,
            mAP_lst=mAP_history,
            val_loss_history=validation_loss_history,
        )
    return bestMod, train_logs
