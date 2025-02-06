import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.metrics import average_precision_score
from torch.cuda.amp import autocast,GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

def weighted_average_metrics(acc, ap, precision , recall, loss, weights=[30, 1.2, 1,1, 1.05]):
    loss=1-loss
    weighted_acc = acc * weights[0]
    weighted_ap = ap * weights[1]
    weighted_precision=precision*weights[2]
    weighted_recall = recall * weights[3]
    weighted_loss = loss * weights[4]

    weighted_average = (weighted_acc + weighted_ap + weighted_precision + weighted_recall + weighted_loss) / sum(weights)

    return weighted_average
def float_equal(a, b, epsilon=1e-4):
    return abs(a - b) < epsilon

def train_one_epoch(
    model, 
    criterion, 
    optimizer, 
    train_loader, 
    device,
    scaler=None,
    multi_loss_weight=[1, 0.3, 0.3]  # 更改变量名拼写
):
    model.train()
    running_loss = 0.0
    use_amp = scaler is not None

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()


        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(inputs)
            if isinstance(outputs, (list, tuple)):
                if len(outputs) != len(multi_loss_weight):
                    raise ValueError(f"outputs和权重长度不匹配！")
                loss = sum(criterion(out, labels)*w for out, w in zip(outputs, multi_loss_weight))
            else:
                loss = criterion(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)
        

def validate_model(model, val_loader, device,only_val=False,AMP=True,criterion=None,multi_loss_weight=[1, 0.3, 0.3]):
    """
    在或测试集验证集上评估模型，返回各项指标及中间需要的结果。
    参数:
        model: 待评估的模型。
        val_loader: 验证数据加载器。
        device: 设备。
    return:
        metrics: 包含 accuracy, precision, recall, ap 等指标的字典。
    """
    model.eval()
    true_labels = []
    predicted_labels = []
    predicted_scores = []
    running_loss = 0.0
    if only_val:
        val_loader=tqdm(val_loader,desc="模型测试中:")#only_val=True时，用户单独调用进行测试和验证，显示进度条
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast(enabled=AMP):
                outputs = model(inputs)
                if criterion is not None:#需要计算loss时
                    if isinstance(outputs, (list, tuple)):
                        if len(outputs) != len(multi_loss_weight):
                            raise ValueError(f"outputs和权重长度不匹配！")
                        loss = sum(criterion(out, labels)*w for out, w in zip(outputs, multi_loss_weight))
                        
                    else:
                        loss = criterion(outputs, labels)
                        running_loss += loss.item()

            # 取预测类别

            _, predicted = torch.max(outputs.data, 1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
            # 假定输出第二列为正类分数
            predicted_scores.extend(outputs[:, 1].cpu().numpy())

    # 计算各项指标
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average="macro")
    recall = recall_score(true_labels, predicted_labels, average="macro")
    f1 = f1_score(true_labels, predicted_labels, average="macro")
    # 注意 average_precision_score 要求输入为二维数组
    ap = average_precision_score(
        np.array(true_labels).reshape(-1, 1),
        np.array(predicted_scores).reshape(-1, 1)
    )

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AP": ap,
        "Loss":running_loss/len(val_loader)
    }
    return metrics

def update_best_model(bestMod, current_metrics, epoch_loss, model, epoch,metrics_weights):
    """
    根据加权指标更新 bestMod.
    current_metrics 为一个字典，包含 accuracy, precision, recall, ap 等指标。
    """
    zh = weighted_average_metrics(
        acc=current_metrics["Accuracy"],
        recall=current_metrics["Recall"],
        ap=current_metrics["AP"],
        loss=epoch_loss,
        precision=current_metrics["Precision"],
        weights=metrics_weights
    )
    try:
        mod_zh = weighted_average_metrics(
            acc=bestMod.acc,
            ap=bestMod.ap,
            loss=bestMod.loss,
            recall=bestMod.recall,
            precision=current_metrics["Precision"],  # 注意这里依然使用当前的 precision
            weights=metrics_weights
        )
    except Exception:
        mod_zh = 0

    if zh >= mod_zh:
        # 若加权指标相同，则比较 epoch_loss
        if float_equal(zh, mod_zh):
            if epoch_loss < bestMod.loss:
                bestMod.update(
                    acc=current_metrics["Accuracy"],
                    model=model,
                    loss=epoch_loss,
                    precision=current_metrics["Precision"],
                    recall=current_metrics["Recall"],
                    ap=current_metrics["AP"],
                    # f1=current_metrics["F1"],
                    epoch=epoch,
                )
        else:
            bestMod.update(
                acc=current_metrics["Accuracy"],
                model=model,
                loss=epoch_loss,
                # f1=current_metrics["f1"],
                precision=current_metrics["Precision"],
                recall=current_metrics["Recall"],
                ap=current_metrics["AP"],
                epoch=epoch,
            )

def train_model(model, 
                criterion, 
                optimizer, 
                train_loader, 
                val_loader,
                bestMod,
                train_logs, 
                config, 
                metrics_weights=[30, 1.2, 1,1, 1.05],
                num_epochs=10,
                checkpoint_interval=0.25,
                show_progress_interval=2,
                AMP=True,
                multi_loss_weight=[1,0.3,0.3],
                lr_scheduler_step=0.6,

    ):
    
    """
        训练模型，并在每个epoch结束后在验证集上评估，
        同时更新最佳模型和日志记录。
        参数:
            model: 待训练的模型。
            criterion: 损失函数。
            optimizer: 优化器。
            train_loader: 训练数据加载器。
            val_loader: 验证数据加载器。
            bestMod: 用于记录最佳模型的对象。
            train_logs: 用于记录训练日志的列表。
            config: 配置对象。
            metrics_weights: 用于计算加权指标的权重列表。
            num_epochs: 训练的周期数。
            checkpoint_interval: 检查点保存的间隔。
            show_progress_interval: 进度条显示的间隔。
            AMP:是否采用混合精度计算
            multi_loss_weight:多个loss的加权
        返回:
            bestMod: 最佳模型对象。
            train_logs: 训练日志记录列表。
            
    """
    num_epochs=config.epochs
    device = config.device
    # 日志记录列表
    loss_history = []
    acc_history = []
    precision_history = []
    recall_history = []
    f1_history = []
    ap_history = []
    validation_loss_history=[]
    if AMP == True:# 混合精度训练
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    if checkpoint_interval < 1:
        final_checkpoint_interval=int(checkpoint_interval*num_epochs)#每百分之epochs检查一次
    else:
        final_checkpoint_interval=checkpoint_interval
    if show_progress_interval < 1:
        show_progress_interval=int(show_progress_interval*num_epochs)#每百分之epochs检查一次
    else:
        show_progress_interval=show_progress_interval
    if final_checkpoint_interval==0:
        final_checkpoint_interval=1
    if show_progress_interval==0:
        show_progress_interval=1
    if lr_scheduler_step!=0:
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=lr_scheduler_step, patience=final_checkpoint_interval, verbose=True)
    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        # 训练阶段
        epoch_loss = train_one_epoch(
                model, 
                criterion,
                optimizer, 
                train_loader, 
                device,scaler=scaler,
                multi_loss_weight=multi_loss_weight
            )
        # 验证阶段
        metrics = validate_model(
            model, 
            val_loader, 
            device,
            AMP=AMP,
            criterion=criterion
        )
        if lr_scheduler_step!=0:
            lr_scheduler.step(metrics["Loss"])#动态学习率
        # 更新日志
        loss_history.append(epoch_loss)
        acc_history.append(metrics["Accuracy"])
        precision_history.append(metrics["Precision"])
        recall_history.append(metrics["Recall"])
        f1_history.append(metrics["F1"])
        ap_history.append(metrics["AP"])
        validation_loss_history.append(metrics["Loss"])

        # 根据加权指标更新最佳模型
        update_best_model(bestMod, metrics, epoch_loss, model, epoch,metrics_weights=metrics_weights)
        if epoch % final_checkpoint_interval ==0:
            #记录检测点
            bestMod.update_checkpoint(model, epoch)
        
        if epoch % show_progress_interval == 0:
            # 打印本周期结果
            print(
                    f'Epoch {epoch + 1}/{num_epochs}, train_Loss: {epoch_loss:.4f}, val_Loss: {metrics["Loss"]:.4f} ,'
                    f'Val Accuracy: {metrics["Accuracy"]:.4f}, '
                    f'Precision: {metrics["Precision"]:.4f}, Recall: {metrics["Recall"]:.4f}'
                    )
            print(f"当前最好的模型：{str(bestMod)}")

        # 更新 train_logs 对象
        train_logs.update(
            loss_lst=loss_history,
            val_accuracy_lst=acc_history,
            precision_lst=precision_history,
            recall_lst=recall_history,
            f1_lst=f1_history,
            AP_lst=ap_history,
            val_loss_history=validation_loss_history,
        )
    return bestMod, train_logs
