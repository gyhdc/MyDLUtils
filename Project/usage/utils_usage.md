下面是关于训练过程数据保存模块的使用文档，详细说明了各个类的功能、使用方法以及如何将它们集成到训练流程中，方便用户记录并保存训练配置、最佳模型指标和训练日志。

------

# 训练过程数据保存模块使用文档

本模块主要提供以下功能：

- **配置管理**：使用 `Config` 类加载、保存和管理训练参数和系统配置。
- **最佳模型指标管理**：使用 `BestSelector` 类记录最佳训练指标，并支持模型的保存和加载。
- **训练日志管理**：使用 `Logs` 类记录训练过程中的各项指标日志，支持 JSON 文件的读写。
- **整体保存流程**：通过 `saveProcess` 函数，将配置、最佳模型指标以及训练日志统一保存到指定目录。

------

## 1. Config 类

### 功能说明

`Config` 类用于记录训练参数和系统配置。该类可以：

- 从 JSON 文件或直接传入字典数据初始化；
- 将 JSON（字典）数据转换为对象的成员变量，方便直接访问各个配置项；
- 支持配置更新，并能将最新配置保存到文件中。

### 使用方法

#### 初始化

- **通过 JSON 文件加载配置**
   当只传入文件路径时，自动读取该文件内容，并设置成员变量。

  ```python
  config = Config("path/to/config.json")
  ```

- **通过直接传入参数创建配置**
   传入参数将会被存储为配置字典，并转换为成员变量。

  ```python
  config = Config(epochs=100, learning_rate=0.001, device="cuda")
  ```

#### 更新与保存

- **更新配置**
   调用 `update` 方法可以更新已有配置，并自动更新成员变量。

  ```python
  config.update(batch_size=32, optimizer="adam")
  ```

- **保存配置**
   调用 `save` 方法保存配置到指定路径（若路径为目录，则保存为 `config.json`）。

  ```python
  config.save("path/to/save_directory")
  ```

#### 示例

```python
# 从 JSON 文件加载配置
config = Config("config.json")
print(config)  # 输出所有配置项

# 直接创建配置
config = Config(epochs=100, learning_rate=0.001, device="cuda")
config.update(momentum=0.9)
config.save("saved_configs")
```

------

## 2. BestSelector 类

### 功能说明

`BestSelector` 类用于记录最佳训练指标，并管理最佳模型的保存与加载。该类会将最佳指标（包括模型、准确率、损失等）转换为成员变量，方便在训练过程中直接访问和比较。

### 使用方法

#### 初始化

- **通过 JSON 文件加载最佳指标**
   当传入文件路径时，会自动读取指标文件，并加载模型（若模型文件不存在，则使用默认路径）。

  ```python
  best_selector = BestSelector("path/to/metrics.json")
  ```

- **通过直接传入指标数据**
   直接传入各项指标数据。

  ```python
  best_selector = BestSelector(acc=0.0, loss=float('inf'))
  ```

#### 更新与保存

- **更新指标**
   使用 `update` 方法更新最佳指标，并自动更新为成员变量。

  ```python
  best_selector.update(acc=new_acc, loss=new_loss, model=model, precision=new_precision, recall=new_recall, ap=new_ap, epoch=current_epoch)
  ```

- **保存指标与模型**
   使用 `save` 方法，将模型保存为 `best.pth`，同时将其他指标保存到 `metrics.json` 文件中，保存到指定目录（若目录不存在，会自动创建）。

  ```python
  best_selector.save("saved_best_model")
  ```

#### 示例

```python
# 从文件加载最佳指标（同时加载模型）
best_selector = BestSelector("metrics.json")
print(best_selector)

# 更新最佳指标（在训练过程中）
best_selector.update(acc=0.95, loss=0.2, model=model, precision=0.94, recall=0.93, ap=0.92, epoch=5)

# 保存当前最佳模型和指标到指定目录
best_selector.save("saved_best_model")
```

------

## 3. Logs 类

### 功能说明

`Logs` 类用于记录训练过程中的各项指标日志（例如：训练损失、验证准确率、精确率、召回率、F1 分数、AP 等），并支持从 JSON 文件加载或保存日志数据。日志数据将转换为成员变量，便于访问和更新。

### 使用方法

#### 初始化

- **通过 JSON 文件加载日志**
   当只传入文件路径时，会自动读取 JSON 文件中的日志数据。

  ```python
  logs = Logs("path/to/logs.json")
  ```

- **通过直接传入日志数据**
   可以直接创建日志实例，并传入初始日志数据。

  ```python
  logs = Logs(train_loss_lst=[], val_acc_lst=[], precision_lst=[], recall_lst=[], f1_lst=[], ap_lst=[])
  ```

#### 更新与保存

- **更新日志数据**
   调用 `update` 方法增加或更新日志数据，同时更新成员变量。

  ```python
  logs.update(train_loss_lst=new_loss_list, val_acc_lst=new_accuracy_list)
  ```

- **保存日志**
   调用 `save` 方法保存日志到指定文件（如果保存路径为目录，则保存为 `logs.json`）。

  ```python
  logs.save("saved_logs")
  ```

#### 示例

```python
# 从 JSON 文件加载日志
logs = Logs("logs.json")
print(logs)

# 或直接创建空日志
logs = Logs(train_loss_lst=[], val_acc_lst=[], precision_lst=[], recall_lst=[], f1_lst=[], ap_lst=[])

# 更新日志数据
logs.update(train_loss_lst=[0.5, 0.4, 0.35], val_acc_lst=[0.8, 0.85, 0.9])
logs.save("saved_logs")
```

------

## 4. saveProcess 函数

### 功能说明

`saveProcess` 函数用于将配置、最佳模型指标以及训练日志统一保存到指定目录。它会执行以下操作：

- 检查并创建保存目录（如果不存在）；
- 调用 `BestSelector.save` 保存最佳模型及其指标；
- 调用 `Config.save` 保存配置文件；
- 调用 `Logs.save` 保存训练日志。

### 使用方法

#### 参数说明

- `saveDir`：保存文件的目录（字符串），若目录不存在则自动创建；
- `bestMod`：`BestSelector` 实例，包含最佳指标和模型；
- `train_log`：`Logs` 实例，记录训练日志数据；
- `config`：`Config` 实例，保存训练配置和参数。

#### 示例

```python
from your_module import Config, BestSelector, Logs, saveProcess

# 初始化配置、最佳指标和日志
config = Config("config.json")
best_selector = BestSelector("metrics.json")
logs = Logs("logs.json")

# 在训练结束后调用 saveProcess 将所有数据保存到一个目录中
save_directory = "saved_training_data"
saveProcess(save_directory, best_selector, logs, config)
```

保存后，该目录下将生成：

- `config.json`：保存训练配置。
- `metrics.json` 与 `best.pth`：分别保存最佳指标和最佳模型权重。
- `logs.json`：保存训练日志数据。

------

## 综合示例

下面给出一个将各个模块结合在一起的综合示例，展示如何在训练流程中记录和保存数据：

```python
import torch
from your_module import Config, BestSelector, Logs, saveProcess

# 假设模型、训练过程等已定义
# model = ...
# optimizer = ...
# criterion = ...
# train_loader = ...
# val_loader = ...

# 1. 加载或创建配置
config = Config(epochs=100, learning_rate=0.001, device="cuda")

# 2. 初始化最佳模型指标（初始时设置默认值）
best_selector = BestSelector(acc=0.0, loss=float('inf'))

# 3. 初始化日志（训练过程中记录各指标变化）
logs = Logs(train_loss_lst=[], val_acc_lst=[], precision_lst=[], recall_lst=[], f1_lst=[], ap_lst=[])

# 4. 在训练过程中更新 best_selector 和 logs（此处为伪代码）
for epoch in range(config.epochs):
    # 执行训练与验证过程，得到当前指标
    # epoch_loss, val_accuracy, precision, recall, f1, ap = train_and_validate(...)
    
    # 示例：更新日志
    logs.update(train_loss_lst=logs.train_loss_lst + [epoch_loss],
                val_acc_lst=logs.val_acc_lst + [val_accuracy],
                precision_lst=logs.precision_lst + [precision],
                recall_lst=logs.recall_lst + [recall],
                f1_lst=logs.f1_lst + [f1],
                ap_lst=logs.ap_lst + [ap])
    
    # 根据当前指标更新最佳模型（内部包含比较逻辑）
    best_selector.update(acc=val_accuracy, loss=epoch_loss, model=model, precision=precision, recall=recall, ap=ap, epoch=epoch)

# 5. 训练结束后，将所有数据保存到指定目录
save_directory = "saved_training_data"
saveProcess(save_directory, best_selector, logs, config)
```

------

## 总结

通过本模块提供的 `Config`、`BestSelector` 和 `Logs` 类以及 `saveProcess` 函数，用户可以轻松实现以下功能：

- **加载和保存训练配置**：方便参数管理与实验复现。
- **记录和更新最佳模型指标**：保存模型权重和相关指标，便于后续恢复或比较。
- **记录训练过程日志**：追踪训练损失、准确率等指标变化，为调试和结果分析提供数据支持。
- **统一保存训练数据**：将所有关键数据（配置、指标、日志）保存到指定目录，简化训练结束后的数据管理。

用户只需按照文档示例调用相应接口，即可将训练过程中的数据完整保存，方便后续分析和模型部署。