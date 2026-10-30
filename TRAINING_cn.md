# NEU-DET 六类缺陷检测训练

本教程使用项目里的 `data/IMAGES`、`data/ANNOTATIONS` 和 `ptfiles/yolo11n.pt`，
训练目标检测模型。XML 中只有边界框标注，因此本流程训练的是目标检测任务。

以下命令在项目根目录的 PowerShell 中执行。

## 1. 准备数据（本次已生成默认目录）

```powershell
.\.venv\Scripts\python.exe tools\prepare_neu_det.py
```

脚本复制原始图片、转换所有 XML 中的目标框，不修改原始数据。
原始 XML 的部分 filename 字段没有扩展名，脚本通过 XML 与图片的同名主体匹配。
一张图中可能有多类缺陷，转换时会保留全部对象，不能只按文件名给对象分类。
超出图片范围的框会裁剪到图片边界，裁剪数量记录在划分清单中。

按文件名中的主要类别分层，使用种子 42 按 8:1:1 划分：

| 子集 | 图片数 | 用途 |
| --- | ---: | --- |
| train | 1440 | 更新模型参数 |
| val | 180 | 选取最佳权重、调整超参数 |
| test | 180 | 方案确定后进行最终评测 |

这是本项目自行生成的随机划分，不代表官方基准划分；若样本来自同一工件或连续帧，
正式实验应按工件或采集批次分组划分，避免相关样本跨子集。

输出目录为 `datasets/NEU-DET-v1`，包含 `images`、`labels`、`neu_det.yaml` 和
记录每个子集文件名的 `split_manifest.json`。再次执行时不会覆盖已有目录；
需要重新转换时，使用 `--output datasets/NEU-DET-v2`，训练时相应指定 `--data`。

YOLO 标签每行是 `类别编号 中心x 中心y 宽 高`，坐标归一化到 0～1。
固定类别顺序为：crazing、inclusion、patches、pitted_surface、rolled-in_scale、scratches。

## 2. 配置 GPU

2026-10-28 检查到本机有 RTX 2060 6GB，但项目虚拟环境使用 `torch 2.8.0+cpu`，
`torch.cuda.is_available()` 为 False。当前 CPU 环境可以做流程验证；使用显卡训练需要
安装对应 CUDA 构建的 PyTorch。

本机有 uv，虚拟环境没有 pip。可执行以下命令，将 PyTorch 和 torchvision
更换为匹配的 CUDA 12.6 构建；安装需要联网下载较大的依赖。

```powershell
uv pip install --python .\.venv\Scripts\python.exe --reinstall-package torch --reinstall-package torchvision torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe train_neu_det.py --check --device 0
```

保持项目使用的 `ultralytics==8.3.100`，便于与 GUI 的模型加载代码保持一致。

## 3. 启动训练

先用少量轮次检查数据、模型加载和显存是否正常：

```powershell
.\.venv\Scripts\python.exe train_neu_det.py --epochs 1 --batch 4 --device 0 --name neu_smoke
```

再训练基线模型：

```powershell
.\.venv\Scripts\python.exe train_neu_det.py --epochs 100 --imgsz 640 --batch 8 --device 0
```

这是一套起始参数，不保证最优效果或显存占用。出现 CUDA out of memory 时，先将
`--batch 8` 减为 `--batch 4` 或 `--batch 2`。默认 `workers=0`，便于 Windows 首次运行。
模型以已有权重初始化，类别数根据数据配置调整为六类。

CPU 流程验证可显式使用以下命令；一次 epoch 仍然会遍历全部训练图片：

```powershell
.\.venv\Scripts\python.exe train_neu_det.py --epochs 1 --imgsz 320 --batch 2 --device cpu --name neu_cpu_smoke
```

首次实际运行时，Ultralytics 可能下载字体或 AMP 检查使用的参考权重。

## 4. 查看结果与评测

默认首次完整训练输出 `runs/detect/neu_yolo11n`。若目录已存在，Ultralytics 自动增加后缀，
以训练结束时终端打印的实际权重路径为准。

- `weights/best.pt`：按验证集指标选出的最佳权重。
- `weights/last.pt`：最后一轮的权重，可用于恢复中断训练。
- `results.csv`、`results.png`：损失及验证集指标。
- `confusion_matrix.png`：混淆矩阵，用于观察类别混淆。

关注 Precision、Recall、mAP50 和更严格的 mAP50-95。损失下降并不保证未见样本效果提高。
实验方案确定后，再使用保留测试集进行最终评测：

```powershell
.\.venv\Scripts\python.exe train_neu_det.py --evaluate --model runs\detect\neu_yolo11n\weights\best.pt --split test --device 0
```

恢复未完成的训练：

```powershell
.\.venv\Scripts\yolo.exe detect train model=runs/detect/neu_yolo11n/weights/last.pt resume=True
```

## 5. 导入 YOLOSHOW

将训练结果复制并命名为包含 `yolo11` 的文件，方便 GUI 判断模型版本。

```powershell
Copy-Item .\runs\detect\neu_yolo11n\weights\best.pt .\ptfiles\yolo11n_neu_det.pt
.\.venv\Scripts\python.exe main.py
```

在模型下拉框选择 `yolo11n_neu_det.pt`，选取图片并开始识别。
也可以先把 best.pt 复制重命名为 `yolo11n_neu_det.pt`，再在 Settings 中通过 Import Model 导入。

## 参考资料

- [Ultralytics 训练说明](https://docs.ultralytics.com/modes/train/)
- [YOLO 数据集和标注格式](https://docs.ultralytics.com/datasets/detect/)
- [PyTorch 2.8.0 的安装命令](https://pytorch.org/get-started/previous-versions/)

本教程参数以本地 Ultralytics 8.3.100 为准，在线文档中的新模型和新参数可能不适用于本项目。
