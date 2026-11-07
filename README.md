# YOLO 钢材表面缺陷检测工作台 · Steel Surface Defect Workbench

[简体中文](#简体中文) · [English](#english) · [安装与启动 / Setup](#setup) · [实验结果 / Results](#results)

## 简体中文

这是一个基于 YOLOSHOW 扩展的本地桌面工作台。它使用 YOLO11n 检测钢材表面缺陷，把相机和视频检测、抓帧标注、数据集版本管理、可视化训练和模型切换放在同一个 PySide6 界面里。右上角可切换简体中文和 English，语言选择会在重启后恢复。

### 工作流程

| 模块 | 实现 |
| --- | --- |
| 现场检测 | OpenCV 后台线程读取 USB 相机或视频，调用当前启用的模型，预览缺陷框并逐帧保存检测结果、耗时和模型版本。 [检测代码](workstation/inspection_page.py) |
| 采集与标注 | 抓取原始帧，手动画框或修正模型预标注。人工审核后保存同名 PNG 和 YOLO TXT，确认无缺陷的图片使用空标签。 [标注代码](workstation/annotations.py) |
| 数据集管理 | 导出已审核样本，校验标签并保存数据快照。选用基础数据集时，新样本进入训练集，原验证和测试划分保留。 [数据快照代码](workstation/dataset.py) |
| 可视化训练 | 选择数据、初始权重和训练参数。独立进程运行 Ultralytics，界面逐轮显示损失、验证 mAP 和日志。 [训练进程](workstation/train_worker.py) |
| 训练历史与模型库 | 按任务保留数据版本、参数、指标和权重。用样图试跑候选模型，再手动启用或切回上一版。 [模型库代码](workstation/store.py) |

例如，发现一帧漏检了划痕，可以抓取原图、补框并审核，导出新版数据集再训练。训练完成后，在模型库用样图试跑，结合固定验证集的结果决定是否启用。

## English

This desktop workbench extends YOLOSHOW with a YOLO11n workflow for steel surface defect detection. It brings video and camera inspection, frame annotation, dataset versioning, training, and model activation into a PySide6 interface. The language selector switches between English and Simplified Chinese and remembers the selection.

1. **Inspect** a USB camera feed or local video. The workbench overlays detections and records each processed frame with its timing and model version.
2. **Annotate** a captured raw frame by drawing boxes or correcting model suggestions. Human-reviewed annotations are saved as YOLO labels. Confirmed defect-free frames have empty label files.
3. **Export** a versioned dataset snapshot. When a base dataset is selected, new samples go into its training split and existing validation and test assignments are preserved.
4. **Train** with a selected dataset and checkpoint. A separate Python process runs Ultralytics while the GUI displays per-epoch losses, validation mAP, and logs.
5. **Review and activate** a completed model. Inspect the run history, try the candidate on a sample image, and activate it or roll back to the previous version.

## Setup

### 安装与启动 · Install and launch

在 Windows PowerShell 中运行。以下命令使用 Python 3.9 和 CPU 版 PyTorch。使用 NVIDIA GPU 时，请将 PyTorch 安装命令替换为与本机驱动兼容的 CUDA 版本。

Run these commands in Windows PowerShell. They use Python 3.9 and the CPU build of PyTorch. For NVIDIA GPU acceleration, install a CUDA build of PyTorch and torchvision compatible with your driver.

~~~powershell
git clone https://github.com/lnzz21/yolo-defect-detection.git
cd yolo-defect-detection
py -3.9 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py --workstation
~~~

### 数据与权重 · Data and checkpoints

数据集、模型权重和运行记录存放在本地。首次运行时，准备与你要执行的步骤对应的文件。

Datasets, checkpoints, and run outputs are stored locally. Prepare the files needed for the steps you plan to run.

- **初始权重 / Training checkpoint** — 将官方 YOLO11n 预训练权重放到 `ptfiles/yolo11n.pt`，也可以在训练页选择本地 YOLO11 检测权重。 Place the official pretrained YOLO11n checkpoint at this path or select a local YOLO11 detection checkpoint in the training tab.
- **训练数据 / Training data** — 将 NEU-DET 原始图片和 XML 分别放在 `data/IMAGES/` 和 `data/ANNOTATIONS/`，执行下方转换命令。 Place the original images and XML annotations in those directories and run the conversion below.
- **检测权重 / Inspection checkpoint** — 在模型库启用工作台训练完成的模型。已有六类 NEU-DET 权重也可放到 `ptfiles/yolo11n_neu_det.pt`，重新打开工作台后会出现在模型库。 Activate a completed workbench model, or place an existing six-class NEU-DET checkpoint at this path and reopen the workbench to register it.

~~~powershell
.\.venv\Scripts\python.exe tools\prepare_neu_det.py
~~~

转换后选择 `datasets/NEU-DET-v1/neu_det.yaml` 进行训练。也可以直接选用符合 YOLO 格式的现有数据集。

Select the generated YAML in the training tab. You can also select an existing YOLO-format dataset.

## Results

### NEU-DET 验证集实验 · Validation results

项目将 1,800 张 NEU-DET 图片划分为 1,440 张训练图、180 张验证图和 180 张测试图。类别顺序为 `crazing`、`inclusion`、`patches`、`pitted_surface`、`rolled-in_scale`、`scratches`。

The project split contains 1,440 training, 180 validation, and 180 test images. The six class IDs follow the order listed above.

| 微调范围 / Fine-tuning scope | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| --- | ---: | ---: | ---: | ---: |
| 全参数 / Full model | 72.93% | 70.55% | **76.65%** | **44.88%** |
| 仅检测头 / Detect head only | 70.80% | 42.46% | 49.73% | 22.48% |
| Neck + Detect | 67.71% | **75.55%** | 75.85% | 44.01% |

三种方案使用相同的初始权重、数据划分和随机种子，表中指标来自同一验证集。全参数微调在这次对比中取得最高 mAP，Neck + Detect 取得最高召回率。

All three runs used the same initial weights, data split, and random seed. On the shared validation set, full fine-tuning achieved the highest mAP and Neck + Detect achieved the highest recall.

## 文档与许可 · Documentation and license

- [工作台操作、输出目录与模型评估 / Workbench guide (中文)](WORKSTATION.md)
- [NEU-DET 数据转换、命令行训练与评估 / NEU-DET training guide (中文)](TRAINING_cn.md)
- [检测头对比脚本 / Detect-head experiment](tools/train_head_only.py) · [Neck + Detect 对比脚本 / Neck + Detect experiment](tools/train_neck_detect.py)
- 本项目基于 YOLOSHOW，采用 AGPL-3.0。 Built on YOLOSHOW and licensed under AGPL-3.0. [YOLOSHOW](https://github.com/SwimmingLiu/YOLOSHOW) · [原版功能说明 / Original feature guide (中文)](README_cn.md) · [LICENSE](LICENSE)
