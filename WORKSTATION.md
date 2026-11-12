# SteelScope 钢材缺陷工作台

在项目目录运行下面的 PowerShell 命令。它使用项目已有的 `.venv`。当前环境验证使用 Python 3.9、PySide6、PyTorch 和 `ultralytics==8.3.100`。如需迁移到另一台电脑，在目标机重建虚拟环境，安装匹配硬件的 PyTorch 与 torchvision 以及 `requirements.txt` 中的依赖。

```powershell
.\launch_workstation.ps1
```

也可以执行 `.\.venv\Scripts\python.exe main.py`。左侧导航包含钢材检测流程和通用推理工具。单模型检测和双模型对比沿用原 YOLOSHOW 的推理功能。`ptfiles/yolo11n_neu_det.pt` 存在时登记为原有 NEU-DET 模型，`ptfiles/yolo11n.pt` 可作为训练起点。

右上角可以切换「简体中文 / English」。切换会更新各页面的界面文字，语言选择会在下次启动时恢复。数据集的类别编号、原始训练日志和已画在图像上的模型输出保留其记录时的内容。

1. **采集与标注。** 打开本地视频或按编号打开 USB 相机，点击「抓帧到标注区」。拖动鼠标画框，单击现有框可选中，再修改类别或删除重画。「模型预标注」使用当前启用模型生成草稿。核对完所有框后点「审核并保存」，确认画面确实没有缺陷时点「确认无缺陷」。检测页也可以把当前原始帧送到标注页。
2. **生成数据集。** 选择基础 YAML 时，审核过的新图加入训练集，原验证和测试划分保留。基础留空时按来源或填写的工件批次分组，至少需要两个含缺陷标注的分组。应用检查不同划分是否包含字节完全相同的图片。导出成功后自动填入训练页。
3. **可视化训练。** 选择本地 YOLO YAML 或恰好包含一个 YAML 的目录，指定 `.pt`、轮数、批大小、输入尺寸和设备。YAML 中相对路径以配置文件所在目录为基准。缺少标注文件会报错，确认无缺陷的图片使用空 TXT。第一次验证可以选少量样本、两轮和 `128` 输入尺寸。训练与工位推理分时运行，停止请求在本轮验证和权重保存后生效。
4. **历史与模型。** 每次任务保存所用数据快照、参数、日志、曲线和权重。选择训练完成的模型，用固定样图试跑，再点「启用所选模型」。点击「切回上一版」可在前后两个版本之间切换。重启后根据已保存的选择加载模型。
5. **现场检测。** 打开相机或视频，每次会话固定所选模型与阈值。帧记录包含时间、来源、模型版本、工件编号（如填写）、检测框和耗时。相机打开或读取失败会记录为未完成检测。会话和结果通过「打开检测记录」查看。

运行数据位于 `runs/workstation`，包括 `annotations`、`datasets`、`jobs`、`inspections` 和 `selection.json`。迁移模型时应保留相应任务目录中的 `run.json`、`training/weights/best.pt` 和 `selection.json`。回退到原有模型还需要保留 `ptfiles/yolo11n_neu_det.pt`。模型权重应来自可信的训练环境。

比较模型时，可以对同一固定数据集用相同尺寸、批大小与设备分别评估两个权重。例如，先将第二条命令中的 `<任务编号>` 替换成模型库显示的编号。

```powershell
.\.venv\Scripts\python.exe train_neu_det.py --evaluate --split test --imgsz 640 --batch 4 --device 0 --model ptfiles\yolo11n_neu_det.pt --name reference
.\.venv\Scripts\python.exe train_neu_det.py --evaluate --split test --imgsz 640 --batch 4 --device 0 --model 'runs\workstation\jobs\<任务编号>\training\weights\best.pt' --name candidate
```

工位验收应另用实际相机、照明和有人工结论的工件样本记录误报、漏检及处理耗时。检测页按帧记录预测结果，工件级判定应在明确的工件触发与汇总规则下单独验证。
