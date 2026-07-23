# 本地运维指南

## 安装与启动

在 PowerShell 中运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start.ps1
```

打开 `http://127.0.0.1:5173`。程序只绑定回环地址；图片、模型、训练指标和历史结果默认保存在项目的 `data/` 中。

## 标准工作流

1. 在“数据集”创建数据集，导入三个类别的图片。
2. 在“标注工作台”复核每张图片的人脸框，随后发布不可变版本。
3. 在“训练评估”先做快速试跑，再用标准预设训练。
4. 在“模型库”查看冻结测试集指标。只有总体 `mAP@0.5 >= 0.80` 的模型可启用。
5. 在“识别中心”上传图片或显式启动摄像头。普通摄像头帧不会保存；只有“保存快照”会写入历史。
6. 在“历史记录”查看或删除原图和结果图。

## 资源与故障

- 首次自动标注需要下载固定校验值的 YuNet 模型，首次训练需要 `yolo26n.pt`。
- CPU 可直接使用默认 PyTorch；CUDA 版必须按 PyTorch 官方选择器安装与驱动匹配的构建。
- 显存不足时降低批量大小或图像尺寸后创建新训练任务。
- 服务重启会把未结束训练标为 `interrupted`，不会伪报成功。
- 备份时停止服务并复制整个 `data/` 目录；数据库中的文件路径均相对该目录。

完整检查：`powershell -NoProfile -ExecutionPolicy Bypass -File scripts/test.ps1`。
