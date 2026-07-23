# 镜鉴：眼镜识别检测系统

本地运行的三分类眼镜检测教学工作台，固定识别未戴眼镜、普通眼镜和墨镜。系统支持数据导入、多人脸标注、数据版本、Web 训练、ONNX 推理、图片与摄像头识别及本地历史记录。

> 本项目用于计算机视觉学习和模型评估，不应用于身份判断、就业、执法、医疗、门禁或其他高风险自动决策。

## 本地启动

需要 Python 3.11-3.13、Node.js 24+。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start.ps1
```

访问 `http://127.0.0.1:5173`。

## 模型依赖

训练与 ONNX 推理安装：

```powershell
.venv\Scripts\python -m pip install -e ".[ml,dev]"
```

CUDA 版本的 PyTorch 必须按 PyTorch 官方安装器选择与本机驱动匹配的版本。首次资源下载需要联网；完成后数据标注、训练和推理均在本地执行。

模型库可安装 `mantasu/glasses-detector` 发布的 MIT 许可预训练分类权重。系统将“任意眼镜”和“墨镜”分类器合并导出为本地 ONNX，并结合 Apache-2.0 许可的 YuNet 人脸检测器，对每张人脸互斥输出未戴眼镜、普通眼镜或墨镜。公开指标为任意眼镜 F1 0.9693、墨镜 F1 0.9311；这些分类 F1 不会显示为检测 mAP。

公开教学数据源 `Glasses and Coverings` 采用 CC BY-NC 4.0，仅用于非商业教学。系统不会把数据集提交到仓库。

系统包含概览、数据集、标注工作台、训练评估、模型库、识别中心和历史记录七个页面。详细操作见 [本地运维指南](docs/OPERATIONS.md)，数据处理边界见 [隐私与使用边界](docs/PRIVACY.md)。

## 验证

```powershell
.venv\Scripts\python -m ruff check backend
.venv\Scripts\python -m pytest -q
npm --prefix frontend test
npm --prefix frontend run build
```
