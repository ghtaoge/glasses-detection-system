# 镜鉴：眼镜识别检测系统

本地运行的三分类眼镜检测教学工作台，固定识别未戴眼镜、普通眼镜和墨镜。系统支持数据导入、多人脸标注、数据版本、Web 训练、ONNX 推理、图片与摄像头识别及本地历史记录。

> 本项目用于计算机视觉学习和模型评估，不应用于身份判断、就业、执法、医疗、门禁或其他高风险自动决策。

## 本地启动

需要 Python 3.11-3.13、Node.js 24+。

```powershell
py -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\alembic -c backend/alembic.ini upgrade head
.venv\Scripts\python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

另开终端启动前端：

```powershell
cd frontend
npm install
npm run dev
```

访问 `http://127.0.0.1:5173`。

## 模型依赖

训练与 ONNX 推理安装：

```powershell
.venv\Scripts\python -m pip install -e ".[ml,dev]"
```

CUDA 版本的 PyTorch 必须按 PyTorch 官方安装器选择与本机驱动匹配的版本。首次资源下载需要联网；完成后数据标注、训练和推理均在本地执行。

公开教学数据源 `Glasses and Coverings` 采用 CC BY-NC 4.0，仅用于非商业教学。系统不会把数据集提交到仓库。

## 验证

```powershell
.venv\Scripts\python -m ruff check backend
.venv\Scripts\python -m pytest -q
npm --prefix frontend test
npm --prefix frontend run build
```
