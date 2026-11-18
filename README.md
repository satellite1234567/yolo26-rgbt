# yolo26-rgbt

基于 [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/) 的 **RGBT 双模态（可见光 + 热红外）目标检测**框架，面向羊只姿态检测（`standing` / `lying`）。

## 基础

| 项 | 值 |
|---|---|
| 上游源码 | `ultralytics` 8.4.215 |
| 基线模型 | YOLO26n（2,572,280 参数 / 6.1 GFLOPs） |
| 任务 | 双模态姿态检测（2 类） |

YOLO26 的关键特性：移除 DFL（`reg_max=1`）、端到端无 NMS 双头（one-to-many + one-to-one）、MuSGD 优化器、Progressive Loss + STAL。

## 改造内容

原版 Ultralytics 只支持 3 通道输入，无法直接读取「可见光 BGR + 红外灰度」的 4 通道数据。本项目的改动集中于此。

| 模块 | 位置 | 说明 |
|---|---|---|
| `MEFusion` | `ultralytics/nn/modules/conv.py` | 模态自适应加权融合：4 通道输入 → 逐样本预测模态权重 → 输出 3 通道，直接接 YOLO26 原生首层 |
| 4 通道数据加载 | `ultralytics/data/base.py` | 读取配对的 visible / infrared 图像并合并为 4 通道 |
| 模型配置 | `ultralytics/cfg/models/26/yolo26-mef.yaml` | 在 `yolo26.yaml` 首层前插入 `MEFusion` |

**设计要点**：`MEFusion` 输出 3 通道，因此 YOLO26 原生首层（`Conv in=3`）保持不变，COCO 预训练权重可完整迁移；相比「4 通道直进首层」，不牺牲预训练能力。

## 环境

```bash
python >= 3.9
torch >= 2.0
pip install -e .
```

在该目录下运行时，Python 会优先加载本地 `ultralytics/` 源码，不影响 pip 中已安装的 ultralytics 版本。

## 权重

`weights/*.pt` 与 `*.pt` 已由 `.gitignore` 排除，不纳入版本管理。

首次运行会自动下载：

```python
from ultralytics import YOLO
model = YOLO("weights/yolo26n.pt")   # 不存在时自动下载
```

也可从 [Ultralytics Releases](https://github.com/ultralytics/assets/releases) 手动下载 `yolo26n.pt` 放入 `weights/`。

## 数据格式

```
datasets/<name>/
├── visible/{train,val,test}/*.jpg + *.txt    # 标签与可见光图同目录
├── infrared/{train,val,test}/*.jpg           # 文件名与可见光一一对应
└── data.yaml
```

```yaml
path: /abs/path/to/datasets/<name>
train: visible/train
val: visible/val
test: visible/test

channels: 4

names:
  0: standing
  1: lying
```

## 使用

```bash
# 训练
python -m ultralytics train model=ultralytics/cfg/models/26/yolo26-mef.yaml \
    data=datasets/<name>/data.yaml imgsz=640 epochs=200 batch=8

# 验证
python -m ultralytics val model=runs/detect/train/weights/best.pt \
    data=datasets/<name>/data.yaml
```

## 说明

- YOLO26 的前向输出为 **tuple(dict)**（双头结构），与 YOLO11 的纯 Tensor 不同；自定义推理/评估代码不能直接复用 `non_max_suppression(out)`。
- 使用 one-to-many 头（默认）精度更高、需 NMS；`nms=False` 启用端到端无 NMS 推理。

## License

本项目基于 Ultralytics 修改，遵循 **AGPL-3.0**，详见 `LICENSE`。
