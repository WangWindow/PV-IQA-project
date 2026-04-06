# PV-IQA

基于 `method.md` 的掌静脉图像质量评估项目，覆盖：

1. 掌静脉识别基线训练
2. SDD + CR 双重约束伪标签生成
3. 轻量级 IQA 回归模型训练
4. 基于 ERC / EER / TAR@FAR 的生物识别效用评估

当前默认输入为**已经裁剪好的掌静脉 ROI 图像**，暂不包含 ROI 提取阶段。

## 技术栈

- `uv` 管理项目与环境
- `Python 3.14`
- `PyTorch 2.10+`
- Hugging Face / 开源工具库：`timm`、`evaluate`、`huggingface-hub`
- `wandb` 实验日志
- `hatchling` 构建

## 项目结构

```text
datasets/ROI_Data/           # 已裁剪的掌静脉 ROI 数据
configs/default.yaml         # 主配置
src/pv_iqa/
  cli.py                     # 命令行入口
  config.py                  # 配置模型
  train.py                   # 训练总入口：识别/IQA/伪标签
  eval.py                    # ERC / EER / TAR 评估
  detect.py                  # 单独推理与目录检测
  models/                    # 识别模型与 IQA 模型
  utils/                     # 数据、损失、指标、日志、伪标签工具
```

## 初始化

```bash
uv sync
```

## 典型流程

### 1. 生成元数据

```bash
uv run pv-iqa prepare-data
```

### 2. 训练识别基线并导出特征

```bash
uv run pv-iqa train-recognizer
```

### 3. 生成无监督质量伪标签

```bash
uv run pv-iqa generate-pseudo-labels
```

### 4. 训练 IQA 回归模型

```bash
uv run pv-iqa train-iqa
```

### 5. 评估 ERC

```bash
uv run pv-iqa evaluate
```

### 6. 对目录做质量打分

```bash
uv run pv-iqa detect-folder datasets/ROI_Data/ac_l
```

### 一键全流程

```bash
uv run pv-iqa run-all
```

## 关键配置

- `data.identity_mode`
  - `separate`：左右手分开建类
  - `merge_person`：左右手合并为同一身份
- `logger.wandb_mode`
  - `offline`：默认，适合本地实验
  - `disabled`：完全关闭 wandb
  - `online`：在线同步


## 说明

- 伪标签生成使用识别特征和分类器权重，分别构建 `QSDD` 与 `QCR`，再做归一化融合。
- IQA 回归模型采用轻量级 `MobileNetV3` 特征骨干，并加入局部混合与通道转置注意力模块。
- 评估阶段通过逐步排斥低质量样本，观察剩余集合上的 EER / TAR 变化，形成 ERC 曲线。
- 当前默认假设输入已经是 ROI 图像，因此不额外实现 ROI 检测与裁剪。
