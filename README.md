# PV-IQA

无监督掌静脉图像质量评估。基于识别特征空间的双分量伪标签 + 多尺度回归训练。

## 算法

质量分数由两个分量在识别嵌入空间中融合计算：

$$
Q = 100 \times \text{minmax}(Q^P + \beta \cdot \text{WD})
$$

| 分量 | 来源 | 含义 |
|------|------|------|
| $Q^P$ | PGRG (Zou et al., 2023) | 类内余弦相似度均值 — 特征聚合紧密度 |
| WD | SDD-FIQA (Ou et al., 2021) | 类内/类间 cosine 相似度分布 Wasserstein 距离 |

具体流程：

1. **ArcFace 识别器** — MobileNetV3 + ArcMarginHead 在类别隔离的训练集上训练识别器
2. **特征导出** — 提取 L2 归一化嵌入向量，按类别计算 $Q^P$ 和 WD
3. **伪标签生成** — $Q^P + \beta \cdot \text{WD}$ → minmax 归一化到 [0, 100]；支持 `ours` / `sdd` / `qp_only` 三种模式
4. **IQA 回归训练** — 多尺度 MobileNetV3 backbone 在伪标签上学习质量回归

## IQA 模型架构

多尺度 backbone (`features_only=True`, `out_indices=(1,3,4)`) → 三分支提取：

| 分支 | 输入 | 输出 | 语义 |
|------|------|------|------|
| branch1 | stage1 特征图 | 32d | 纹理/清晰度 |
| branch3 | stage3 特征图 | 64d | 中层结构 |
| branch4 | stage4 特征图 + SE-Attn | $C_4$ | 全局语义 |

concat → FC(128) → FC(1)，线性输出（无激活函数）。

**损失函数**：Huber + LabelRank + DegradeRank

- **LabelRank**：对伪标签分数差距足够大的样本对施加排序约束
- **DegradeRank**：对低质量样本施加合成退化（高斯模糊/过曝/欠曝/遮挡），约束 mild 退化分数 > severe 退化分数

## 用法

### 训练

```bash
uv sync
uv run python run.py
```

全流程：`build_metadata → train_recognizer → export_features → generate_pseudo_labels → train_iqa → export_onnx`

产物输出至 `checkpoints/<name>/iqa/best.pt`，同时导出 ONNX 模型 `best.onnx`。

### 评估

```python
from pv_iqa.eval import evaluate_err_roi, evaluate_eer_aoc

evaluate_err_roi(config, ckpt_path)      # AUC / ScoreGap / Overlap
evaluate_eer_aoc(config, ckpt_path)      # EER-AOC 曲线（类别隔离测试集）
```

支持指标：MAE, RMSE, Pearson/Spearman 相关系数, 排序准确率, EER (1e-2/1e-3/1e-4 FAR), AOC。

### 关键配置

修改 `src/pv_iqa/config.py`：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data_root` | `HFUT-PV-ROI-origin` | 数据集路径 |
| `recog_backbone` | `mobilenetv3_large_100` | 识别器 backbone |
| `iqa_backbone` | `mobilenetv3_large_100` | IQA 回归 backbone |
| `pseudo_beta` | `1.0` | WD 权重（0=禁用） |
| `pseudo_mode` | `ours` | 伪标签模式：`ours` / `sdd` / `qp_only` |
| `iqa_epochs` | `40` | IQA 训练轮数 |
| `iqa_huber_delta` | `2.0` | Huber 损失 delta |
| `iqa_rank_weight` | `0.2` | LabelRank 损失权重 |
| `iqa_degrade_rank_weight` | `0.5` | DegradeRank 损失权重 |
| `image_size` | `224` | 输入分辨率 |
| `seed` | `2026` | 随机种子 |

## Web 演示

```bash
cd app && bun install && bun restart
```

- 前端 `http://localhost:6006` — React 19 + shadcn/ui + Recharts
- API `http://localhost:6005` — FastAPI，支持 Python / Rust CPU / Rust CUDA 三引擎切换

## 目录

```
checkpoints/<run>/       训练产物（best.pt / best.onnx）
datasets/<dataset>/      掌静脉 ROI 数据集
src/pv_iqa/              算法核心（配置 / 模型 / 训练 / 伪标签 / 评估）
run.py                   训练入口
app/                     Bun + React 前端
app/backend/             FastAPI 后端（评分 / 任务 / 认证）
```

## 依赖

Python ≥3.14, PyTorch ≥2.11, timm, ONNX, FastAPI, wandb。

> [!NOTE]
> 训练需 CUDA 环境。CPU 推理可通过 PV-IQA-rs CLI 支持。
