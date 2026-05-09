# PV-IQA

无监督掌静脉图像质量评估。

## 核心算法

$$
Q = 100 \times \text{minmax}\big(Q^P + \alpha \cdot \text{CR} + \beta \cdot \text{WD}\big)
$$

| 分量 | 来源 | 公式 |
|------|------|------|
| $Q^P$ | PGRG (Zou et al., 2023) | 类内余弦相似度均值 |
| CR | CR-FIQA (Boutros et al., 2021) | $\cos(e_i, w_y) \;/\; (\max_{j \neq y} \cos(e_i, w_j) + 1 + \varepsilon)$ |
| WD | SDD-FIQA (Ou et al., 2021) | 类内/类间 Wasserstein 距离 |

## 训练

```bash
uv sync
uv run python train.py
```

全流程：`build_metadata → train_recognizer → export_features → generate_pseudo_labels → train_iqa → export_onnx`

<details>
<summary>关键配置</summary>

| 参数 | 示例值 | 说明 |
|------|--------|------|
| `iqa_backbone` | `mobilenetv3_large_100` | IQA 回归模型 backbone |
| `pseudo_alpha` | `0.5` | CR 分量权重 |
| `pseudo_beta` | `0.0` | WD 分量权重（0=禁用） |
| `iqa_epochs` | `20` | IQA 训练轮数 |
| `iqa_huber_delta` | `0.1` | Huber loss delta |
| `iqa_rank_weight` | `0.3` | 排序损失权重 |

修改 `src/pv_iqa/config.py` 调整参数。
</details>

## Web 演示

```bash
cd app && bun install && bun restart
```

- 前端: `http://localhost:6006`
- API: `http://localhost:6005`
- 支持 Python/Rust CPU/Rust CUDA 三引擎切换

## 目录

```
checkpoints/<run>/iqa/   训练产物（best.pt / best.onnx / best.onnx.json）
datasets/<dataset>/      掌静脉 ROI 数据
src/pv_iqa/              算法实现（训练/推理/伪标签/评估）
app/                     Bun + React 前端
app/backend              FastAPI 后端
```

> [!NOTE]
> 依赖 CUDA 环境进行训练。仅推理的 CPU 模式通过 `PV-IQA-rs` CLI 支持。
