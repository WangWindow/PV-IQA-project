# PV-IQA

基于深度学习的掌静脉图像质量评估项目，覆盖：

1. 掌静脉识别基线训练
2. SDD + CR 双重约束伪标签生成
3. 轻量级 IQA 回归模型训练
4. 基于 ERC / EER / TAR@FAR 的生物识别效用评估

当前默认输入为**已经裁剪好的掌静脉 ROI 图像**，暂不包含 ROI 提取阶段。

## 代码工具

- `uv` 管理项目与环境
- `Python 3.14`
- `PyTorch 2.10+`
- Hugging Face / 开源工具库：`timm`、`evaluate`、`huggingface-hub`
- `wandb` 实验日志
- `hatchling` 构建


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
- `iqa.use_waveformer_layer`
  - `false`：默认，保持现有 MobileNetV3 + TAB 结构
  - `true`：在 IQA 高层特征分支插入 WaveFormer 风格频域增强层

----

## 演示前端

前端位于 `app/`，采用 `Vite + React + Tailwind CSS + shadcn + Bun`，并增加了 `react-router-dom` 路由结构与 `motion` 动画。

当前支持两条推理路径：

- `Python`：沿用 `uv run pv-iqa ...`
- `Rust Candle`：由 Bun 调用 `/root/autodl-tmp/PV-IQA-rs` 中的常驻服务

### 开发启动

```bash
cd app
bun install
bun run dev
```

- 前端开发页：`http://localhost:6006`
- Bun API：默认 `http://localhost:6005`
- 如需改端口，可在启动前设置 `PV_IQA_API_PORT` 或 `PORT`，前端代理会自动跟随

### 生产启动

```bash
cd app
bun run build
bun run start
```

生产模式下由 Bun 在 `6006` 同时提供前端静态资源和 API。

### 前端能力

- 支持**单图**与**文件夹**两种评分模式
- 支持**拖拽上传**
- 支持**异步任务状态**、阶段文本、已处理数量
- 支持**历史任务查看与彻底删除**
- 支持在工作台中切换 **Python / Rust** 推理后端
- 页面拆分为：
  - `/workspace`：评分工作台
  - `/jobs`：任务管理与结果查看

### 运行与存储说明

- 默认会自动选择 `checkpoints/` 下最近且包含 `iqa/best.pt` 的 run
- 可通过环境变量 `PV_IQA_RUN_NAME` 指定默认推理 run
- SQLite 数据库存放在 `app/data/demo.sqlite`
- 上传文件存放在 `app/data/uploads/<jobId>/`
- 删除历史任务时，会同时删除数据库记录和该任务对应的上传文件

### Rust / Candle 推理后端

Rust 项目位于 `/root/autodl-tmp/PV-IQA-rs`，提供：

- `GET /health`
- `POST /score/image`
- `POST /score/batch`

#### 导出 Rust 可用模型

```bash
uv run pv-iqa export-rust-model --run-name <run-name>
```

默认会把导出产物写到对应 checkpoint 同目录：

```text
checkpoints/<run-name>/iqa/best.pt
checkpoints/<run-name>/iqa/best.onnx
checkpoints/<run-name>/iqa/best.onnx.json
```

- `best.onnx` 为**单文件** ONNX，Rust 侧直接加载
- 如果导出过程中产生 `best.onnx.data`，工具会自动内联权重并删除该 sidecar
- 导出逻辑位于 `src/pv_iqa/utils/export_onnx.py`

#### 手动启动 Rust 服务

```bash
cd /root/autodl-tmp/PV-IQA-rs
PV_IQA_REPO_ROOT=/root/workspace/PV-IQA cargo run --release
```

Rust 服务默认监听 `127.0.0.1:7007`，可通过以下环境变量覆盖：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PV_IQA_REPO_ROOT` | `/root/workspace/PV-IQA` | PV-IQA 仓库根目录 |
| `PV_IQA_RS_HOST` | `127.0.0.1` | Rust 服务监听地址 |
| `PV_IQA_RS_PORT` | `7007` | Rust 服务监听端口 |
| `PV_IQA_RS_DEVICE` | `auto` | 运行设备：`auto` / `cpu` / `cuda` |
| `PV_IQA_RS_CUDA_ORDINAL` | `0` | CUDA 设备编号 |

#### Bun 桥接层环境变量

`app/server.ts` 还支持以下 Rust 相关配置：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PV_IQA_RS_URL` | `http://127.0.0.1:7007` | Bun 访问 Rust 服务的地址 |
| `PV_IQA_RS_PROJECT_ROOT` | `/root/autodl-tmp/PV-IQA-rs` | Rust 项目目录；仅在缺少可执行文件时用于构建/发布 |
| `PV_IQA_RS_BINARY` | `""` | 指定要直接启动的 Rust 可执行文件；可用仓库相对路径或绝对路径 |
| `PV_IQA_RS_AUTOSTART` | `1` | Rust 不可用时是否自动拉起已发布二进制 |
| `PV_IQA_RS_RELEASE` | `1` | 缺少二进制时，发布步骤默认使用 `cargo build --release` |
| `PV_IQA_RS_CARGO_FEATURES` | `""` | 透传给 Cargo 的 feature；CUDA 用 `cuda`，可选 `cudnn` |
| `PV_IQA_RS_HEALTH_TIMEOUT_MS` | `1500` | 健康检查超时 |
| `PV_IQA_RS_REQUEST_TIMEOUT_MS` | `300000` | Rust 推理请求超时 |
| `PV_IQA_RS_START_TIMEOUT_MS` | `300000` | 自动拉起服务后的等待时长 |

当工作台选择 `Rust` 后端时：

1. Bun 会先确认 `best.onnx` / `best.onnx.json` 是否存在
2. 若缺失则自动执行 `uv run pv-iqa export-rust-model --run-name <run-name>`
3. 若 `bin/` 中已有匹配当前配置的二进制，则直接使用最新版本启动
4. 若缺少二进制，则从 `PV_IQA_RS_PROJECT_ROOT` 构建一次，并复制为 `bin/pv-iqa-rs-<profile>-<feature>-<timestamp>`
5. 任务结果会在 SQLite 中记录所使用的 `backend`

### 性能对比（当前环境）

开启 release 模式并保持 Rust 服务常驻后，当前环境下的实测结果为：

| 场景 | 耗时 |
| --- | --- |
| Rust CPU release `/score/image` warm 单图平均耗时 | `~0.45s` |
| Rust CUDA release `/score/image` warm 单图平均耗时 | `~0.28s` |
| Bun `/api/score/image` 单图任务从提交到完成（Rust CPU release） | `~0.48s` |

如果看到 4s 以上的耗时，通常不是模型计算本身慢，而是 Rust 服务以 debug 版本运行，或 `bin/` 中还没有现成二进制，导致首次发布阶段触发了 `cargo build`。正常情况下，后续启动都会直接复用 `PV-IQA/bin` 中已发布的可执行文件。


## 说明

- 伪标签生成使用识别特征和分类器权重，分别构建 `QSDD` 与 `QCR`，再做归一化融合。
- IQA 回归模型采用轻量级 `MobileNetV3` 特征骨干，并加入局部混合、可选 WaveFormer 频域增强层与通道转置注意力模块。
- 评估阶段通过逐步排斥低质量样本，观察剩余集合上的 EER / TAR 变化，形成 ERC 曲线。
- 当前默认假设输入已经是 ROI 图像，因此不额外实现 ROI 检测与裁剪。
