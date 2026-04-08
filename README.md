# PV-IQA

基于深度学习的掌静脉图像质量评估（Palm Vein Image Quality Assessment）项目，覆盖**训练、评估、导出、前端演示**与 **Rust/Candle 推理服务集成**。

当前实现面向**已经完成 ROI 裁剪**的掌静脉图像，重点解决以下问题：

1. 掌静脉识别基线训练与特征导出
2. 基于 SDD + CR 的无监督质量伪标签生成
3. 轻量级 IQA 回归模型训练
4. 基于 ERC / EER / TAR@FAR 的生物识别效用评估
5. Python 与 Rust/Candle 双后端推理

---

## 1. 核心能力

- **完整训练链路**：从元数据生成到识别训练、伪标签生成、IQA 训练、ERC 评估
- **可选 WaveFormer 频域增强层**：在轻量级 IQA 网络中插入频域建模层
- **Rust/Candle 推理后端**：支持 CPU / CUDA 两种发布二进制
- **前端演示系统**：支持单图、文件夹、任务历史、后端切换、结果预览
- **Python / Rust 推理一致性**：Rust 默认复用 Python/Pillow 预处理链，保证与 Python 推理结果对齐

---

## 2. 仓库结构

| 路径 | 说明 |
| --- | --- |
| `configs/` | 项目配置 |
| `src/pv_iqa/` | Python 主代码 |
| `datasets/` | 数据集与 ROI 图像 |
| `checkpoints/` | 训练输出、导出模型、评估结果 |
| `app/` | Bun + Vite + React 演示前端 |
| `bin/` | 运行脚本与已发布 Rust 二进制 |
| `/root/autodl-tmp/PV-IQA-rs` | Rust/Candle 推理服务项目 |

---

## 3. 环境要求

### 必需

- `uv`
- `Python 3.14`
- `PyTorch 2.10+`

### 可选

- `Bun`：前端与 Bun API
- `Rust stable`：Rust 推理服务构建
- `CUDA / nvcc`：构建与运行 CUDA 版 `pv-iqa-rs`

---

## 4. 初始化

### Python 环境

```bash
uv sync
```

### 前端环境

```bash
cd app
bun install
```

---

## 5. 训练与评估流程

### 5.1 生成元数据

```bash
uv run pv-iqa prepare-data
```

### 5.2 训练识别基线并导出特征

```bash
uv run pv-iqa train-recognizer
```

### 5.3 生成无监督质量伪标签

```bash
uv run pv-iqa generate-pseudo-labels
```

### 5.4 训练 IQA 回归模型

```bash
uv run pv-iqa train-iqa
```

### 5.5 ERC 评估

```bash
uv run pv-iqa evaluate
```

### 5.6 一键全流程

```bash
uv run pv-iqa run-all
```

---

## 6. 推理命令

### 单图评分

```bash
uv run pv-iqa detect-image datasets/ROI_Data/ac_l/1.jpg --run-name <run-name>
```

### 文件夹评分

```bash
uv run pv-iqa detect-folder datasets/ROI_Data/ac_l --run-name <run-name>
```

### 导出 Rust 可用模型

```bash
uv run pv-iqa export-rust-model --run-name <run-name>
```

导出产物默认位于：

```text
checkpoints/<run-name>/iqa/best.pt
checkpoints/<run-name>/iqa/best.onnx
checkpoints/<run-name>/iqa/best.onnx.json
```

说明：

- `best.onnx` 为**单文件** ONNX，Rust 侧可直接加载
- 若导出过程中出现 `best.onnx.data`，工具会自动内联并删除 sidecar
- 导出逻辑位于 `src/pv_iqa/utils/export_onnx.py`

---

## 7. 关键配置

### 数据与身份

- `data.identity_mode`
  - `separate`：左右手分开建类
  - `merge_person`：左右手合并为同一身份

### 日志

- `logger.wandb_mode`
  - `offline`：默认，适合本地实验
  - `disabled`：关闭 wandb
  - `online`：在线同步

### IQA 网络

- `iqa.use_waveformer_layer`
  - `false`：关闭 WaveFormer 层
  - `true`：在高层特征分支插入 WaveFormer 风格频域增强层

---

## 8. 前端演示系统

前端位于 `app/`，采用：

- `Bun`
- `Vite`
- `React`
- `Tailwind CSS`
- `shadcn/ui`
- `react-router-dom`
- `motion`

当前支持两条推理路径：

- **Python**：沿用 `uv run pv-iqa ...`
- **Rust Candle**：由 Bun 调用 `/root/autodl-tmp/PV-IQA-rs` 中的常驻服务

### 开发启动

```bash
cd app
bun run dev
```

- 前端开发页：`http://localhost:6006`
- Bun API：默认 `http://localhost:6005`

如需改 API 端口，可在启动前设置：

- `PV_IQA_API_PORT`
- 或 `PORT`

### 生产启动

```bash
cd app
bun run build
bun run start
```

生产模式下由 Bun 在 `6006` 同时提供前端静态资源与 API。

### 前端能力

- 单图与文件夹评分
- 拖拽上传
- 异步任务状态与处理进度
- 历史任务查看与彻底删除
- Python / Rust 后端切换

主要页面：

- `/workspace`：上传、评分、结果预览
- `/jobs`：任务历史与结果查看

### 运行与存储约定

- 默认选择 `checkpoints/` 下最近且含 `iqa/best.pt` 的 run
- 可通过 `PV_IQA_RUN_NAME` 指定默认推理 run
- SQLite 数据库：`app/data/demo.sqlite`
- 上传目录：`app/data/uploads/<jobId>/`

---

## 9. Rust / Candle 推理后端集成

Rust 项目位于：

```text
/root/autodl-tmp/PV-IQA-rs
```

提供接口：

- `GET /health`
- `POST /score/image`
- `POST /score/batch`

### 9.1 手动启动 Rust 服务

```bash
cd /root/autodl-tmp/PV-IQA-rs
PV_IQA_REPO_ROOT=/root/workspace/PV-IQA cargo run --release
```

Rust 服务默认监听 `127.0.0.1:7007`。

### 9.2 Rust 服务环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PV_IQA_REPO_ROOT` | `/root/workspace/PV-IQA` | PV-IQA 仓库根目录 |
| `PV_IQA_RS_HOST` | `127.0.0.1` | Rust 服务监听地址 |
| `PV_IQA_RS_PORT` | `7007` | Rust 服务监听端口 |
| `PV_IQA_RS_DEVICE` | `auto` | 运行设备：`auto` / `cpu` / `cuda` |
| `PV_IQA_RS_CUDA_ORDINAL` | `0` | CUDA 设备编号 |
| `PV_IQA_RS_PREPROCESS_MODE` | `python-pillow` | 预处理模式；默认复用 Python/Pillow 预处理以保持推理一致性 |
| `PV_IQA_RS_RESIZE_MODE` | `fast-convolution-bilinear` | 纯 Rust 预处理模式下的 resize 策略（调试用） |
| `PV_IQA_RS_DECODE_MODE` | `image` | 纯 Rust 预处理模式下的 JPEG 解码策略（调试用） |

### 9.3 Bun 桥接层环境变量

`app/server.ts` 支持以下 Rust 相关配置：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PV_IQA_RS_URL` | `http://127.0.0.1:7007` | Bun 访问 Rust 服务的地址 |
| `PV_IQA_RS_PROJECT_ROOT` | `/root/autodl-tmp/PV-IQA-rs` | Rust 项目目录；仅在缺少可执行文件时用于构建/发布 |
| `PV_IQA_RS_BINARY` | `""` | 指定要直接启动的 Rust 可执行文件 |
| `PV_IQA_RS_AUTOSTART` | `1` | Rust 不可用时是否自动拉起二进制 |
| `PV_IQA_RS_RELEASE` | `1` | 缺少二进制时，是否默认按 `release` 构建 |
| `PV_IQA_RS_CARGO_FEATURES` | `cuda` | 构建 CUDA 版二进制时使用的 Cargo features |
| `PV_IQA_RS_PREPROCESS_MODE` | `python-pillow` | Bun 自动拉起 Rust 时使用的预处理模式 |
| `PV_IQA_RS_HEALTH_TIMEOUT_MS` | `1500` | 健康检查超时 |
| `PV_IQA_RS_REQUEST_TIMEOUT_MS` | `300000` | Rust 推理请求超时 |
| `PV_IQA_RS_START_TIMEOUT_MS` | `300000` | 自动拉起服务后的等待时长 |

### 9.4 已发布二进制与选择规则

`bin/` 中维护两类已发布 Rust 二进制：

- `bin/pv-iqa-rs-<profile>-cpu-<timestamp>`
- `bin/pv-iqa-rs-<profile>-cuda-<timestamp>`

规则：

1. 若 `best.onnx` / `best.onnx.json` 缺失，Bun 会先自动导出
2. 若缺少已发布二进制，Bun 会补齐 CPU 与 CUDA 两个版本
3. 默认优先选择 `cuda` 版
4. 若显式设置 `PV_IQA_RS_DEVICE=cpu`，则强制选择 `cpu` 版

### 9.5 官方双版本构建脚本

Rust 项目提供正式构建脚本：

```bash
cd /root/autodl-tmp/PV-IQA-rs
./scripts/build-binaries.sh --repo-root /root/workspace/PV-IQA
```

脚本会：

1. 分别构建 CPU 与 CUDA 版本
2. 按时间戳发布到 `PV-IQA/bin/`
3. 输出最终产物路径

---

## 10. 一键清理 API 进程

如果需要在重启前清空 Bun API 与 Rust API 进程，可执行：

```bash
./bin/stop-pv-iqa-api
```

或在 `app/` 下执行：

```bash
bun run stop:api
```

也可先用 dry-run 查看命中的进程：

```bash
./bin/stop-pv-iqa-api --dry-run
```

---

## 11. 结果输出

常见输出目录：

| 路径 | 内容 |
| --- | --- |
| `checkpoints/<run-name>/recognizer/` | 识别模型与特征导出 |
| `checkpoints/<run-name>/pseudo_labels/` | 伪标签 |
| `checkpoints/<run-name>/iqa/` | IQA 模型与 Rust ONNX 导出 |
| `checkpoints/<run-name>/evaluation/` | ERC 与评估结果 |
| `app/data/` | Demo 数据库与上传缓存 |

---

## 12. 说明

- 当前默认输入为**已经裁剪好的 ROI 图像**
- 暂不包含 ROI 提取阶段
- Rust 端默认通过 Python/Pillow helper 对齐预处理；模型前向仍由 Rust/Candle 执行
- 评估阶段通过逐步排斥低质量样本，观察剩余集合上的 EER / TAR 变化形成 ERC 曲线

