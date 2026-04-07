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

----

## 演示前端

前端位于 `app/`，采用 `Vite + React + Tailwind CSS + shadcn + Bun`，并增加了 `react-router-dom` 路由结构与 `motion` 动画。

模型推理仍由 `uv run pv-iqa ...` 提供，Bun 仅做上传桥接、任务调度与 SQLite 持久化。

### 开发启动

```bash
cd app
bun install
bun run dev
```

- 前端开发页：`http://localhost:6006`
- Bun API：`http://localhost:6007`

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
- 页面拆分为：
  - `/workspace`：评分工作台
  - `/jobs`：任务管理与结果查看

### 运行与存储说明

- 默认会自动选择 `checkpoints/` 下最近且包含 `iqa/best.pt` 的 run
- 可通过环境变量 `PV_IQA_RUN_NAME` 指定默认推理 run
- SQLite 数据库存放在 `app/data/demo.sqlite`
- 上传文件存放在 `app/data/uploads/<jobId>/`
- 删除历史任务时，会同时删除数据库记录和该任务对应的上传文件


## 说明

- 伪标签生成使用识别特征和分类器权重，分别构建 `QSDD` 与 `QCR`，再做归一化融合。
- IQA 回归模型采用轻量级 `MobileNetV3` 特征骨干，并加入局部混合与通道转置注意力模块。
- 评估阶段通过逐步排斥低质量样本，观察剩余集合上的 EER / TAR 变化，形成 ERC 曲线。
- 当前默认假设输入已经是 ROI 图像，因此不额外实现 ROI 检测与裁剪。
