from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


# fmt: off
@dataclass
class Config:
    # -- 通用 --------------------------------------------------------------------
    name: str = "auto"                      # 实验名称，auto=自动时间戳
    output_root: str = "checkpoints"        # 产物输出根目录
    seed: int = 2026                        # 随机种子
    device: str = "auto"                    # 训练设备 (auto/cpu/cuda)
    amp: bool = True                        # 自动混合精度（仅 CUDA 生效）
    num_workers: int = 4                    # DataLoader 工作线程数
    wandb_enabled: bool = False             # 是否启用 WandB 日志

    # -- 数据 --------------------------------------------------------------------
    data_root: str = "datasets/HFUT-PV-ROI" # 数据集根目录
    metadata_path: str = "auto"             # 元数据 CSV 路径，auto=实验目录下生成
    identity_mode: str = "separate"         # 身份解析模式：separate=左右手分开, merge_person=合并
    image_size: int = 224                   # 输入图像尺寸
    batch_size: int = 32                    # 训练批次大小

    # 数据集分割，用于训练识别器和 IQA 模型，以及测试
    class_recognition_ratio: float = 0.25   # 识别数据集类占比 (0=复用已有识别器)
    class_iqa_ratio: float = 0.50           # IQA 数据集类占比
    # 测试数据集类占比 = 1 − recognition_ratio − iqa_ratio

    grayscale_to_rgb: bool = True           # 单通道灰度转三通道 RGB

    # -- 识别器 (ArcFace) ---------------------------------------------------------
    recog_backbone: str = "mobilenetv3_large_100"
    recog_pretrained: bool = True           # 是否加载 ImageNet 预训练权重
    recog_embedding_dim: int = 256          # Embedding 输出维度
    recog_dropout: float = 0.1              # Dropout 比例
    recog_margin: float = 0.3               # ArcFace 角度边际 m (Deng et al., CVPR 2019)
    recog_scale: float = 30.0               # ArcFace 特征缩放 s
    recog_epochs: int = 10                  # 识别器训练轮数
    recog_lr: float = 3e-4                  # 识别器学习率
    recog_wd: float = 1e-4                  # 识别器权重衰减
    recog_warmup_epochs: int = 1            # 识别器学习率预热轮数

    # -- 伪标签生成 --------------------------------------------------------------
    # 公式: Q = 100 × minmax( δ·minmax(Q^P) + β·minmax(WD) + γ·minmax(Q^V) )
    #   Q^P: 类内余弦相似度期望 (PGRG)
    #   WD:  Wasserstein 分布距离 (SDD-FIQA)
    #   Q^V: 视觉质量 = (清晰度 + 对比度 + 曝光平衡) / 3
    pseudo_split: str = "all"               # 伪标签计算所用 split
    pseudo_delta: float = 1.0               # Q^P 权重
    pseudo_beta: float = 0.1                # WD 权重
    pseudo_gamma: float = 1.0               # Q^V 权重
    pseudo_qv_weights: str = "1,1,1"        # Q^V 子指标权重 (清晰度,对比度,曝光)
    pseudo_per_component_norm: bool = True  # 每分量独立 minmax 后再融合
    pseudo_degrade_penalty: float = 0.5     # 退化惩罚系数 (0.5=减半)

    # -- IQA 回归模型 -------------------------------------------------------------
    iqa_backbone: str = "mobilenetv3_large_100"
    iqa_pretrained: bool = True             # 是否加载 ImageNet 预训练权重
    iqa_epochs: int = 20                    # IQA 训练轮数
    iqa_lr: float = 2e-4                    # IQA 学习率
    iqa_wd: float = 5e-5                    # IQA 权重衰减
    iqa_warmup_epochs: int = 2              # IQA 学习率预热轮数
    iqa_grad_clip: float = 1.0              # 梯度裁剪阈值
    iqa_huber_delta: float = 0.1            # Huber Loss 的 delta 参数
    iqa_rank_weight: float = 0.3            # PGRG 排序损失权重
    iqa_min_rank_gap: float = 0.05          # 排序对的最小伪标签差距

    # -- 评估 --------------------------------------------------------------------
    eval_split: str = "test"                # 评估所用 split
    eval_batch_size: int = 64               # 评估批次大小
    eval_reject_steps: list[float] = field(
        default_factory=lambda: [0.0, 0.05, 0.1, 0.2, 0.3],
    )
    eval_far_targets: list[float] = field(default_factory=lambda: [1e-2, 1e-3, 1e-4])
    eval_max_impostor_pairs: int = 20000

    @property
    def experiment_dir(self) -> Path:
        return Path(self.output_root) / self.name

    def resolve(self) -> "Config":
        if self.name in ("", "auto"):
            self.name = f"{datetime.now():%Y%m%d-%H%M%S}"
        if self.metadata_path in ("", "auto"):
            self.metadata_path = str(self.experiment_dir / "data" / "metadata.csv")
        return self
# fmt: on
