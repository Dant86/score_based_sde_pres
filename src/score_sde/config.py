from dataclasses import dataclass, field


@dataclass
class SDEConfig:
    sde_type: str = "vp"  # "vp" | "ve" | "subvp"
    # VP / Sub-VP
    beta_min: float = 0.1
    beta_max: float = 20.0
    # VE
    sigma_min: float = 0.01
    sigma_max: float = 50.0
    T: float = 1.0


@dataclass
class ModelConfig:
    sample_size: int = 32
    in_channels: int = 3
    out_channels: int = 3
    block_out_channels: tuple[int, ...] = (128, 256, 256, 256)
    layers_per_block: int = 2


@dataclass
class TrainConfig:
    sde: SDEConfig = field(default_factory=SDEConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    # Optimizer
    lr: float = 1e-4
    eps: float = 1e-7
    weight_decay: float = 1e-4
    grad_clip: float = 0.5
    # Training loop
    epochs: int = 100
    batch_size: int = 128
    num_workers: int = 4
    checkpoint_every: int = 5
    checkpoint_dir: str = "checkpoints"
    resume_from: str | None = None
    # Data
    data_dir: str = "./data"


@dataclass
class EvalConfig:
    sde: SDEConfig = field(default_factory=SDEConfig)
    checkpoint_path: str = ""
    sampler: str = "euler_maruyama"  # "euler_maruyama" | "predictor_corrector" | "ode"
    n_samples: int = 5000
    n_steps: int = 1000
    batch_size: int = 64
    data_dir: str = "./data"
    output_dir: str = "./eval_output"
    # Classifier guidance
    use_guidance: bool = False
    guidance_scale: float = 1.0
    target_class: int = 0
    classifier_path: str | None = None
