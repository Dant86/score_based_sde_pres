"""Modal entrypoint: train VP, VE, and Sub-VP score models on H100 GPUs in parallel.

Usage
-----
Train all three models (fans out to three parallel H100 containers):
    modal run scripts/train_modal.py

Download finished checkpoints to your laptop:
    modal run scripts/train_modal.py::download --local-dir ./checkpoints
"""

import pathlib

import modal

# ---------------------------------------------------------------------------
# App & infrastructure
# ---------------------------------------------------------------------------

APP_NAME = "score-sde"
VOLUME_NAME = "score-sde-ckpts"
CHECKPOINT_ROOT = "/checkpoints"
GPU_TYPE = "H100"
TIMEOUT_SECONDS = 6 * 60 * 60  # 6 h per model

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

_repo_root = pathlib.Path(__file__).parent.parent

image = (
    modal.Image.debian_slim(python_version="3.14")
    # Install all PyPI dependencies declared in pyproject.toml
    .pip_install_from_pyproject(str(_repo_root / "pyproject.toml"))
    # Copy the local package source into the image
    .copy_local_dir(str(_repo_root / "src"), "/root/src")
    # Make the package importable without an editable install
    .env({"PYTHONPATH": "/root/src"})
)


# ---------------------------------------------------------------------------
# Training function (runs remotely)
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=TIMEOUT_SECONDS,
    volumes={CHECKPOINT_ROOT: volume},
)
def train_one(sde_type: str) -> str:
    """Train a single score-based SDE model and return the final checkpoint path."""
    import torch

    from score_sde.config import ModelConfig, SDEConfig, TrainConfig
    from score_sde.data.cifar100 import get_cifar100_loaders
    from score_sde.models.score_net import ScoreNet
    from score_sde.sdes.subvp import SubVPSDE
    from score_sde.sdes.ve import VESDE
    from score_sde.sdes.vp import VPSDE
    from score_sde.training.trainer import Trainer

    device = torch.device("cuda")
    print(f"[{sde_type}] Starting on {torch.cuda.get_device_name(0)}")

    sde_cfg = SDEConfig(sde_type=sde_type)
    model_cfg = ModelConfig()
    train_cfg = TrainConfig(
        sde=sde_cfg,
        model=model_cfg,
        checkpoint_dir=f"{CHECKPOINT_ROOT}/{sde_type}",
        data_dir="/tmp/data",  # ephemeral container-local storage for dataset
    )

    sde_map = {
        "vp": VPSDE(beta_min=sde_cfg.beta_min, beta_max=sde_cfg.beta_max, T=sde_cfg.T),
        "ve": VESDE(sigma_min=sde_cfg.sigma_min, sigma_max=sde_cfg.sigma_max, T=sde_cfg.T),
        "subvp": SubVPSDE(beta_min=sde_cfg.beta_min, beta_max=sde_cfg.beta_max, T=sde_cfg.T),
    }
    sde = sde_map[sde_type]

    model = ScoreNet(
        sample_size=model_cfg.sample_size,
        in_channels=model_cfg.in_channels,
        out_channels=model_cfg.out_channels,
        block_out_channels=model_cfg.block_out_channels,
        layers_per_block=model_cfg.layers_per_block,
    )

    train_loader, _ = get_cifar100_loaders(
        data_dir="/tmp/data",
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
    )

    trainer = Trainer(train_cfg, model, sde, train_loader, device)
    trainer.train()

    # Flush writes to the persistent volume
    volume.commit()

    final_ckpt = f"{CHECKPOINT_ROOT}/{sde_type}/epoch_{train_cfg.epochs - 1:04d}.pt"
    print(f"[{sde_type}] Done → {final_ckpt}")
    return final_ckpt


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main() -> None:
    """Fan out training of VP, VE, and Sub-VP in parallel on three H100 containers."""
    sde_types = ["vp", "ve", "subvp"]
    print(f"Launching parallel training for: {sde_types}")
    results = list(train_one.map(sde_types))
    print("\nAll training runs complete. Checkpoints written to Modal Volume:")
    for path in results:
        print(f"  {path}")
    print(f"\nTo download weights:\n  modal run scripts/train_modal.py::download")


@app.local_entrypoint()
def download(local_dir: str = "./checkpoints") -> None:
    """Stream all .pt checkpoints from the Modal Volume to `local_dir`."""
    import os

    downloaded = 0
    for entry in volume.listdir("/", recursive=True):
        if not entry.path.endswith(".pt"):
            continue
        dest = os.path.join(local_dir, entry.path.lstrip("/"))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print(f"  /{entry.path}  →  {dest}")
        with open(dest, "wb") as f:
            for chunk in volume.read_file(entry.path):
                f.write(chunk)
        downloaded += 1

    if downloaded == 0:
        print("No .pt files found in the volume. Has training finished?")
    else:
        print(f"\nDone — {downloaded} checkpoint(s) saved to {local_dir}/")
