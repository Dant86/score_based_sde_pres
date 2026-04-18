"""Modal evaluation app for score-based SDE models.

Run order
---------
1. Train score models (first time only):
       modal run scripts/train_modal.py

2. Train the noisy classifier (required for class-conditional eval only):
       modal run scripts/eval_modal.py::train_classifier

3. Run all evaluations and generate Plotly figures locally:
       modal run scripts/eval_modal.py::evaluate_all

Individual entrypoints are available if you want to re-run a single eval:
       modal run scripts/eval_modal.py::fid_only
       modal run scripts/eval_modal.py::sampler_comparison
       modal run scripts/eval_modal.py::class_conditional
"""

import glob
import pathlib

import modal

# ---------------------------------------------------------------------------
# Infrastructure (reuses the same volume as training)
# ---------------------------------------------------------------------------

APP_NAME = "score-sde-eval"
VOLUME_NAME = "score-sde-ckpts"
CHECKPOINT_ROOT = "/checkpoints"
GPU_TYPE = "H100"
TIMEOUT_SECONDS = 3 * 60 * 60  # 3 h

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

_repo_root = pathlib.Path(__file__).parent.parent

image = (
    modal.Image.debian_slim(python_version="3.14")
    .pip_install_from_pyproject(str(_repo_root / "pyproject.toml"))
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir(str(_repo_root / "src"), "/root/src")
)

# ---------------------------------------------------------------------------
# Helpers (run inside remote functions)
# ---------------------------------------------------------------------------

_SDE_LABELS = {"vp": "VP-SDE", "ve": "VE-SDE", "subvp": "Sub-VP SDE"}
_SAMPLER_LABELS = {
    "euler_maruyama": "Euler-Maruyama",
    "predictor_corrector": "Predictor-Corrector",
    "ode": "Probability Flow ODE",
}
_DEFAULT_CLASSES = [19, 51, 70]  # cattle, mushroom, rose — visually distinct


def _latest_ckpt(sde_type: str) -> str:
    """Return the path to the most recently saved score-model checkpoint.

    Args:
        sde_type (str): one of ``"vp"``, ``"ve"``, or ``"subvp"``.

    Returns:
        str: absolute path to the latest ``.pt`` file for that SDE type.

    Raises:
        FileNotFoundError: if no checkpoint exists for ``sde_type``.
    """
    pattern = f"{CHECKPOINT_ROOT}/{sde_type}/epoch_*.pt"
    ckpts = sorted(glob.glob(pattern))
    if not ckpts:
        raise FileNotFoundError(
            f"No checkpoints found at {pattern}. "
            "Run `modal run scripts/train_modal.py` first."
        )
    return ckpts[-1]


def _load_score_model(sde_type: str, device):
    """Load a ScoreNet from the latest checkpoint for the given SDE type.

    Args:
        sde_type (str): one of ``"vp"``, ``"ve"``, or ``"subvp"``.
        device (torch.device): device to map weights onto.

    Returns:
        ScoreNet: model in eval mode on ``device``.
    """
    import torch

    from score_sde.config import ModelConfig
    from score_sde.models.score_net import ScoreNet

    cfg = ModelConfig()
    model = ScoreNet(
        sample_size=cfg.sample_size,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        block_out_channels=cfg.block_out_channels,
        layers_per_block=cfg.layers_per_block,
    )
    ckpt = torch.load(_latest_ckpt(sde_type), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    return model.eval().to(device)


def _build_sde(sde_type: str):
    """Instantiate the SDE for the given type string.

    Args:
        sde_type (str): one of ``"vp"``, ``"ve"``, or ``"subvp"``.

    Returns:
        SDE: configured SDE instance.
    """
    from score_sde.sdes.subvp import SubVPSDE
    from score_sde.sdes.ve import VESDE
    from score_sde.sdes.vp import VPSDE

    match sde_type:
        case "vp":
            return VPSDE()
        case "ve":
            return VESDE()
        case "subvp":
            return SubVPSDE()
        case _:
            raise ValueError(f"Unknown SDE type: {sde_type!r}")


def _build_sampler(sampler_type: str):
    """Instantiate a sampler by name.

    Args:
        sampler_type (str): one of ``"euler_maruyama"``,
            ``"predictor_corrector"``, or ``"ode"``.

    Returns:
        Sampler: configured sampler instance.
    """
    from score_sde.samplers.euler_maruyama import EulerMaruyama
    from score_sde.samplers.ode import ProbabilityFlowODE
    from score_sde.samplers.predictor_corrector import PredictorCorrector

    match sampler_type:
        case "euler_maruyama":
            return EulerMaruyama()
        case "predictor_corrector":
            return PredictorCorrector()
        case "ode":
            return ProbabilityFlowODE()
        case _:
            raise ValueError(f"Unknown sampler: {sampler_type!r}")


# ---------------------------------------------------------------------------
# Remote functions
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=TIMEOUT_SECONDS,
    volumes={CHECKPOINT_ROOT: volume},
)
def generate_sde_grid(n_cols: int = 8, n_steps: int = 1000) -> dict[str, list]:
    """Generate ``n_cols`` samples for each of VP-SDE, VE-SDE, and Sub-VP SDE.

    Uses the Euler-Maruyama sampler for all three models so that differences
    between rows reflect only the SDE, not the sampler.

    Args:
        n_cols (int): number of samples (images) per SDE row.
        n_steps (int): number of reverse diffusion steps.

    Returns:
        dict[str, list]: mapping from human-readable SDE label to a list of
            ``n_cols`` images, each a (H, W, C) uint8 numpy array.
    """
    import torch

    from score_sde.evaluation.sampler_runner import generate_samples
    from score_sde.evaluation.visualize import _to_uint8
    from score_sde.samplers.euler_maruyama import EulerMaruyama

    device = torch.device("cuda")
    sampler = EulerMaruyama()
    result: dict[str, list] = {}

    for sde_type, label in _SDE_LABELS.items():
        model = _load_score_model(sde_type, device)
        sde = _build_sde(sde_type)
        score_fn = model.as_score_fn(sde)
        samples = generate_samples(sde, score_fn, sampler, n_cols, n_cols, n_steps, device)
        result[label] = [_to_uint8(samples[i]).tolist() for i in range(n_cols)]
        print(f"  [{label}] done")

    return result


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=TIMEOUT_SECONDS,
    volumes={CHECKPOINT_ROOT: volume},
)
def compute_all_fid(n_samples: int = 5000, n_steps: int = 1000) -> dict[str, float]:
    """Compute FID for VP-SDE, VE-SDE, and Sub-VP SDE using Euler-Maruyama.

    Args:
        n_samples (int): number of generated images to use for FID computation.
        n_steps (int): number of reverse diffusion steps per sample.

    Returns:
        dict[str, float]: mapping from human-readable SDE label to FID score.
    """
    import torch

    from score_sde.data.cifar100 import get_cifar100_loaders
    from score_sde.evaluation.fid import compute_fid
    from score_sde.evaluation.sampler_runner import generate_samples
    from score_sde.samplers.euler_maruyama import EulerMaruyama

    device = torch.device("cuda")
    _, val_loader = get_cifar100_loaders(data_dir="/tmp/data", batch_size=64)
    sampler = EulerMaruyama()
    scores: dict[str, float] = {}

    for sde_type, label in _SDE_LABELS.items():
        print(f"  [{label}] generating {n_samples} samples …")
        model = _load_score_model(sde_type, device)
        sde = _build_sde(sde_type)
        score_fn = model.as_score_fn(sde)
        fake = generate_samples(sde, score_fn, sampler, n_samples, 64, n_steps, device)
        fid = compute_fid(val_loader, fake, device, n_real=n_samples)
        scores[label] = fid
        print(f"  [{label}] FID = {fid:.2f}")

    return scores


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=TIMEOUT_SECONDS,
    volumes={CHECKPOINT_ROOT: volume},
)
def generate_sampler_grid(n_cols: int = 8, n_steps: int = 1000) -> dict[str, list]:
    """Generate ``n_cols`` Sub-VP SDE samples for each of the three samplers.

    Args:
        n_cols (int): number of samples per sampler row.
        n_steps (int): number of reverse diffusion steps.

    Returns:
        dict[str, list]: mapping from sampler label to a list of ``n_cols``
            images as (H, W, C) uint8 numpy arrays.
    """
    import torch

    from score_sde.evaluation.sampler_runner import generate_samples
    from score_sde.evaluation.visualize import _to_uint8

    device = torch.device("cuda")
    model = _load_score_model("subvp", device)
    sde = _build_sde("subvp")
    result: dict[str, list] = {}

    for sampler_type, label in _SAMPLER_LABELS.items():
        sampler = _build_sampler(sampler_type)
        score_fn = model.as_score_fn(sde)
        samples = generate_samples(sde, score_fn, sampler, n_cols, n_cols, n_steps, device)
        result[label] = [_to_uint8(samples[i]).tolist() for i in range(n_cols)]
        print(f"  [{label}] done")

    return result


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=TIMEOUT_SECONDS,
    volumes={CHECKPOINT_ROOT: volume},
)
def generate_class_grid(
    classes: list[int] | None = None,
    n_per_class: int = 8,
    n_steps: int = 1000,
    guidance_scale: float = 3.0,
) -> dict[str, list]:
    """Generate class-conditional Sub-VP SDE samples via classifier guidance.

    Requires a trained NoisyClassifier at
    ``/checkpoints/classifier.pt``. Run ``train_classifier`` first.

    Args:
        classes (list[int] | None): CIFAR-100 class indices to generate.
            Defaults to ``[19, 51, 70]`` (cattle, mushroom, rose).
        n_per_class (int): number of samples per class row.
        n_steps (int): number of reverse diffusion steps.
        guidance_scale (float): γ in ``s̃ = s + γ · ∇_x log p(y | x_t)``.

    Returns:
        dict[str, list]: mapping from ``"class_idx: class_name"`` label to a
            list of ``n_per_class`` images as (H, W, C) uint8 numpy arrays.

    Raises:
        FileNotFoundError: if the classifier checkpoint is missing.
    """
    import torch

    from score_sde.evaluation.sampler_runner import generate_samples
    from score_sde.evaluation.visualize import CIFAR100_CLASSES, _to_uint8
    from score_sde.guidance.classifier import NoisyClassifier, make_guided_score_fn
    from score_sde.samplers.euler_maruyama import EulerMaruyama

    if classes is None:
        classes = _DEFAULT_CLASSES

    cls_path = f"{CHECKPOINT_ROOT}/classifier.pt"
    if not pathlib.Path(cls_path).exists():
        raise FileNotFoundError(
            f"Classifier checkpoint not found at {cls_path}. "
            "Run `modal run scripts/eval_modal.py::train_classifier` first."
        )

    device = torch.device("cuda")
    classifier = NoisyClassifier(num_classes=100).to(device)
    classifier.load_state_dict(torch.load(cls_path, map_location=device, weights_only=True))
    classifier.eval()

    model = _load_score_model("subvp", device)
    sde = _build_sde("subvp")
    sampler = EulerMaruyama()
    result: dict[str, list] = {}

    for cls_idx in classes:
        label = f"{cls_idx}: {CIFAR100_CLASSES.get(cls_idx, str(cls_idx))}"
        score_fn = model.as_score_fn(sde)
        guided_fn = make_guided_score_fn(score_fn, classifier, cls_idx, guidance_scale)
        samples = generate_samples(sde, guided_fn, sampler, n_per_class, n_per_class, n_steps, device)
        result[label] = [_to_uint8(samples[i]).tolist() for i in range(n_per_class)]
        print(f"  [{label}] done")

    return result


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=TIMEOUT_SECONDS,
    volumes={CHECKPOINT_ROOT: volume},
)
def train_classifier_fn(n_epochs: int = 50) -> str:
    """Train a NoisyClassifier on noisy CIFAR-100 using the Sub-VP SDE.

    The Sub-VP SDE is used for corruption because it produces a conservative
    noise schedule — the classifier sees a representative range of noise levels.
    Saves the classifier to the Modal volume at ``/checkpoints/classifier.pt``.

    Args:
        n_epochs (int): number of training epochs.

    Returns:
        str: path to the saved classifier checkpoint inside the volume.
    """
    import torch

    from score_sde.data.cifar100 import get_cifar100_loaders
    from score_sde.sdes.subvp import SubVPSDE
    from score_sde.training.classifier_trainer import train_noisy_classifier

    device = torch.device("cuda")
    sde = SubVPSDE()
    train_loader, _ = get_cifar100_loaders(data_dir="/tmp/data", batch_size=128)
    ckpt_path = f"{CHECKPOINT_ROOT}/classifier.pt"

    train_noisy_classifier(sde, train_loader, device, n_epochs=n_epochs, checkpoint_path=ckpt_path)
    volume.commit()
    return ckpt_path


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def train_classifier(n_epochs: int = 50) -> None:
    """Train and persist the NoisyClassifier. Run this before evaluate_all.

    Args:
        n_epochs (int): number of training epochs (default 50, ~30 min on H100).
    """
    print(f"Training NoisyClassifier for {n_epochs} epochs on Modal …")
    path = train_classifier_fn.remote(n_epochs=n_epochs)
    print(f"Classifier saved to volume: {path}")
    print("You can now run: modal run scripts/eval_modal.py::evaluate_all")


@app.local_entrypoint()
def evaluate_all(
    output_dir: str = "./eval_output",
    n_steps: int = 1000,
    n_fid_samples: int = 5000,
) -> None:
    """Run all four evaluations in parallel and write Plotly HTML figures locally.

    Generates:
    - ``sde_comparison.html``      — 3×8 sample grid (VP / VE / Sub-VP)
    - ``fid_comparison.html``      — FID bar chart for all three models
    - ``sampler_comparison.html``  — Sub-VP 3×8 grid across samplers
    - ``class_conditional.html``   — Sub-VP 3×8 class-conditional grid

    Args:
        output_dir (str): local directory to write HTML figures into.
        n_steps (int): reverse diffusion steps for all samplers.
        n_fid_samples (int): images used for FID computation.
    """
    import os

    import numpy as np
    import torch

    from score_sde.evaluation.visualize import plot_fid_bars, plot_sample_grid, save_figure

    os.makedirs(output_dir, exist_ok=True)
    print("Launching all eval jobs on Modal …")

    # Fan out — all four run in parallel on separate H100 containers
    sde_handle = generate_sde_grid.spawn(n_steps=n_steps)
    fid_handle = compute_all_fid.spawn(n_samples=n_fid_samples, n_steps=n_steps)
    sampler_handle = generate_sampler_grid.spawn(n_steps=n_steps)
    class_handle = generate_class_grid.spawn(n_steps=n_steps)

    def _deserialise(raw: dict[str, list]) -> dict[str, torch.Tensor]:
        """Reconstruct (N, C, H, W) float tensors from serialised uint8 lists."""
        out: dict[str, torch.Tensor] = {}
        for label, imgs in raw.items():
            arr = np.array(imgs, dtype=np.uint8)          # (N, H, W, C)
            t = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 127.5 - 1.0
            out[label] = t
        return out

    print("Waiting for results …")
    sde_samples = _deserialise(sde_handle.get())
    fid_scores: dict[str, float] = fid_handle.get()
    sampler_samples = _deserialise(sampler_handle.get())

    try:
        class_samples = _deserialise(class_handle.get())
        fig4 = plot_sample_grid(class_samples, title="Sub-VP SDE — Class-Conditional Samples (guidance γ=3)")
        save_figure(fig4, os.path.join(output_dir, "class_conditional.html"))
    except Exception as exc:
        print(f"  class_conditional skipped: {exc}")

    fig1 = plot_sample_grid(sde_samples, title="Generated Samples by SDE Type (Euler-Maruyama)")
    save_figure(fig1, os.path.join(output_dir, "sde_comparison.html"))

    fig2 = plot_fid_bars(fid_scores)
    save_figure(fig2, os.path.join(output_dir, "fid_comparison.html"))

    fig3 = plot_sample_grid(sampler_samples, title="Sub-VP SDE — Sampler Comparison")
    save_figure(fig3, os.path.join(output_dir, "sampler_comparison.html"))

    print(f"\nAll figures written to {output_dir}/")
    print("FID scores:")
    for label, fid in fid_scores.items():
        print(f"  {label}: {fid:.2f}")


@app.local_entrypoint()
def fid_only(n_samples: int = 5000, n_steps: int = 1000, output_dir: str = "./eval_output") -> None:
    """Compute and display FID for all three models, without generating figures.

    Args:
        n_samples (int): images to generate for FID.
        n_steps (int): reverse diffusion steps.
        output_dir (str): directory to write ``fid_comparison.html`` into.
    """
    import os

    from score_sde.evaluation.visualize import plot_fid_bars, save_figure

    os.makedirs(output_dir, exist_ok=True)
    scores = compute_all_fid.remote(n_samples=n_samples, n_steps=n_steps)
    print("\nFID scores:")
    for label, fid in scores.items():
        print(f"  {label}: {fid:.2f}")
    fig = plot_fid_bars(scores)
    save_figure(fig, os.path.join(output_dir, "fid_comparison.html"))


@app.local_entrypoint()
def sampler_comparison(n_steps: int = 1000, output_dir: str = "./eval_output") -> None:
    """Generate the Sub-VP sampler comparison grid only.

    Args:
        n_steps (int): reverse diffusion steps.
        output_dir (str): directory to write ``sampler_comparison.html`` into.
    """
    import os

    import numpy as np
    import torch

    from score_sde.evaluation.visualize import plot_sample_grid, save_figure

    os.makedirs(output_dir, exist_ok=True)
    raw = generate_sampler_grid.remote(n_steps=n_steps)

    samples: dict[str, torch.Tensor] = {}
    for label, imgs in raw.items():
        arr = np.array(imgs, dtype=np.uint8)
        samples[label] = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 127.5 - 1.0

    fig = plot_sample_grid(samples, title="Sub-VP SDE — Sampler Comparison")
    save_figure(fig, os.path.join(output_dir, "sampler_comparison.html"))


@app.local_entrypoint()
def class_conditional(
    n_steps: int = 1000,
    guidance_scale: float = 3.0,
    output_dir: str = "./eval_output",
) -> None:
    """Generate the class-conditional sample grid for Sub-VP SDE.

    Requires the classifier to be trained first via ``train_classifier``.

    Args:
        n_steps (int): reverse diffusion steps.
        guidance_scale (float): classifier guidance strength γ.
        output_dir (str): directory to write ``class_conditional.html`` into.
    """
    import os

    import numpy as np
    import torch

    from score_sde.evaluation.visualize import plot_sample_grid, save_figure

    os.makedirs(output_dir, exist_ok=True)
    raw = generate_class_grid.remote(n_steps=n_steps, guidance_scale=guidance_scale)

    samples: dict[str, torch.Tensor] = {}
    for label, imgs in raw.items():
        arr = np.array(imgs, dtype=np.uint8)
        samples[label] = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 127.5 - 1.0

    fig = plot_sample_grid(samples, title=f"Sub-VP SDE — Class-Conditional Samples (γ={guidance_scale})")
    save_figure(fig, os.path.join(output_dir, "class_conditional.html"))
