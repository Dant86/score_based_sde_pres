# Score-Based SDE

Score-based generative models via stochastic differential equations (Song et al. 2021, [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)).

Trains three SDE variants — **VP**, **VE**, **Sub-VP** — on CIFAR-100 and evaluates
each with three samplers: Euler-Maruyama, Predictor-Corrector, and Probability Flow ODE.

---

## Setup

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync the environment (Python 3.14 + all deps)
uv sync --extra dev
```

---

## Local training (smoke test / CPU)

Good for verifying the pipeline works before committing GPU time.

```bash
# Train VP-SDE for 5 epochs, small batch
uv run scripts/train.py --sde vp --epochs 5 --batch-size 32

# Resume from a checkpoint
uv run scripts/train.py --sde vp --resume-from checkpoints/vp/epoch_0004.pt
```

Available flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--sde` | `vp` | SDE type: `vp`, `ve`, `subvp` |
| `--epochs` | `100` | Total training epochs |
| `--batch-size` | `128` | Batch size |
| `--checkpoint-dir` | `./checkpoints` | Root dir; checkpoints land in `<dir>/<sde>/` |
| `--resume-from` | — | Path to `.pt` checkpoint to resume from |
| `--data-dir` | `./data` | CIFAR-100 download/cache directory |

---

## Train on Modal (GPU, all three models in parallel)

### 1 — Install Modal and authenticate

```bash
pip install modal
modal setup        # opens browser for one-time auth
```

### 2 — Launch training

```bash
modal run scripts/train_modal.py::main
```

This fans out **three parallel H100 containers** — one per SDE type.
Each run trains for 100 epochs on CIFAR-100 (≈ 6 hours per container).

Checkpoints are written into the Modal Volume `score-sde-ckpts`:
```
/checkpoints/vp/epoch_0099.pt
/checkpoints/ve/epoch_0099.pt
/checkpoints/subvp/epoch_0099.pt
```
Intermediate checkpoints are saved every 5 epochs as well.

> **GPU quota**: defaults to H100. If you hit quota limits, edit `GPU_TYPE = "A100"`
> in `scripts/train_modal.py` and re-run.

### 3 — Download weights to your laptop

```bash
modal run scripts/train_modal.py::download --local-dir ./checkpoints
```

This streams every `.pt` file from the volume to `./checkpoints/`, preserving
the directory structure:
```
./checkpoints/vp/epoch_0099.pt
./checkpoints/ve/epoch_0099.pt
./checkpoints/subvp/epoch_0099.pt
```

---

## Evaluate locally (frozen checkpoints)

All evaluation runs on your laptop against the downloaded weights.
No GPU required (though it helps for speed).

### FID with Euler-Maruyama (default)

```bash
uv run scripts/evaluate.py --sde vp --ckpt checkpoints/vp/epoch_0099.pt
uv run scripts/evaluate.py --sde ve --ckpt checkpoints/ve/epoch_0099.pt
uv run scripts/evaluate.py --sde subvp --ckpt checkpoints/subvp/epoch_0099.pt
```

### Swap the sampler

```bash
# Predictor-Corrector
uv run scripts/evaluate.py \
  --sde vp --ckpt checkpoints/vp/epoch_0099.pt \
  --sampler predictor_corrector

# Probability Flow ODE (deterministic)
uv run scripts/evaluate.py \
  --sde vp --ckpt checkpoints/vp/epoch_0099.pt \
  --sampler ode
```

### Classifier guidance (conditional generation)

Train the noisy classifier first (see `src/score_sde/guidance/classifier.py`), then:

```bash
uv run scripts/evaluate.py \
  --sde vp \
  --ckpt checkpoints/vp/epoch_0099.pt \
  --use-guidance \
  --guidance-scale 2.0 \
  --target-class 5 \
  --classifier-path checkpoints/classifier.pt
```

Available eval flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--sde` | required | `vp`, `ve`, `subvp` |
| `--ckpt` | required | Path to score network checkpoint |
| `--sampler` | `euler_maruyama` | `euler_maruyama`, `predictor_corrector`, `ode` |
| `--n-samples` | `5000` | Images to generate for FID |
| `--n-steps` | `1000` | Discretisation steps |
| `--batch-size` | `64` | Generation batch size |
| `--output-dir` | `./eval_output` | Where to save generated `.pt` tensors |

---

## Run tests

```bash
uv run pytest
```

Tests cover SDE shape contracts, sampler output shapes + clipping,
and ScoreNet `.sample` extraction / attention-free architecture.

---

## Project structure

```
src/score_sde/
├── config.py                   # TrainConfig, EvalConfig, SDEConfig, ModelConfig
├── sdes/
│   ├── base.py                 # SDE ABC + ScoreFn Protocol
│   ├── vp.py                   # VP-SDE
│   ├── ve.py                   # VE-SDE
│   └── subvp.py                # Sub-VP SDE
├── samplers/
│   ├── base.py                 # Sampler ABC
│   ├── euler_maruyama.py       # First-order stochastic
│   ├── predictor_corrector.py  # EM predictor + Langevin corrector
│   └── ode.py                  # Probability flow ODE (Heun / RK2)
├── models/
│   └── score_net.py            # UNet2DModel wrapper; noise_pred + score_fn
├── guidance/
│   └── classifier.py           # NoisyClassifier + guided score fn factory
├── data/
│   └── cifar100.py             # CIFAR-100 DataLoader factory
├── training/
│   ├── losses.py               # Denoising score matching loss
│   ├── param_groups.py         # Bias/norm → no-decay param split
│   └── trainer.py              # Training loop, grad clip, checkpointing
└── evaluation/
    ├── sampler_runner.py       # Batched sample generation
    └── fid.py                  # FID via torchmetrics

scripts/
├── train.py                    # Local training CLI
├── evaluate.py                 # Local evaluation CLI
└── train_modal.py              # Modal app: train + download entrypoints
```

---

## Key implementation notes

- **score = −ε_θ(x,t) / σ(t)** — noise prediction is converted to score inside `ScoreNet.as_score_fn()`.
- **No attention blocks** — `UNet2DModel` uses `DownBlock2D`/`UpBlock2D` throughout; confirmed by `test_no_attention_blocks`.
- **Timestep scaling** — continuous `t ∈ [0, 1]` is mapped to `int(t × 999)` before passing to the UNet's positional embedding.
- **Sub-VP discount** — diffusion coefficient uses `1 − exp(4 · log_mean_coeff)` (not `2 ·`); see `subvp.py` for derivation.
- **Modal volume** — checkpoints persist in `score-sde-ckpts`; the volume survives container shutdown and can be re-mounted in future runs for resuming.
