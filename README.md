# Yahtzee-MAML-with-PPO-core

Reinforcement learning for Yahtzee, written in JAX/Flax. The repo contains two training pipelines:

1. **FOMAML (meta-RL)**: First-Order MAML with a PPO (or A2C) inner loop. The meta-learner finds a policy initialization that can adapt to any reward-shaped Yahtzee task in a few gradient steps.
2. **Standalone A2C**: a reproduction of the single-agent A2C baseline from Pape (2025), [arXiv:2601.00007](https://arxiv.org/abs/2601.00007) ([reference code](https://github.com/papetronics/case-studies-final-project)).

## Quick start

This runs a short training on CPU, with no accounts or GPU needed.

```bash
# 1. Get the code
git clone https://github.com/Kilometrs/Yahtzee-MAML-with-PPO-core.git
cd Yahtzee-MAML-with-PPO-core

# 2. Install uv (skip if you have it), then the dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# 3. Run a ~1 minute smoke test (logs to TensorBoard instead of ClearML)
CLEARML_OFF=1 uv run python scripts/train_a2c.py --config configs/test-standalone-a2c.yaml

# 4. Look at the training curves at http://localhost:6006
uv run tensorboard --logdir runs/
```

If the smoke test ends with `Training complete.`, your setup works. For a real run, use `configs/standalone-a2c.yaml` (1M games, best on a GPU; see [Setup](#setup)) or the FOMAML configs described under [Training](#training). Checkpoints are saved in `checkpoints/`.

## How it works

- **Environment**: A full Yahtzee environment with all 13 scoring categories, rerolls, upper-section bonus logic, and valid-action masking. It supports 1 or more score columns (`env.n_columns`).
- **Agent**: An ActorCritic network with configurable width, depth, activation, LayerNorm and dropout. It has an optional auxiliary head that predicts the upper-section score (used by A2C).
- **FOMAML inner loop**: PPO updates (clipped objective + GAE) run for `n_inner_steps` on a sampled task. Configs with an `a2c:` section instead of `ppo:` use an A2C inner loop.
- **FOMAML outer loop**: Samples `n_tasks_per_batch` reward-shaped tasks, adapts the policy on each, evaluates on a query rollout, and accumulates gradients back onto the shared meta-parameters.
- **Tasks**: Reward-shaping variants (e.g. threshold-beating) that stress-test different scoring strategies.
- **A2C**: Parallel-environment advantage actor-critic with entropy and learning-rate schedules, an upper-score regression loss and reward shaping, following the paper.
- **Experiment tracking**: [ClearML](https://clear.ml) by default, or local TensorBoard with `CLEARML_OFF=1`.

## Project structure

```
configs/          # YAML configs: FOMAML (default/server/test, *-a2c = A2C inner loop)
                  #   and standalone A2C (standalone-a2c, test-standalone-a2c)
scripts/
  train.py            # FOMAML training entry point
  train_a2c.py        # Standalone A2C training entry point
  evaluate.py         # Evaluate a FOMAML checkpoint + ClearML upload
  fetch_experiment.py # Download a ClearML run (config, scalars, artifacts)
src/
  env/            # YahtzeeEnv, scoring, constants
  agents/         # ActorCritic, PPO and A2C losses
  meta/           # FOMAML outer loop, inner loop
  tasks/          # Reward-shaping task definitions
  training/       # MetaTrainer, A2CTrainer, Evaluator
  logging_utils/  # ClearML / TensorBoard loggers
tests/            # pytest suite
notebooks/        # Analysis notebooks
checkpoints/      # Saved checkpoints (created at runtime, git-ignored)
```

## Requirements

- Python >= 3.10
- [uv](https://github.com/astral-sh/uv) (package manager)
- Optional: a GPU. Everything also runs on CPU.

## Setup

CPU (works everywhere):

```bash
uv sync
```

NVIDIA GPU (CUDA 12):

```bash
uv sync --extra cuda
```

For other accelerators (e.g. AMD ROCm), install the matching JAX build following the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html). The code itself is backend-agnostic. At startup, training prints `JAX devices: [...]` so you can check which device is in use.

For dev dependencies (Jupyter, pytest), add `--group dev`.

### Experiment tracking

Experiments log to ClearML by default. Copy the env file and fill in your credentials:

```bash
cp .env.example .env   # set CLEARML_API_ACCESS_KEY / CLEARML_API_SECRET_KEY
```

If you don't have a ClearML account, set `CLEARML_OFF=1` to log to TensorBoard in `runs/`:

```bash
CLEARML_OFF=1 uv run python scripts/train_a2c.py --config configs/test-standalone-a2c.yaml
uv run tensorboard --logdir runs/
```

## Training

**FOMAML** (PPO inner loop; use `configs/default-a2c.yaml` for an A2C inner loop):

```bash
uv run python scripts/train.py --config configs/default.yaml
```

**Standalone A2C** (paper reproduction):

```bash
uv run python scripts/train_a2c.py --config configs/standalone-a2c.yaml
```

For a quick smoke test, use `configs/test.yaml` (FOMAML) or `configs/test-standalone-a2c.yaml` (A2C).

### Checkpoints

Checkpoints are written to `<checkpoint_dir>/<task_id>/step_<N>/` (`params.npz`, `opt_state.npz`, `meta.npz`), where `checkpoint_dir` comes from the config (default `checkpoints/`). To resume training, pass the step directory:

```bash
uv run python scripts/train.py --config configs/default.yaml --resume checkpoints/<task_id>/step_500
```

### Docker (NVIDIA GPU)

The Docker setup requires an NVIDIA GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
docker compose up --build                     # FOMAML (configs/server.yaml)
GPU_ID=0 docker compose run --rm trainer-a2c --config configs/standalone-a2c.yaml
```

The compose file mounts `configs/`, `checkpoints/`, `data/` and `runs/`, so your runs persist on the host. A VS Code devcontainer (`.devcontainer/`) with the same CUDA setup is also included.

## Configuration

Key FOMAML settings in `configs/default.yaml`:

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `meta` | `n_meta_steps` | 10000 | Total outer-loop iterations |
| `meta` | `n_tasks_per_batch` | 3 | Tasks sampled per meta-step |
| `meta` | `n_inner_steps` | 5 | PPO steps per task adaptation |
| `meta` | `n_parallel_envs` | 16 | Parallel episodes per task per inner step |
| `meta` | `inner_lr` | 0.01 | Inner loop learning rate |
| `meta` | `outer_lr` | 0.0003 | Outer loop (Adam) learning rate |
| `ppo` | `clip_epsilon` | 0.2 | PPO clip ratio |
| `training` | `checkpoint_every` | 500 | Save frequency (meta-steps) |
| `training` | `checkpoint_dir` | `checkpoints/` | Where checkpoints are written |

The A2C settings (network, entropy/LR schedules, auxiliary losses) are documented inline in `configs/standalone-a2c.yaml`.

## Evaluation

```bash
uv run python scripts/evaluate.py --checkpoint checkpoints/<task_id>/step_500 --config configs/default.yaml
```

Results and artifacts are uploaded to the configured ClearML project.

## Tests

```bash
uv sync --group dev
uv run pytest
```

## License

GPL-3.0. See [LICENSE](LICENSE).
