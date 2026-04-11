# Yahtzee-MAML-with-PPO-core

A meta-reinforcement learning system that trains an agent to play Yahtzee using **First-Order MAML (FOMAML)** with a **PPO** inner loop. The meta-learner finds a policy initialization that can rapidly adapt to any reward-shaped Yahtzee task in just a few gradient steps.

## How it works

- **Environment**: A full Yahtzee gym environment with all 13 scoring categories, rerolls, upper-section bonus logic, and valid-action masking.
- **Agent**: An ActorCritic network with configurable hidden dimensions and depth.
- **Inner loop**: PPO updates (clipped objective + GAE advantage estimation) run for `n_inner_steps` on a sampled task.
- **Outer loop (FOMAML)**: Samples `n_tasks_per_batch` reward-shaped tasks, adapts the policy on each, evaluates on a query rollout, and accumulates gradients back onto the shared meta-parameters.
- **Tasks**: Reward-shaping variants (e.g. threshold-beating) that stress-test different scoring strategies.
- **Experiment tracking**: [ClearML](https://clear.ml) for metrics, artifacts, and checkpoints.

## Project structure

```
configs/          # YAML configs (default, server, …)
scripts/
  train.py        # Training entry point
  evaluate.py     # Evaluation + ClearML upload
src/
  env/            # YahtzeeEnv, scoring, constants
  agents/         # ActorCritic (PPO)
  meta/           # FOMAML outer loop, inner loop
  tasks/          # Reward-shaping task definitions
  training/       # MetaTrainer orchestration
  logging_utils/  # Shared logging helpers
checkpoints/      # Saved model checkpoints
notebooks/        # Analysis notebooks
```

## Requirements

- Python >= 3.10
- CUDA-capable GPU (recommended)
- [uv](https://github.com/astral-sh/uv) (package manager)

## Setup

```bash
uv sync
```

For dev dependencies (Jupyter, pytest):

```bash
uv sync --group dev
```

Copy and edit the environment file if you want ClearML tracking:

```bash
cp .env.example .env   # then fill in CLEARML_API_KEY etc.
```

## Start training

**Locally:**

```bash
uv run python scripts/train.py --config configs/default.yaml
```

Resume from a checkpoint:

```bash
uv run python scripts/train.py --config configs/default.yaml --resume checkpoints/500.pt
```

**Via Docker (GPU):**

```bash
docker compose up --build
```

**Standalone A2C (interactive, e.g. inside tmux):**

```bash
GPU_ID=1 docker compose run --rm -e TF_CPP_MIN_LOG_LEVEL=2 -e CLEARML_OFF=0 trainer-a2c --config configs/standalone-a2c.yaml
```

The compose file mounts `configs/`, `checkpoints/`, and `data/` so your runs persist on the host.

## Configuration

Edit `configs/default.yaml` to tune the key knobs:

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `meta` | `n_meta_steps` | 10000 | Total outer-loop iterations |
| `meta` | `n_tasks_per_batch` | 3 | Tasks sampled per meta-step |
| `meta` | `n_inner_steps` | 5 | PPO steps per task adaptation |
| `meta` | `inner_lr` | 0.01 | Inner loop learning rate |
| `meta` | `outer_lr` | 0.0003 | Outer loop (Adam) learning rate |
| `ppo` | `clip_epsilon` | 0.2 | PPO clip ratio |
| `training` | `checkpoint_every` | 500 | Save frequency (meta-steps) |
| `training` | `device` | `cuda` | `cuda` or `cpu` |

## Evaluation

```bash
uv run python scripts/evaluate.py --checkpoint checkpoints/500.pt --config configs/default.yaml
```

Results and artifacts are uploaded to the configured ClearML project.

## Tests

```bash
uv run pytest
```
