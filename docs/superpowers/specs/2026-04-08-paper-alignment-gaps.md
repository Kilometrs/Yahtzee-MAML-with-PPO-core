# Paper-Alignment Gaps Spec

**Goal:** Close the remaining gaps between our standalone A2C implementation and the reference paper (Pape 2025, arXiv 2601.00007, [code](https://github.com/papetronics/case-studies-final-project)).

**Branch:** `experimental/standalone-a2c`

---

## Gap 1: Entropy schedule fractions

Config-only. Change `configs/standalone-a2c.yaml`:

```yaml
# Before
entropy_hold: 0.075
entropy_anneal: 0.9

# After (matching paper Table 1 "Baseline" regime)
entropy_hold: 0.3
entropy_anneal: 0.6
```

No code changes.

---

## Gap 2: Upper score auxiliary head + reward shaping

### 2a. New output head on ActorCritic

Add a 4th output head to `src/agents/actor_critic.py` that predicts the normalized final upper section score.

**Architecture** (paper Fig 2):
```
trunk output (x)
  → Dense(head_hidden_dim)
  → Swish
  → LayerNorm
  → Dense(1)  # no activation
```

**Output range:** `[-1, 5/3]` representing `(upper_final / 63) - 1`.

**Return signature change:** `model.apply(params, obs, phase)` currently returns `(logits, value)`. Change to `(logits, value, upper_pred)`. The upper_pred is always returned (scalar). Callers that don't need it can ignore it with `_`.

**Backward compatibility:** All existing call sites (PPO loss, FOMAML inner loop, evaluator, etc.) must be updated to unpack the new 3-tuple. Places that don't use `upper_pred` just discard it.

**Condition:** Only create the head when `head_hidden_dim > 0`. When `head_hidden_dim == 0`, return `0.0` as a dummy upper_pred so the signature stays consistent.

### 2b. Regression loss in A2C loss function

Add auxiliary L2 loss term to `src/agents/a2c.py`:

```
L_upper = upper_regression_weight * mean((upper_pred - upper_target)^2)
```

**Target:** The actual final upper section score for the episode, normalized: `upper_final / 63 - 1`. This is a per-episode constant broadcast to all timesteps in that episode.

**Signature change:** `a2c_loss` gets new optional params:
- `upper_regression_weight: float = 0.0` (0.0 = disabled, backward compat)
- `upper_targets: jnp.ndarray = None` (shape matches batch, normalized final upper scores)

When `upper_regression_weight > 0` and `upper_targets` is provided, compute the regression loss and add it. Otherwise, no effect.

### 2c. Potential-based reward shaping

In `src/training/a2c_trainer.py`, modify `_collect_episodes` and `_prepare_data`:

**Collection phase:** At each timestep, the model now outputs `upper_pred` alongside logits and values. Save `upper_pred` in the trajectory data.

**Shaping phase (post-collection, pre-advantage computation):**

1. Compute potential: `Φ(s) = 35 * clamp(63 * (upper_pred + 1), 0, 63)`
2. Compute Φ(s') by shifting: `Φ_next[t] = Φ[t+1]` for t < T-1, `Φ_next[T-1] = 0`
3. At episode boundaries (where done=True), set `Φ_next = 0`
4. Shaped reward: `R'[t] = R[t] + β_shape * (γ * Φ_next[t] - Φ[t])`

**Upper score targets:** After collection, extract the upper section score from the environment state at episode-end timesteps. Normalize as `upper_score / 63 - 1`. Backward-fill to all timesteps in the same episode (scan backward, carry the target, reset on done).

**Config knobs** in `a2c` section:
```yaml
upper_regression_weight: 1.0
upper_shaping_weight: 1.0
```

Both default to 0.0 for backward compatibility.

### 2d. Impact on FOMAML path

The FOMAML inner loop (`inner_loop.py`, `maml.py`) calls the model and A2C/PPO loss. These need to handle the new 3-tuple return from ActorCritic. The upper score features are ignored in the FOMAML path — `upper_regression_weight` stays 0.0 in FOMAML configs. Changes are limited to unpacking the extra return value.

---

## Gap 3: Missing observation features

Add to `make_obs` in `src/env/yahtzee_env.py`:

### 3a. Potential scoring opportunities (14 dims)

For each of the 13 categories, compute `score_category(dice, i) / MAX_SCORE[i]` where `MAX_SCORE` is a constant array of maximum achievable scores per category:
```
[5, 10, 15, 20, 25, 30, 30, 30, 25, 30, 40, 50, 30]
```
(Ones max=5, Twos max=10, ..., Chance max=30)

Plus 1 joker indicator: `1.0` if all dice are the same AND the Yahtzee slot is already filled with 50.

Total: 14 new dims.

### 3b. Phase scalar (1 dim)

`float(state.phase)` — 0.0 for roll, 1.0 for score.

### 3c. Has-earned-yahtzee (1 dim)

`1.0` if any column has scored exactly 50 in the Yahtzee category (index 11).

### obs_dim update

Formula changes from `41 + 33*n_cols` to `57 + 33*n_cols`.

Breakdown of fixed portion: 30 (dice_onehot) + 6 (bin_counts) + 3 (rolls_oh) + 14 (potential_scores) + 1 (phase) + 1 (has_yahtzee) + 1 (yahtzee_bonus) + 1 (game_progress) = 57.

Per-column: 13 (filled) + 13 (scores) + 1 (upper_progress) + 6 (lock-in) = 33. Unchanged.

All `obs_dim()` callers (tests, meta_trainer, etc.) automatically pick up the change since they call the function.

### Constants needed

Add `MAX_CATEGORY_SCORES` array to `src/env/constants.py`:
```python
MAX_CATEGORY_SCORES = [5, 10, 15, 20, 25, 30, 30, 30, 25, 30, 40, 50, 30]
```

---

## Files changed

| File | Changes |
|------|---------|
| `src/env/constants.py` | Update `obs_dim`, add `MAX_CATEGORY_SCORES` |
| `src/env/yahtzee_env.py` | Add 3 new feature groups to `make_obs` |
| `src/agents/actor_critic.py` | Add upper score prediction head, return 3-tuple |
| `src/agents/a2c.py` | Add upper regression loss term |
| `src/agents/ppo.py` | Update to unpack 3-tuple (discard upper_pred) |
| `src/agents/common.py` | No change |
| `src/training/a2c_trainer.py` | Reward shaping, upper targets, pass regression weight |
| `src/meta/inner_loop.py` | Unpack 3-tuple from model |
| `src/meta/maml.py` | No change (reads loss_fn from config) |
| `src/training/evaluator.py` | Unpack 3-tuple from model |
| `configs/standalone-a2c.yaml` | Entropy schedule, upper score weights |
| `tests/env/test_yahtzee_env.py` | Updated OBS_DIM, new feature tests |
| `tests/agents/test_actor_critic.py` | Updated for 3-tuple, upper head tests |
| `tests/agents/test_a2c.py` | Updated OBS_DIM |
| `tests/agents/test_ppo.py` | Updated OBS_DIM, 3-tuple |
| `tests/meta/test_maml.py` | Updated OBS_DIM |
| `tests/test_integration.py` | Updated OBS_DIM, configs |
