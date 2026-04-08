# Paper-Alignment Gaps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining gaps between the standalone A2C implementation and the reference paper (Pape 2025, arXiv 2601.00007) — add missing observation features, upper score auxiliary head with reward shaping, and fix entropy schedule.

**Architecture:** Three independent changes: (1) add 16 new obs features to shared `make_obs`, (2) add upper score prediction head to ActorCritic with 3-tuple return + regression loss + potential-based reward shaping in the standalone trainer, (3) fix entropy hold/anneal config. The model return signature changes from `(logits, value)` to `(logits, value, upper_pred)` which requires updating all call sites. The FOMAML path ignores upper_pred but must still unpack it.

**Tech Stack:** JAX, Flax Linen, jax.numpy, optax, pytest

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/env/constants.py` | Modify | Update `obs_dim()`, add `MAX_CATEGORY_SCORES` |
| `src/env/yahtzee_env.py` | Modify | Add potential scores, phase, has_yahtzee to `make_obs` |
| `src/agents/actor_critic.py` | Modify | Add upper score head, return 3-tuple |
| `src/agents/a2c.py` | Modify | Add upper regression loss term |
| `src/agents/ppo.py` | Modify | Unpack 3-tuple (discard upper_pred) |
| `src/meta/inner_loop.py` | Modify | Unpack 3-tuple in episode collection |
| `src/training/evaluator.py` | Modify | Unpack 3-tuple in eval loop |
| `src/training/a2c_trainer.py` | Modify | Reward shaping, upper targets, regression weight |
| `configs/standalone-a2c.yaml` | Modify | Entropy schedule, upper score weights |
| `tests/env/test_yahtzee_env.py` | Modify | Updated OBS_DIM, new feature tests |
| `tests/agents/test_actor_critic.py` | Modify | 3-tuple tests, upper head tests |
| `tests/agents/test_a2c.py` | Modify | Updated OBS_DIM, regression loss test |
| `tests/agents/test_ppo.py` | Modify | Updated OBS_DIM, 3-tuple unpack |
| `tests/meta/test_maml.py` | Modify | Updated OBS_DIM |
| `tests/test_integration.py` | Modify | Updated OBS_DIM |

---

### Task 1: Add missing observation features to make_obs

Add potential scoring opportunities (14 dims), phase scalar (1 dim), has-earned-yahtzee (1 dim) to `make_obs`. Update `obs_dim` from `41 + 33*n_cols` to `57 + 33*n_cols`.

**Files:**
- Modify: `src/env/constants.py`
- Modify: `src/env/yahtzee_env.py:51-80`
- Modify: `tests/env/test_yahtzee_env.py`

- [ ] **Step 1: Add MAX_CATEGORY_SCORES and update obs_dim in constants.py**

In `src/env/constants.py`, add after `PHASE_SCORE = 1`:

```python
# Maximum achievable score per category (for normalizing potential scores)
MAX_CATEGORY_SCORES = jnp.array([
    5, 10, 15, 20, 25, 30,   # Ones..Sixes (N_DICE * face_value)
    30, 30, 25, 30, 40, 50, 30,  # 3oK, 4oK, FH, SS, LS, Yahtzee, Chance
], dtype=jnp.float32)
```

Add `import jax.numpy as jnp` at the top of constants.py.

Update `obs_dim`:

```python
def obs_dim(n_columns):
    """Observation vector length: 57 + 33 * n_columns.

    Components: sorted one-hot dice (30) + bin counts (6) + one-hot rolls (3)
    + potential scores (13) + joker indicator (1) + phase (1) + has_yahtzee (1)
    + filled mask (13*n_cols) + scores (13*n_cols) + yahtzee bonus (1)
    + upper bonus progress (n_cols) + lock-in (6*n_cols) + game progress (1).
    """
    return 57 + 33 * n_columns
```

- [ ] **Step 2: Update make_obs in yahtzee_env.py**

Replace the existing `make_obs` function with:

```python
def make_obs(state, n_columns):
    sorted_dice = jnp.sort(state.dice)
    dice_onehot = jax.nn.one_hot(sorted_dice - 1, N_SIDES).flatten()  # (30,)
    bin_counts = jnp.zeros(N_SIDES, dtype=jnp.float32).at[sorted_dice - 1].add(1.0)  # (6,)
    rolls_onehot = jax.nn.one_hot(state.rerolls, MAX_REROLLS + 1)  # (3,)

    all_scores = compute_all_scores(sorted_dice)
    potential_normalized = all_scores.astype(jnp.float32) / MAX_CATEGORY_SCORES  # (13,)
    all_dice_same = jnp.all(sorted_dice == sorted_dice[0])
    yahtzee_filled = jnp.any(state.scores[YAHTZEE, :] == 50)
    joker = jnp.array([all_dice_same & yahtzee_filled], dtype=jnp.float32)  # (1,)

    phase = jnp.array([state.phase], dtype=jnp.float32)  # (1,)
    has_yahtzee = jnp.array([yahtzee_filled], dtype=jnp.float32)  # (1,)

    upper_sums = jnp.sum(state.scores[:6, :], axis=0).astype(jnp.float32)
    upper_progress = jnp.minimum(upper_sums / UPPER_BONUS_THRESHOLD, 1.0)  # (n_cols,)

    upper_with_score = upper_sums + all_scores[:6, None]  # (6, n_cols) broadcast
    lockin = (upper_with_score >= UPPER_BONUS_THRESHOLD).astype(jnp.float32)
    lockin = lockin * (~state.filled_mask[:6, :]).astype(jnp.float32)
    lockin_flat = lockin.flatten()  # (6*n_cols,)

    n_filled = jnp.sum(state.filled_mask).astype(jnp.float32)
    total_slots = N_CATEGORIES * n_columns
    game_progress = jnp.array([n_filled / total_slots])  # (1,)

    return jnp.concatenate([
        dice_onehot,                                           # 30
        bin_counts,                                            # 6
        rolls_onehot,                                          # 3
        potential_normalized,                                  # 13
        joker,                                                 # 1
        phase,                                                 # 1
        has_yahtzee,                                           # 1
        state.filled_mask.flatten().astype(jnp.float32),       # 13 * n_cols
        state.scores.flatten().astype(jnp.float32),            # 13 * n_cols
        jnp.array([state.yahtzee_bonus], dtype=jnp.float32),   # 1
        upper_progress,                                        # n_cols
        lockin_flat,                                           # 6 * n_cols
        game_progress,                                         # 1
    ])
```

Add `MAX_CATEGORY_SCORES` and `YAHTZEE` to the imports from constants at the top of the file (YAHTZEE is already imported).

- [ ] **Step 3: Update tests**

In `tests/env/test_yahtzee_env.py`, the `OBS_DIM` is already computed via `obs_dim(N_COLS)` so it auto-updates. Update the `TestMakeObsFeatures` class:

- Fix `test_rolls_onehot_section`: offset changes from `obs[36:39]` to `obs[36:39]` (unchanged — rolls are still at index 36).
- Add `test_potential_scores_section`:
```python
def test_potential_scores_section(self):
    rng = jax.random.PRNGKey(0)
    state, _ = env_reset(rng, N_COLS)
    obs = make_obs(state, N_COLS)
    potential = obs[39:52]  # 13 normalized potential scores
    assert jnp.all(potential >= 0.0)
    assert jnp.all(potential <= 1.0)
```

- Add `test_phase_feature`:
```python
def test_phase_feature(self):
    rng = jax.random.PRNGKey(0)
    state, _ = env_reset(rng, N_COLS)
    obs = make_obs(state, N_COLS)
    assert obs[53] == 0.0  # Roll phase
```

- Update `test_upper_progress_starts_zero` offset: `up_start = 55 + 26 * N_COLS + 1`

- Update `test_game_progress_starts_zero`: `obs[-1] == 0.0` (unchanged — still last element).

- [ ] **Step 4: Run env tests**

Run: `cd /app && python -m pytest tests/env/test_yahtzee_env.py -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/env/constants.py src/env/yahtzee_env.py tests/env/test_yahtzee_env.py
git commit -m "feat: add potential scores, phase, has_yahtzee to observation"
```

---

### Task 2: Update ActorCritic to return 3-tuple with upper score head

Add a 4th output head that predicts normalized final upper section score. Change return from `(logits, value)` to `(logits, value, upper_pred)`.

**Files:**
- Modify: `src/agents/actor_critic.py`
- Modify: `tests/agents/test_actor_critic.py`

- [ ] **Step 1: Update ActorCritic class**

Replace the class in `src/agents/actor_critic.py`:

```python
class ActorCritic(nn.Module):
    hidden_dim: int = 600
    n_layers: int = 2
    n_columns: int = 6
    use_layer_norm: bool = True
    activation: str = "swish"
    dropout_rate: float = 0.0
    head_hidden_dim: int = 0
    norm_position: str = "pre"

    @nn.compact
    def __call__(self, obs: jnp.ndarray, phase: jnp.int32,
                 deterministic: bool = True):
        max_actions = max(32, 13 * self.n_columns)
        act_fn = nn.swish if self.activation == "swish" else nn.relu

        x = obs
        for _ in range(self.n_layers):
            x = nn.Dense(self.hidden_dim)(x)
            if self.norm_position == "pre" and self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = act_fn(x)
            if self.norm_position == "post" and self.use_layer_norm:
                x = nn.LayerNorm()(x)
            if self.dropout_rate > 0.0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

        if self.head_hidden_dim > 0:
            roll_h = act_fn(nn.Dense(self.head_hidden_dim, name="roll_hidden")(x))
            if self.use_layer_norm:
                roll_h = nn.LayerNorm(name="roll_ln")(roll_h)
            roll_logits = nn.Dense(32, name="roll_out")(roll_h)

            score_h = act_fn(nn.Dense(self.head_hidden_dim, name="score_hidden")(x))
            if self.use_layer_norm:
                score_h = nn.LayerNorm(name="score_ln")(score_h)
            score_logits = nn.Dense(13 * self.n_columns, name="score_out")(score_h)

            upper_h = act_fn(nn.Dense(self.head_hidden_dim, name="upper_hidden")(x))
            if self.use_layer_norm:
                upper_h = nn.LayerNorm(name="upper_ln")(upper_h)
            upper_pred = nn.Dense(1, name="upper_out")(upper_h).squeeze(-1)
        else:
            roll_logits = nn.Dense(32)(x)
            score_logits = nn.Dense(13 * self.n_columns)(x)
            upper_pred = jnp.float32(0.0)

        value = nn.elu(nn.Dense(1)(x)).squeeze(-1)

        roll_padded = jnp.concatenate([roll_logits, jnp.full(max_actions - 32, -jnp.inf)])
        score_padded = score_logits
        if 13 * self.n_columns < max_actions:
            score_padded = jnp.concatenate([score_logits, jnp.full(max_actions - 13 * self.n_columns, -jnp.inf)])

        logits = jax.lax.cond(phase == 0, lambda: roll_padded, lambda: score_padded)
        return logits, value, upper_pred
```

- [ ] **Step 2: Update test_actor_critic.py**

All assertions checking `(logits, value)` tuple must unpack 3 values. Update fixture and all tests that call `model.apply`:

Change every `logits, value = model.apply(...)` to `logits, value, upper_pred = model.apply(...)`.

Add tests:
```python
def test_output_is_3_tuple(self, model_and_params):
    model, params = model_and_params
    result = model.apply(params, jnp.ones(OBS_DIM), jnp.int32(0))
    assert len(result) == 3

def test_upper_pred_with_head(self):
    model = ActorCritic(hidden_dim=64, n_layers=2, n_columns=N_COLS,
                        head_hidden_dim=32)
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, jnp.zeros(OBS_DIM), jnp.int32(0))
    _, _, upper_pred = model.apply(params, jnp.ones(OBS_DIM), jnp.int32(0))
    assert upper_pred.shape == ()
    assert jnp.isfinite(upper_pred)

def test_upper_pred_zero_without_head(self):
    model = ActorCritic(hidden_dim=64, n_layers=2, n_columns=N_COLS,
                        head_hidden_dim=0)
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, jnp.zeros(OBS_DIM), jnp.int32(0))
    _, _, upper_pred = model.apply(params, jnp.ones(OBS_DIM), jnp.int32(0))
    assert upper_pred == 0.0
```

- [ ] **Step 3: Run tests**

Run: `cd /app && python -m pytest tests/agents/test_actor_critic.py -v`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add src/agents/actor_critic.py tests/agents/test_actor_critic.py
git commit -m "feat: add upper score prediction head, return 3-tuple"
```

---

### Task 3: Update all 3-tuple unpack sites (PPO, A2C loss, inner_loop, evaluator)

Every file that calls `model.apply` and unpacks `(logits, value)` must be updated to `(logits, value, _)` or `(logits, value, upper_pred)`.

**Files:**
- Modify: `src/agents/ppo.py:11`
- Modify: `src/agents/a2c.py:28,34`
- Modify: `src/meta/inner_loop.py:32`
- Modify: `src/training/evaluator.py:36`
- Modify: `src/training/a2c_trainer.py:90,257,259`
- Modify: `tests/agents/test_ppo.py`
- Modify: `tests/agents/test_a2c.py`
- Modify: `tests/meta/test_maml.py`
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Update ppo.py**

Line 11, change:
```python
logits, values = jax.vmap(model.apply, in_axes=(None, 0, 0))(params, obs, phases)
```
to:
```python
logits, values, _ = jax.vmap(model.apply, in_axes=(None, 0, 0))(params, obs, phases)
```

- [ ] **Step 2: Update a2c.py**

Line 28, change `logits, values = ...` to `logits, values, _ = ...`
Line 34, change `logits, values = ...` to `logits, values, _ = ...`

(The upper_pred will be used by `a2c_loss` in a later task, but for now discard it to keep things compiling.)

- [ ] **Step 3: Update inner_loop.py**

Line 32, change `logits, values = ...` to `logits, values, _ = ...`

- [ ] **Step 4: Update evaluator.py**

Line 36, change `logits, value = model.apply(...)` to `logits, value, _ = model.apply(...)`

- [ ] **Step 5: Update a2c_trainer.py**

Line 90: `logits, values = ...` → `logits, values, _ = ...`
Line 257: `logits, values = ...` → `logits, values, _ = ...`
Line 259: `logits, values = ...` → `logits, values, _ = ...`

(These will be updated again in Task 5 to capture `upper_pred` for shaping.)

- [ ] **Step 6: Run all tests**

Run: `cd /app && python -m pytest -v`
Expected: All pass.

- [ ] **Step 7: Commit**

```bash
git add src/agents/ppo.py src/agents/a2c.py src/meta/inner_loop.py src/training/evaluator.py src/training/a2c_trainer.py tests/
git commit -m "refactor: update all call sites for 3-tuple model return"
```

---

### Task 4: Add upper regression loss to A2C loss function

Add optional L2 regression term for upper score prediction.

**Files:**
- Modify: `src/agents/a2c.py`
- Modify: `tests/agents/test_a2c.py`

- [ ] **Step 1: Update a2c_loss to accept and use upper targets**

Replace the full `a2c_loss` function in `src/agents/a2c.py`:

```python
def a2c_loss(params, model, obs, actions, phases, masks,
             old_log_probs, advantages, returns,
             entropy_coef=0.01, value_loss_coef=0.5,
             entropy_coef_roll=None, entropy_coef_score=None,
             upper_targets=None, upper_regression_weight=0.0,
             deterministic=True, rng=None):
    """A2C loss with optional upper score regression (Pape 2025 Section 4.5.1).

    old_log_probs is accepted but unused — keeps the inner loop
    algorithm-agnostic.

    Args:
        upper_targets: Normalized final upper scores per timestep. Shape (batch,).
            Target is (upper_final / 63) - 1, range [-1, 5/3].
            None or weight=0.0 disables the regression loss.
        upper_regression_weight: Coefficient for upper score regression loss.
    """
    if deterministic or rng is None:
        logits, values, upper_preds = jax.vmap(model.apply, in_axes=(None, 0, 0))(
            params, obs, phases)
    else:
        rngs = jax.random.split(rng, obs.shape[0])
        def apply_with_dropout(obs_i, phase_i, rng_i):
            return model.apply(params, obs_i, phase_i, deterministic=False,
                               rngs={"dropout": rng_i})
        logits, values, upper_preds = jax.vmap(apply_with_dropout)(obs, phases, rngs)

    logits = jnp.where(masks, logits, -jnp.inf)
    log_probs = jax.nn.log_softmax(logits)
    action_log_probs = jnp.take_along_axis(log_probs, actions[:, None], axis=1).squeeze(1)
    probs = jax.nn.softmax(logits)
    safe_log_probs = jnp.where(masks, log_probs, 0.0)
    entropy = -jnp.sum(probs * safe_log_probs, axis=-1)

    policy_loss = -(action_log_probs * advantages).mean()
    value_loss = 0.5 * jnp.mean((values - returns) ** 2)

    eff_roll = entropy_coef if entropy_coef_roll is None else entropy_coef_roll
    eff_score = entropy_coef if entropy_coef_score is None else entropy_coef_score

    is_roll = (phases == 0).astype(jnp.float32)
    is_score = (phases == 1).astype(jnp.float32)
    n_roll = jnp.maximum(is_roll.sum(), 1.0)
    n_score = jnp.maximum(is_score.sum(), 1.0)
    roll_entropy = (entropy * is_roll).sum() / n_roll
    score_entropy = (entropy * is_score).sum() / n_score
    entropy_loss = -(eff_roll * roll_entropy + eff_score * score_entropy)

    upper_loss = jnp.where(
        (upper_targets is not None) & (upper_regression_weight > 0.0),
        upper_regression_weight * jnp.mean((upper_preds - upper_targets) ** 2),
        0.0,
    ) if upper_targets is not None else 0.0

    return policy_loss + value_loss_coef * value_loss + entropy_loss + upper_loss
```

- [ ] **Step 2: Add test for regression loss**

In `tests/agents/test_a2c.py`, add:

```python
def test_upper_regression_loss(self, setup):
    """Upper regression loss should increase total loss when enabled."""
    base_loss = a2c_loss(setup["params"], setup["model"], setup["obs"],
                         setup["actions"], setup["phases"], setup["masks"],
                         setup["old_log_probs"], setup["advantages"], setup["returns"])
    upper_targets = jnp.ones(setup["obs"].shape[0]) * 0.5
    reg_loss = a2c_loss(setup["params"], setup["model"], setup["obs"],
                        setup["actions"], setup["phases"], setup["masks"],
                        setup["old_log_probs"], setup["advantages"], setup["returns"],
                        upper_targets=upper_targets, upper_regression_weight=1.0)
    assert not jnp.allclose(base_loss, reg_loss, atol=1e-4)
```

- [ ] **Step 3: Run tests**

Run: `cd /app && python -m pytest tests/agents/test_a2c.py -v`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add src/agents/a2c.py tests/agents/test_a2c.py
git commit -m "feat: add upper score regression loss to A2C"
```

---

### Task 5: Wire reward shaping and upper targets into standalone A2C trainer

This is the most complex task. Modify episode collection to capture `upper_pred` and upper scores at episode boundaries, compute shaped rewards and regression targets, and pass them through to the loss function.

**Files:**
- Modify: `src/training/a2c_trainer.py`

- [ ] **Step 1: Update _collect_episodes to capture upper_pred and upper scores**

In the `scan_step` function inside `_collect_episodes`, change the model call to capture `upper_pred`:

```python
logits, values, upper_preds = jax.vmap(model.apply, in_axes=(None, 0, 0))(
    params, obs, phases)
```

Also capture the upper section score from the state at each step:
```python
upper_score = jnp.sum(states.scores[:6, :]).astype(jnp.float32)
```

Note: this is the total upper score summed across all columns, computed per-env. For vectorized envs, use:
```python
upper_scores = jax.vmap(lambda s: jnp.sum(s.scores[:6, :]).astype(jnp.float32))(states)
```

Actually, since `states.scores` has shape `(n_envs, 13, n_cols)` inside vmap scan, we can compute:
```python
upper_scores = jnp.sum(states.scores[:, :6, :], axis=(1, 2)).astype(jnp.float32)
```

Update `step_data` to include `upper_preds` and `upper_scores`:
```python
step_data = (obs, actions, action_log_probs, rewards, dones,
             values, phases, masks, upper_preds, upper_scores)
```

- [ ] **Step 2: Update _prepare_data with reward shaping and upper targets**

Add new params to `_prepare_data`:

```python
def _prepare_data(trajectories, *, gamma, gae_lambda, upper_shaping_weight=0.0):
```

Unpack the extended trajectory:
```python
obs, actions, log_probs, rewards, dones, values, phases, masks, upper_preds, upper_scores = trajectories
```

**Compute shaped rewards** (before GAE):
```python
if upper_shaping_weight > 0.0:
    # Potential function: Φ(s) = 35 * clamp(63 * (upper_pred + 1), 0, 63)
    phi = 35.0 * jnp.clip(63.0 * (upper_preds + 1.0), 0.0, 63.0)
    # Φ(s') = next step's phi, 0 at episode boundaries
    phi_next = jnp.concatenate([phi[1:], jnp.zeros_like(phi[:1])], axis=0)
    phi_next = phi_next * (1.0 - dones)  # zero at episode ends
    shaping = upper_shaping_weight * (gamma * phi_next - phi)
    rewards = rewards + shaping
```

**Compute upper targets** (backward-fill final upper score per episode):
```python
# upper_scores has shape (max_steps, n_envs)
# Normalize: target = upper_score_at_episode_end / 63 - 1
# Backward-fill: carry the end-of-episode upper score back to all steps

def backward_fill_one_env(upper_scores_env, dones_env):
    """Scan backward: at done, capture upper score; carry backward."""
    def scan_fn(carry, t):
        next_target = t[0] / 63.0 - 1.0
        is_done = t[1]
        target = jnp.where(is_done, next_target, carry)
        return target, target
    _, targets = jax.lax.scan(
        scan_fn, 0.0,
        (upper_scores_env[::-1], dones_env[::-1]),
    )
    return targets[::-1]

upper_targets = jax.vmap(backward_fill_one_env, in_axes=(1, 1), out_axes=1)(
    upper_scores, dones)
```

Then flatten `upper_targets` alongside the rest:
```python
upper_targets_flat = upper_targets.reshape(T)
```

Return the extended tuple:
```python
return (obs_flat, actions_flat, phases_flat, masks_flat,
        log_probs_flat, advantages_flat, returns_flat, upper_targets_flat)
```

- [ ] **Step 3: Update A2CTrainer to wire everything together**

In `__init__`, read new config:
```python
self.upper_regression_weight = a2c_cfg.get("upper_regression_weight", 0.0)
self.upper_shaping_weight = a2c_cfg.get("upper_shaping_weight", 0.0)
```

Update `self._prepare`:
```python
self._prepare = jax.jit(
    functools.partial(_prepare_data,
                      gamma=self.gamma,
                      gae_lambda=self.gae_lambda,
                      upper_shaping_weight=self.upper_shaping_weight),
)
```

In `_build_train_step`, capture the new config:
```python
upper_regression_weight = self.upper_regression_weight
```

Update the data unpack inside `train_step`:
```python
data = prepare_fn(trajectories)
obs, actions, phases, masks, log_probs, advantages, returns, upper_targets = data
```

Update the loss call:
```python
return a2c_loss(
    p, model, obs, actions, phases, masks,
    log_probs, advantages, returns,
    value_loss_coef=value_loss_coef,
    entropy_coef_roll=ent_roll,
    entropy_coef_score=ent_score,
    upper_targets=upper_targets,
    upper_regression_weight=upper_regression_weight,
)
```

- [ ] **Step 4: Run standalone A2C tests**

Run: `cd /app && python -m pytest tests/test_integration.py -v -k "a2c"`
Expected: Pass (integration test exercises the full pipeline).

- [ ] **Step 5: Commit**

```bash
git add src/training/a2c_trainer.py
git commit -m "feat: reward shaping and upper regression in standalone A2C trainer"
```

---

### Task 6: Update config and run full test suite

Fix entropy schedule and add upper score config knobs.

**Files:**
- Modify: `configs/standalone-a2c.yaml`
- Run full test suite

- [ ] **Step 1: Update standalone-a2c.yaml**

Change entropy schedule to match paper Table 1 "Baseline":
```yaml
a2c:
  gamma: 0.99
  gae_lambda: 0.0
  value_loss_coef: 0.005
  entropy_coef_roll: [0.1, 0.02]
  entropy_coef_score: [0.03, 0.01]
  entropy_hold: 0.3
  entropy_anneal: 0.6
  upper_regression_weight: 1.0
  upper_shaping_weight: 1.0
```

- [ ] **Step 2: Run full test suite**

Run: `cd /app && python -m pytest -v`
Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add configs/standalone-a2c.yaml
git commit -m "config: paper-aligned entropy schedule and upper score shaping weights"
```
