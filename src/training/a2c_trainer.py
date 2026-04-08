"""Standalone A2C trainer for Yahtzee (Pape 2025 reproduction).

No FOMAML, no meta-learning. Direct A2C with TD(0) advantages,
warmup/plateau/decay LR schedule, and annealed entropy coefficients.
"""
import os
import functools
import numpy as np
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from src.agents.actor_critic import ActorCritic
from src.agents.a2c import a2c_loss
from src.agents.common import compute_gae
from src.env.yahtzee_env import env_reset, env_step, make_obs, get_action_mask
from src.env.constants import obs_dim
from src.logging_utils import create_logger


def build_lr_schedule(peak_lr, min_ratio, warmup_frac, plateau_frac, total_steps):
    """Warmup -> plateau -> linear decay LR schedule (Pape 2025, Section 4.4.3).

    Args:
        peak_lr: Maximum learning rate.
        min_ratio: Final LR as fraction of peak (e.g. 0.05).
        warmup_frac: Fraction of total steps for linear warmup (e.g. 0.05).
        plateau_frac: Fraction of total steps at peak LR (e.g. 0.70).
        total_steps: Total number of gradient updates.

    Returns:
        An optax-compatible schedule function: step -> lr.
    """
    warmup_steps = int(total_steps * warmup_frac)
    plateau_steps = int(total_steps * plateau_frac)
    decay_steps = total_steps - warmup_steps - plateau_steps

    return optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, peak_lr, warmup_steps),
            optax.constant_schedule(peak_lr),
            optax.linear_schedule(peak_lr, peak_lr * min_ratio, decay_steps),
        ],
        boundaries=[warmup_steps, warmup_steps + plateau_steps],
    )


def anneal_entropy(step, total_steps, max_val, min_val, hold_frac, anneal_frac):
    """Linear entropy annealing: hold -> decay -> floor (Pape 2025, Table 6).

    Pure JAX implementation — safe to use inside JIT.

    Args:
        step: Current training step (JAX scalar or Python int).
        total_steps: Total number of gradient updates.
        max_val: Starting (maximum) entropy coefficient.
        min_val: Floor (minimum) entropy coefficient.
        hold_frac: Fraction of training to hold at max_val (e.g. 0.075).
        anneal_frac: Fraction of training for linear decay (e.g. 0.9).

    Returns:
        Current entropy coefficient (JAX scalar).
    """
    hold_end = total_steps * hold_frac
    anneal_end = hold_end + total_steps * anneal_frac

    progress = (step - hold_end) / jnp.maximum(anneal_end - hold_end, 1.0)
    progress = jnp.clip(progress, 0.0, 1.0)
    annealed = max_val + (min_val - max_val) * progress

    return jnp.where(step < hold_end, max_val, annealed)


def _collect_episodes(params, rng_key, *, model, n_parallel_envs, n_columns,
                      max_steps, threshold=250):
    """Collect trajectories for standalone A2C (always task_id=0, MaxScore)."""
    rng_key, init_rng = jax.random.split(rng_key)
    init_keys = jax.random.split(init_rng, n_parallel_envs)
    states, _ = jax.vmap(env_reset, in_axes=(0, None))(init_keys, n_columns)

    def scan_step(carry, _):
        states, rng = carry
        rng, act_rng, reset_rng = jax.random.split(rng, 3)

        obs = jax.vmap(make_obs, in_axes=(0, None))(states, n_columns)
        masks = jax.vmap(get_action_mask, in_axes=(0, None))(states, n_columns)
        phases = states.phase

        logits, values, _ = jax.vmap(model.apply, in_axes=(None, 0, 0))(
            params, obs, phases)
        logits = jnp.where(masks, logits, -jnp.inf)

        actions = jax.random.categorical(act_rng, logits)
        log_probs = jax.nn.log_softmax(logits)
        action_log_probs = jnp.take_along_axis(
            log_probs, actions[:, None], axis=1).squeeze(1)

        task_id = jnp.int32(0)
        new_states, _, rewards, dones, _ = jax.vmap(
            env_step, in_axes=(0, 0, None, None, None)
        )(states, actions, task_id, n_columns, threshold)

        reset_keys = jax.random.split(reset_rng, n_parallel_envs)
        reset_states, _ = jax.vmap(env_reset, in_axes=(0, None))(
            reset_keys, n_columns)
        next_states = jax.tree.map(
            lambda new, rst: jnp.where(
                jnp.broadcast_to(
                    dones.reshape(-1, *((1,) * (new.ndim - 1))), new.shape),
                rst, new),
            new_states, reset_states)

        step_data = (obs, actions, action_log_probs, rewards, dones,
                     values, phases, masks)
        return (next_states, rng), step_data

    _, trajectories = jax.lax.scan(
        scan_step, (states, rng_key), None, length=max_steps)
    return trajectories


def _prepare_data(trajectories, *, gamma, gae_lambda):
    """Flatten trajectories and compute TD(0) advantages."""
    obs, actions, log_probs, rewards, dones, values, phases, masks = trajectories
    max_steps, n_envs = rewards.shape

    batched_gae = jax.vmap(compute_gae, in_axes=(1, 1, 1, None, None), out_axes=1)
    advantages, returns = batched_gae(rewards, values, dones, gamma, gae_lambda)

    T = max_steps * n_envs
    obs_flat = obs.reshape(T, -1)
    actions_flat = actions.reshape(T)
    phases_flat = phases.reshape(T)
    masks_flat = masks.reshape(T, -1)
    log_probs_flat = log_probs.reshape(T)
    advantages_flat = advantages.reshape(T)
    returns_flat = returns.reshape(T)

    adv_mean = jnp.mean(advantages_flat)
    adv_std = jnp.std(advantages_flat) + 1e-8
    advantages_flat = (advantages_flat - adv_mean) / adv_std

    return (obs_flat, actions_flat, phases_flat, masks_flat,
            log_probs_flat, advantages_flat, returns_flat)


class A2CTrainer:
    def __init__(self, config):
        self.config = config
        self.model = ActorCritic(
            hidden_dim=config["agent"]["hidden_dim"],
            n_layers=config["agent"]["n_layers"],
            n_columns=config["env"]["n_columns"],
            use_layer_norm=config["agent"].get("use_layer_norm", False),
            activation=config["agent"].get("activation", "relu"),
            dropout_rate=config["agent"].get("dropout_rate", 0.0),
            head_hidden_dim=config["agent"].get("head_hidden_dim", 0),
            norm_position=config["agent"].get("norm_position", "pre"),
        )
        self.n_columns = config["env"]["n_columns"]
        self.max_steps = config["env"].get("max_steps_per_episode", 200)
        self.n_parallel_envs = config["training"]["n_parallel_envs"]
        self.n_games = config["training"]["n_games"]
        self.n_updates = self.n_games // self.n_parallel_envs
        self.threshold = config["tasks"]["threshold_beater_score"]

        a2c_cfg = config["a2c"]
        self.gamma = a2c_cfg.get("gamma", 0.99)
        self.gae_lambda = a2c_cfg.get("gae_lambda", 0.0)
        self.value_loss_coef = a2c_cfg.get("value_loss_coef", 0.005)

        self.entropy_roll_range = a2c_cfg.get("entropy_coef_roll", [0.01, 0.01])
        self.entropy_score_range = a2c_cfg.get("entropy_coef_score", [0.01, 0.01])
        self.entropy_hold = a2c_cfg.get("entropy_hold", 0.075)
        self.entropy_anneal_frac = a2c_cfg.get("entropy_anneal", 0.9)

        t_cfg = config["training"]
        lr_schedule = build_lr_schedule(
            peak_lr=t_cfg["lr"],
            min_ratio=t_cfg.get("lr_min_ratio", 0.05),
            warmup_frac=t_cfg.get("lr_warmup", 0.05),
            plateau_frac=t_cfg.get("lr_plateau", 0.70),
            total_steps=self.n_updates,
        )
        self.optimizer = optax.adam(learning_rate=lr_schedule)

        self.rng = jax.random.PRNGKey(config["env"]["seed"])
        self.rng, init_rng = jax.random.split(self.rng)
        od = obs_dim(self.n_columns)
        self.params = self.model.init(init_rng, jnp.zeros(od), jnp.int32(0))
        self.opt_state = self.optimizer.init(self.params)

        self.logger = create_logger(
            config["clearml"]["project_name"],
            config["clearml"]["task_name"],
            config)
        base_dir = t_cfg.get("checkpoint_dir", "checkpoints")
        if base_dir:
            self.checkpoint_dir = os.path.join(base_dir, self.logger.task_id)
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        else:
            self.checkpoint_dir = None
        self.checkpoint_every = t_cfg.get("checkpoint_every", 1000)
        self.eval_every = t_cfg.get("eval_every", 0)
        self.n_eval_episodes = t_cfg.get("n_eval_episodes", 1000)
        self.log_every = t_cfg.get("log_every", 50)

        self._collect = jax.jit(
            functools.partial(_collect_episodes,
                              model=self.model,
                              n_parallel_envs=self.n_parallel_envs,
                              n_columns=self.n_columns,
                              max_steps=self.max_steps,
                              threshold=self.threshold),
        )
        self._prepare = jax.jit(
            functools.partial(_prepare_data,
                              gamma=self.gamma,
                              gae_lambda=self.gae_lambda),
        )
        self._train_step = self._build_train_step()

    def _build_train_step(self):
        """Build a single JIT-compiled train step (collect + grad + update)."""
        model = self.model
        collect_fn = self._collect
        prepare_fn = self._prepare
        optimizer = self.optimizer
        value_loss_coef = self.value_loss_coef
        n_updates = self.n_updates
        ent_roll_max, ent_roll_min = self.entropy_roll_range
        ent_score_max, ent_score_min = self.entropy_score_range
        ent_hold = self.entropy_hold
        ent_anneal = self.entropy_anneal_frac
        has_dropout = model.dropout_rate > 0.0

        @jax.jit
        def train_step(params, opt_state, rng, step):
            rng, collect_rng, dropout_rng = jax.random.split(rng, 3)

            trajectories = collect_fn(params, collect_rng)
            data = prepare_fn(trajectories)
            obs, actions, phases, masks, log_probs, advantages, returns = data

            ent_roll = anneal_entropy(
                step, n_updates, ent_roll_max, ent_roll_min, ent_hold, ent_anneal)
            ent_score = anneal_entropy(
                step, n_updates, ent_score_max, ent_score_min, ent_hold, ent_anneal)

            def loss_fn(p):
                if has_dropout:
                    rngs = jax.random.split(dropout_rng, obs.shape[0])
                    def apply_dropout(o, ph, r):
                        return model.apply(p, o, ph, deterministic=False,
                                           rngs={"dropout": r})
                    logits, values, _ = jax.vmap(apply_dropout)(obs, phases, rngs)
                else:
                    logits, values, _ = jax.vmap(model.apply, in_axes=(None, 0, 0))(
                        p, obs, phases)
                return a2c_loss(
                    p, model, obs, actions, phases, masks,
                    log_probs, advantages, returns,
                    value_loss_coef=value_loss_coef,
                    entropy_coef_roll=ent_roll,
                    entropy_coef_score=ent_score,
                )

            loss, grads = jax.value_and_grad(loss_fn)(params)

            grad_norm = jnp.sqrt(jax.tree_util.tree_reduce(
                lambda acc, g: acc + jnp.sum(g ** 2), grads, initializer=0.0))
            scale = jnp.minimum(1.0, 1.0 / (grad_norm + 1e-8))
            grads = jax.tree.map(lambda g: g * scale, grads)

            updates, opt_state_new = optimizer.update(grads, opt_state, params)
            params_new = optax.apply_updates(params, updates)

            return params_new, opt_state_new, rng, loss

        return train_step

    def train(self, start_step=0):
        """Run the full training loop. Returns list of losses."""
        losses = []
        pbar = tqdm(range(start_step, self.n_updates),
                    initial=start_step, total=self.n_updates)
        for step in pbar:
            step_jax = jnp.int32(step)
            self.params, self.opt_state, self.rng, loss = self._train_step(
                self.params, self.opt_state, self.rng, step_jax)
            loss_val = float(loss)
            losses.append(loss_val)
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})

            if self.log_every and step % self.log_every == 0:
                self.logger.log_scalar("train", "loss", float(loss), step)
                ent_roll = float(anneal_entropy(
                    step_jax, self.n_updates,
                    self.entropy_roll_range[0], self.entropy_roll_range[1],
                    self.entropy_hold, self.entropy_anneal_frac))
                ent_score = float(anneal_entropy(
                    step_jax, self.n_updates,
                    self.entropy_score_range[0], self.entropy_score_range[1],
                    self.entropy_hold, self.entropy_anneal_frac))
                self.logger.log_scalar("entropy", "roll_coef", ent_roll, step)
                self.logger.log_scalar("entropy", "score_coef", ent_score, step)

            if self.checkpoint_dir and self.checkpoint_every and \
               (step + 1) % self.checkpoint_every == 0:
                self._save_checkpoint(step + 1)

            if self.eval_every and (step + 1) % self.eval_every == 0:
                self._run_eval(step + 1)

        self.logger.flush_scalars()
        return losses

    def _save_checkpoint(self, step):
        params_np = jax.device_get(self.params)
        opt_state_np = jax.device_get(self.opt_state)
        path = os.path.join(self.checkpoint_dir, f"step_{step}")
        os.makedirs(path, exist_ok=True)
        flat_params = {}
        for k, v in jax.tree_util.tree_leaves_with_path(params_np):
            key = "/".join(str(p) for p in k)
            flat_params[key] = np.asarray(v)
        np.savez(os.path.join(path, "params.npz"), **flat_params)
        flat_opt = {}
        for k, v in jax.tree_util.tree_leaves_with_path(opt_state_np):
            key = "/".join(str(p) for p in k)
            flat_opt[key] = np.asarray(v)
        np.savez(os.path.join(path, "opt_state.npz"), **flat_opt)
        np.savez(os.path.join(path, "meta.npz"), meta_step=step)
        self.logger.log_artifact(f"checkpoint_step{step}", path)
        return path

    def _run_eval(self, step):
        from src.training.evaluator import _build_batched_eval
        batched_eval = _build_batched_eval(
            self.model, self.n_columns, self.max_steps, self.threshold)
        rng = jax.random.PRNGKey(step)
        ep_rngs = jax.random.split(rng, self.n_eval_episodes)
        _, ep_summary = batched_eval(self.params, ep_rngs)
        ep_np = jax.device_get(ep_summary)
        mean_score = float(np.mean(ep_np["final_score"]))
        self.logger.log_scalar("eval", "mean_final_score", mean_score, step)
        print(f"  Eval step {step}: mean_score={mean_score:.1f}")

    def load_checkpoint(self, path):
        """Load params and opt_state from a checkpoint directory. Returns step number."""
        meta_data = np.load(os.path.join(path, "meta.npz"), allow_pickle=True)
        step = int(meta_data["meta_step"])
        params_data = dict(np.load(os.path.join(path, "params.npz"), allow_pickle=True))
        flat_params = [jnp.array(params_data["/".join(str(p) for p in k)])
                       for k, _ in jax.tree_util.tree_leaves_with_path(self.params)]
        self.params = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(self.params), flat_params)
        opt_data = dict(np.load(os.path.join(path, "opt_state.npz"), allow_pickle=True))
        flat_opt = [jnp.array(opt_data["/".join(str(p) for p in k)])
                    for k, _ in jax.tree_util.tree_leaves_with_path(self.opt_state)]
        self.opt_state = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(self.opt_state), flat_opt)
        return step
