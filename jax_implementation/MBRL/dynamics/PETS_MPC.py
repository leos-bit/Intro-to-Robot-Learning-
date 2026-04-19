import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
import optax
from flax import serialization
from flax.training import train_state

import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from jax_implementation.env import _sync_viewer_data, default_config, newDrone
from jax_implementation.MBRL.dynamics.PETS_Pretrain import (
    ACTION_DIM,
    GRAD_CLIP_NORM,
    MAX_OBSTACLES,
    OBS_DIM,
    EnsembleDynamics,
    assert_finite_metrics,
    compute_normalization_stats,
    eval_step,
    load_arrays,
    save_params,
    split_env_indices,
    train_step,
)


DEFAULT_DATA_PATH = (
    "jax_implementation/MBRL/dyn_data/"
    "pets_pretrain_envB1024_T10000_pid_noisy_seed0.npz"
)
DEFAULT_CHECKPOINT_DIR = "jax_implementation/MBRL/checkpoints/pets_pretrain"
DEFAULT_SAVE_DIR = "jax_implementation/MBRL/checkpoints/pets_online"
DEFAULT_ONLINE_DATA_DIR = "jax_implementation/MBRL/dyn_data"
AGENT_POS_XY_SLICE = slice(0, 2)
AGENT_POS_Z_SLICE = slice(2, 3)
AGENT_VEL_SLICE = slice(7, 10)
TARGET_SLICE = slice(10, 13)
LIDAR_SLICE = slice(17, 35)
OBSTACLE_REL_SLICE = slice(35, 80)
OBSTACLE_MASK_SLICE = slice(80, 95)
HORIZONTAL_LIDAR_INDICES = np.asarray([0, 1, 2, 3, 6, 7, 8, 9], dtype=np.int32)


@dataclass
class ReplayFrame:
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    mocap_pos: np.ndarray | None
    mocap_quat: np.ndarray | None


@dataclass
class EpisodeReplay:
    episode_id: int
    frames: list[ReplayFrame]


def _parse_override(raw_value: str) -> Any:
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return raw_value


def _apply_env_overrides(cfg, override_items: list[str]) -> dict[str, Any]:
    applied: dict[str, Any] = {}
    for item in override_items:
        if "=" not in item:
            raise ValueError(f"Invalid --env_override '{item}'. Expected KEY=VALUE.")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --env_override '{item}'. Empty key.")
        value = _parse_override(raw_value.strip())
        cfg[key] = value
        applied[key] = value
    return applied


def flatten_obs(obs: dict[str, jax.Array], obs_keys: tuple[str, ...]) -> jax.Array:
    flat_parts = [
        jnp.asarray(obs[key], dtype=jnp.float32).reshape((-1,))
        for key in obs_keys
    ]
    return jnp.concatenate(flat_parts, axis=0)


def build_flat_obs_bounds(env: newDrone, obs_keys: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    low_parts = []
    high_parts = []
    for key in obs_keys:
        spec = env.obs_spec[key]
        low_parts.append(np.asarray(spec["low"], dtype=np.float32).reshape(-1))
        high_parts.append(np.asarray(spec["high"], dtype=np.float32).reshape(-1))
    low = np.concatenate(low_parts, axis=0).astype(np.float32, copy=False)
    high = np.concatenate(high_parts, axis=0).astype(np.float32, copy=False)
    return low, high


def load_norm_stats(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {
        "x_mean": np.asarray(data["x_mean"], dtype=np.float32),
        "x_std": np.asarray(data["x_std"], dtype=np.float32),
        "y_mean": np.asarray(data["y_mean"], dtype=np.float32),
        "y_std": np.asarray(data["y_std"], dtype=np.float32),
    }


def flatten_transition_block(
    arrays: dict[str, np.ndarray],
    env_idx: np.ndarray,
) -> dict[str, np.ndarray]:
    obs = np.asarray(arrays["obs"][env_idx], dtype=np.float32).reshape(-1, OBS_DIM)
    action = np.asarray(arrays["applied_action"][env_idx], dtype=np.float32).reshape(-1, ACTION_DIM)
    next_obs = np.asarray(arrays["next_obs"][env_idx], dtype=np.float32).reshape(-1, OBS_DIM)
    reward = np.asarray(arrays["reward"][env_idx], dtype=np.float32).reshape(-1)
    return {
        "obs": obs,
        "action": action,
        "next_obs": next_obs,
        "reward": reward,
    }


def build_rollout_reward_state(info: dict[str, jax.Array]) -> dict[str, jax.Array]:
    return {
        "prev_action": jnp.asarray(info["prev_action"], dtype=jnp.float32).reshape((ACTION_DIM,)),
        "prev_distance": jnp.asarray(info["prev_distance"], dtype=jnp.float32),
        "min_distance_to_goal": jnp.asarray(info["min_distance_to_goal"], dtype=jnp.float32),
        "collision_streak": jnp.asarray(info["collision_streak"], dtype=jnp.int32),
        "goal_hold_streak": jnp.asarray(info["goal_hold_streak"], dtype=jnp.int32),
        "prev_obstacle_risk": jnp.asarray(info["prev_obstacle_risk"], dtype=jnp.float32),
        "prev_goal_path_risk": jnp.asarray(info["prev_goal_path_risk"], dtype=jnp.float32),
        "step": jnp.asarray(info["step"], dtype=jnp.int32),
    }


def build_reward_params(env: newDrone) -> dict[str, Any]:
    return {
        "horizontal_lidar_indices": np.asarray(
            jax.device_get(env._horizontal_lidar_indices),
            dtype=np.int32,
        ),
        "lidar_max_dist": float(env.lidar_max_dist),
        "lidar_softmin_tau": float(env.lidar_softmin_tau),
        "lidar_warn_dist": float(env.lidar_warn_dist),
        "obstacle_safe_dist": float(env.obstacle_safe_dist),
        "lidar_risk_weight": float(env.lidar_risk_weight),
        "true_obstacle_risk_weight": float(env.true_obstacle_risk_weight),
        "drone_clearance_radius": float(env.drone_clearance_radius),
        "obstacle_radius": float(env.obstacle_radius),
        "goal_path_clearance_margin": float(env.goal_path_clearance_margin),
        "w_progress": float(env.w_progress),
        "w_goal_path_clear": float(env.w_goal_path_clear),
        "w_obs": float(env.w_obs),
        "r_collision": float(env.r_collision),
        "w_energy": float(env.w_energy),
        "w_smooth": float(env.w_smooth),
        "w_goal_best_progress": float(env.w_goal_best_progress),
        "goal_proximity_scale": float(env.goal_proximity_scale),
        "w_goal_regress": float(env.w_goal_regress),
        "w_goal_proximity": float(env.w_goal_proximity),
        "safety_xy_scale": float(env.safety_xy_scale),
        "xylim": float(env.xylim),
        "safety_z_low": float(env.safety_z_low),
        "safety_z_high_scale": float(env.safety_z_high_scale),
        "zlim": float(env.zlim),
        "safety_speed_scale": float(env.safety_speed_scale),
        "vellim": float(env.vellim),
        "w_speed": float(env.w_speed),
        "eps_goal": float(env.eps_goal),
        "hover_speed_epsilon": float(env.hover_speed_epsilon),
        "w_goal_hover": float(env.w_goal_hover),
        "hover_success_steps": int(env.hover_success_steps),
        "r_goal": float(env.r_goal),
        "terminate_on_collision": bool(env.terminate_on_collision),
        "collision_terminate_steps": int(env.collision_terminate_steps),
        "max_steps": int(env.max_steps),
        "termination_penalty": float(env.termination_penalty),
        "terminal_distance_penalty": float(env.terminal_distance_penalty),
    }


def _softmin(x: jax.Array, temperature: jax.Array) -> jax.Array:
    x = jnp.asarray(x, dtype=jnp.float32)
    tau = jnp.maximum(jnp.asarray(temperature, dtype=jnp.float32), 1e-3)
    return -tau * jax.nn.logsumexp(-x / tau, axis=-1)


def _split_rollout_obs(obs: jax.Array) -> dict[str, jax.Array]:
    agent_location = jnp.concatenate(
        [obs[..., AGENT_POS_XY_SLICE], obs[..., AGENT_POS_Z_SLICE]],
        axis=-1,
    )
    return {
        "agent_location": agent_location,
        "agent_vel": obs[..., AGENT_VEL_SLICE],
        "target": obs[..., TARGET_SLICE],
        "lidar": obs[..., LIDAR_SLICE],
        "obstacle_rel": obs[..., OBSTACLE_REL_SLICE].reshape(*obs.shape[:-1], MAX_OBSTACLES, 3),
        "obstacle_mask": obs[..., OBSTACLE_MASK_SLICE],
    }


def _rollout_obstacle_terms(
    lidar: jax.Array,
    obstacle_rel: jax.Array,
    obstacle_mask: jax.Array,
    reward_params: dict[str, jax.Array],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    horizontal_lidar = jnp.asarray(lidar, dtype=jnp.float32)[..., reward_params["horizontal_lidar_indices"]]
    lidar_clearance = _softmin(horizontal_lidar, reward_params["lidar_softmin_tau"])
    obstacle_mask_bool = jnp.asarray(obstacle_mask > 0.0)
    obstacle_xy_dist = jnp.sqrt(jnp.sum(jnp.square(obstacle_rel[..., :2]), axis=-1) + 1e-8)
    true_clearance_all = obstacle_xy_dist - (
        reward_params["drone_clearance_radius"] + reward_params["obstacle_radius"]
    )
    true_clearance_all = jnp.where(
        obstacle_mask_bool,
        true_clearance_all,
        jnp.full_like(true_clearance_all, jnp.inf),
    )
    true_clearance = jnp.where(
        jnp.any(obstacle_mask_bool, axis=-1),
        jnp.min(true_clearance_all, axis=-1),
        reward_params["lidar_max_dist"],
    )
    lidar_risk = reward_params["lidar_risk_weight"] * jnp.square(
        jnp.maximum(0.0, reward_params["lidar_warn_dist"] - lidar_clearance)
    )
    true_risk = reward_params["true_obstacle_risk_weight"] * jnp.square(
        jnp.maximum(0.0, reward_params["obstacle_safe_dist"] - true_clearance)
    )
    obstacle_risk = lidar_risk + true_risk
    return lidar_clearance, true_clearance, obstacle_risk


def _rollout_goal_path_risk(
    agent_location: jax.Array,
    target: jax.Array,
    obstacle_rel: jax.Array,
    obstacle_mask: jax.Array,
    reward_params: dict[str, jax.Array],
) -> jax.Array:
    goal_vec_xy = target[..., :2] - agent_location[..., :2]
    goal_dist_xy = jnp.sqrt(jnp.sum(jnp.square(goal_vec_xy), axis=-1) + 1e-8)
    goal_dir_xy = goal_vec_xy / jnp.maximum(goal_dist_xy[..., None], 1e-6)
    rel_xy = obstacle_rel[..., :2]
    along = jnp.sum(rel_xy * goal_dir_xy[..., None, :], axis=-1)
    lateral_vec = rel_xy - (along[..., None] * goal_dir_xy[..., None, :])
    lateral = jnp.sqrt(jnp.sum(jnp.square(lateral_vec), axis=-1) + 1e-8)
    corridor_radius = (
        reward_params["drone_clearance_radius"]
        + reward_params["obstacle_radius"]
        + reward_params["goal_path_clearance_margin"]
    )
    between = (obstacle_mask > 0.0) & (along > 0.0) & (along < goal_dist_xy[..., None])
    shortfall = jnp.where(between, jnp.maximum(0.0, corridor_radius - lateral), 0.0)
    return jnp.max(shortfall, axis=-1)


class OnlineReplayBuffer:
    def __init__(
        self,
        base_obs: np.ndarray,
        base_action: np.ndarray,
        base_next_obs: np.ndarray,
        base_reward: np.ndarray,
    ) -> None:
        self.base = {
            "obs": np.asarray(base_obs, dtype=np.float32),
            "action": np.asarray(base_action, dtype=np.float32),
            "next_obs": np.asarray(base_next_obs, dtype=np.float32),
            "reward": np.asarray(base_reward, dtype=np.float32).reshape(-1),
        }
        self.base_size = int(self.base["obs"].shape[0])
        self._online_obs_parts: list[np.ndarray] = []
        self._online_action_parts: list[np.ndarray] = []
        self._online_next_obs_parts: list[np.ndarray] = []
        self._online_reward_parts: list[np.ndarray] = []
        self._online_cache: dict[str, np.ndarray] | None = None

    @property
    def online_size(self) -> int:
        if self._online_cache is not None:
            return int(self._online_cache["reward"].shape[0])
        return int(sum(part.shape[0] for part in self._online_obs_parts))

    @property
    def total_size(self) -> int:
        return self.base_size + self.online_size

    def add_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        reward: float,
    ) -> None:
        self._online_obs_parts.append(np.asarray(obs, dtype=np.float32).reshape(1, OBS_DIM))
        self._online_action_parts.append(np.asarray(action, dtype=np.float32).reshape(1, ACTION_DIM))
        self._online_next_obs_parts.append(
            np.asarray(next_obs, dtype=np.float32).reshape(1, OBS_DIM)
        )
        self._online_reward_parts.append(np.asarray([reward], dtype=np.float32))
        self._online_cache = None

    def _build_online_cache(self) -> dict[str, np.ndarray]:
        if self._online_cache is not None:
            return self._online_cache
        if not self._online_obs_parts:
            self._online_cache = {
                "obs": np.zeros((0, OBS_DIM), dtype=np.float32),
                "action": np.zeros((0, ACTION_DIM), dtype=np.float32),
                "next_obs": np.zeros((0, OBS_DIM), dtype=np.float32),
                "reward": np.zeros((0,), dtype=np.float32),
            }
            return self._online_cache
        self._online_cache = {
            "obs": np.concatenate(self._online_obs_parts, axis=0).astype(np.float32, copy=False),
            "action": np.concatenate(self._online_action_parts, axis=0).astype(
                np.float32, copy=False
            ),
            "next_obs": np.concatenate(self._online_next_obs_parts, axis=0).astype(
                np.float32, copy=False
            ),
            "reward": np.concatenate(self._online_reward_parts, axis=0).astype(
                np.float32, copy=False
            ),
        }
        return self._online_cache

    def sample_transitions(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        out = {
            "obs": np.empty((idx.shape[0], OBS_DIM), dtype=np.float32),
            "action": np.empty((idx.shape[0], ACTION_DIM), dtype=np.float32),
            "next_obs": np.empty((idx.shape[0], OBS_DIM), dtype=np.float32),
            "reward": np.empty((idx.shape[0],), dtype=np.float32),
        }
        base_mask = idx < self.base_size
        if np.any(base_mask):
            base_idx = idx[base_mask]
            out["obs"][base_mask] = self.base["obs"][base_idx]
            out["action"][base_mask] = self.base["action"][base_idx]
            out["next_obs"][base_mask] = self.base["next_obs"][base_idx]
            out["reward"][base_mask] = self.base["reward"][base_idx]
        if np.any(~base_mask):
            online = self._build_online_cache()
            online_idx = idx[~base_mask] - self.base_size
            out["obs"][~base_mask] = online["obs"][online_idx]
            out["action"][~base_mask] = online["action"][online_idx]
            out["next_obs"][~base_mask] = online["next_obs"][online_idx]
            out["reward"][~base_mask] = online["reward"][online_idx]
        return out

    def sample_mixed_transitions(
        self,
        rng: np.random.Generator,
        batch_size: int,
        online_fraction: float,
    ) -> dict[str, np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not 0.0 <= online_fraction <= 1.0:
            raise ValueError("online_fraction must be in [0, 1].")

        use_online = self.online_size > 0 and online_fraction > 0.0
        if not use_online:
            idx = rng.integers(0, self.base_size, size=batch_size, dtype=np.int64)
            return self.sample_transitions(idx)

        n_online = int(round(batch_size * online_fraction))
        n_online = min(max(n_online, 1), batch_size)
        n_base = batch_size - n_online

        base_idx = (
            rng.integers(0, self.base_size, size=n_base, dtype=np.int64)
            if n_base > 0
            else np.zeros((0,), dtype=np.int64)
        )
        online_idx = self.base_size + rng.integers(
            0,
            self.online_size,
            size=n_online,
            dtype=np.int64,
        )
        idx = np.concatenate([base_idx, online_idx], axis=0)
        rng.shuffle(idx)
        return self.sample_transitions(idx)

    def save_online_only(self, path: Path, metadata: dict[str, Any]) -> Path:
        online = self._build_online_cache()
        np.savez(
            path,
            obs=online["obs"],
            action=online["action"],
            applied_action=online["action"],
            next_obs=online["next_obs"],
            reward=online["reward"],
            metadata_json=np.asarray(json.dumps(metadata), dtype=np.str_),
        )
        return path


class PETSEnsembleTrainer:
    def __init__(
        self,
        ensemble_size: int,
        lr: float,
        online_fraction: float,
        seed: int,
        norm_stats: dict[str, np.ndarray],
        init_params_path: Path | None,
    ) -> None:
        self.ensemble_size = int(ensemble_size)
        self.online_fraction = float(online_fraction)
        self.norm_stats = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in norm_stats.items()
        }
        if not 0.0 <= self.online_fraction <= 1.0:
            raise ValueError("online_fraction must be in [0, 1].")
        self.x_mean = self.norm_stats["x_mean"]
        self.x_std = self.norm_stats["x_std"]
        self.y_mean = self.norm_stats["y_mean"]
        self.y_std = self.norm_stats["y_std"]
        self.y_std_jax = jnp.asarray(self.y_std, dtype=jnp.float32)
        self.model = EnsembleDynamics(ensemble_size=self.ensemble_size)
        dummy_x = jnp.zeros((self.ensemble_size, 1, OBS_DIM + ACTION_DIM), dtype=jnp.float32)
        params_template = self.model.init(jax.random.PRNGKey(seed), dummy_x)["params"]
        if init_params_path is not None:
            params = serialization.from_bytes(params_template, init_params_path.read_bytes())
        else:
            params = params_template
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=optax.chain(
                optax.clip_by_global_norm(GRAD_CLIP_NORM),
                optax.adam(lr),
            ),
        )
        self.rng = np.random.default_rng(seed)

    def _build_xy(self, transitions: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        obs = np.asarray(transitions["obs"], dtype=np.float32)
        action = np.asarray(transitions["action"], dtype=np.float32)
        next_obs = np.asarray(transitions["next_obs"], dtype=np.float32)
        reward = np.asarray(transitions["reward"], dtype=np.float32).reshape(-1, 1)
        x = np.concatenate([obs, action], axis=-1)
        y = np.concatenate([next_obs - obs, reward], axis=-1)
        x = (x - self.x_mean) / self.x_std
        y = (y - self.y_mean) / self.y_std
        return x.astype(np.float32, copy=False), y.astype(np.float32, copy=False)

    def train_updates(
        self,
        replay: OnlineReplayBuffer,
        batch_size: int,
        num_updates: int,
    ) -> dict[str, float]:
        if replay.total_size <= 0:
            raise ValueError("Replay buffer is empty.")
        if batch_size <= 0 or num_updates <= 0:
            raise ValueError("batch_size and num_updates must be positive.")

        stats = {
            "loss": 0.0,
            "mse": 0.0,
            "raw_mse": 0.0,
            "raw_delta_mse": 0.0,
            "raw_reward_mse": 0.0,
        }

        for step_id in range(num_updates):
            member_batches = []
            for _ in range(self.ensemble_size):
                transitions = replay.sample_mixed_transitions(
                    rng=self.rng,
                    batch_size=batch_size,
                    online_fraction=self.online_fraction,
                )
                member_batches.append(self._build_xy(transitions))
            batch_x = jnp.asarray(
                np.stack([bx for bx, _ in member_batches], axis=0),
                dtype=jnp.float32,
            )
            batch_y = jnp.asarray(
                np.stack([by for _, by in member_batches], axis=0),
                dtype=jnp.float32,
            )
            self.state, metrics = train_step(self.state, batch_x, batch_y, self.y_std_jax)
            metrics = jax.device_get(metrics)
            assert_finite_metrics(metrics, f"online train step={step_id + 1}")
            stats["loss"] += float(metrics["loss"])
            stats["mse"] += float(metrics["mse"])
            stats["raw_mse"] += float(metrics["raw_mse"])
            stats["raw_delta_mse"] += float(metrics["raw_delta_mse"])
            stats["raw_reward_mse"] += float(metrics["raw_reward_mse"])

        denom = float(max(num_updates, 1))
        return {key: value / denom for key, value in stats.items()}

    def evaluate(
        self,
        val_transitions: dict[str, np.ndarray] | None,
        batch_size: int,
        max_batches: int,
    ) -> dict[str, float] | None:
        if val_transitions is None:
            return None
        total = int(val_transitions["reward"].shape[0])
        if total <= 0:
            return None
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        stats = {
            "loss": 0.0,
            "mse": 0.0,
            "raw_mse": 0.0,
            "raw_delta_mse": 0.0,
            "raw_reward_mse": 0.0,
        }
        num_batches = 0
        for start in range(0, total, batch_size):
            stop = min(start + batch_size, total)
            batch = {
                key: np.asarray(value[start:stop])
                for key, value in val_transitions.items()
            }
            bx, by = self._build_xy(batch)
            bx_j = jnp.asarray(bx, dtype=jnp.float32)
            by_j = jnp.asarray(by, dtype=jnp.float32)
            bx_j = jnp.broadcast_to(bx_j[None], (self.ensemble_size, *bx_j.shape))
            by_j = jnp.broadcast_to(by_j[None], (self.ensemble_size, *by_j.shape))
            metrics = eval_step(self.state.params, self.state.apply_fn, bx_j, by_j, self.y_std_jax)
            metrics = jax.device_get(metrics)
            assert_finite_metrics(metrics, f"online val batch={num_batches + 1}")
            stats["loss"] += float(metrics["loss"])
            stats["mse"] += float(metrics["mse"])
            stats["raw_mse"] += float(metrics["raw_mse"])
            stats["raw_delta_mse"] += float(metrics["raw_delta_mse"])
            stats["raw_reward_mse"] += float(metrics["raw_reward_mse"])
            num_batches += 1
            if max_batches > 0 and num_batches >= max_batches:
                break

        denom = float(max(num_batches, 1))
        return {key: value / denom for key, value in stats.items()}


def _analytic_rollout_reward(
    next_obs: jax.Array,
    action: jax.Array,
    prev_action: jax.Array,
    prev_distance: jax.Array,
    min_distance_to_goal: jax.Array,
    collision_streak: jax.Array,
    goal_hold_streak: jax.Array,
    prev_obstacle_risk: jax.Array,
    prev_goal_path_risk: jax.Array,
    step: jax.Array,
    reward_params: dict[str, jax.Array],
) -> tuple[jax.Array, dict[str, jax.Array]]:
    obs_parts = _split_rollout_obs(next_obs)
    dist = jnp.sqrt(
        jnp.sum(jnp.square(obs_parts["target"] - obs_parts["agent_location"]), axis=-1) + 1e-8
    )
    delta_action = action - prev_action
    _, true_clearance, obstacle_risk = _rollout_obstacle_terms(
        obs_parts["lidar"],
        obs_parts["obstacle_rel"],
        obs_parts["obstacle_mask"],
        reward_params,
    )
    goal_path_risk = _rollout_goal_path_risk(
        obs_parts["agent_location"],
        obs_parts["target"],
        obs_parts["obstacle_rel"],
        obs_parts["obstacle_mask"],
        reward_params,
    )

    collision_contact = true_clearance <= 0.0
    next_collision_streak = jnp.where(
        collision_contact,
        collision_streak + jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    collision = next_collision_streak >= reward_params["collision_terminate_steps"]

    r_prog = reward_params["w_progress"] * (prev_distance - dist)
    r_obs = reward_params["w_obs"] * (prev_obstacle_risk - obstacle_risk)
    r_goal_path = reward_params["w_goal_path_clear"] * (prev_goal_path_risk - goal_path_risk)
    r_coll = jnp.where(collision, -reward_params["r_collision"], 0.0)
    r_energy = -reward_params["w_energy"] * jnp.sum(jnp.square(action), axis=-1)
    r_smooth = -reward_params["w_smooth"] * jnp.sum(jnp.square(delta_action), axis=-1)

    next_min_distance = jnp.minimum(min_distance_to_goal, dist)
    r_goal_best = reward_params["w_goal_best_progress"] * (
        min_distance_to_goal - next_min_distance
    )
    min_goal_near_frac = jnp.clip(
        1.0 - (min_distance_to_goal / reward_params["goal_proximity_scale"]),
        0.0,
        1.0,
    )
    goal_regress = jnp.maximum(0.0, dist - min_distance_to_goal)
    r_goal_regress = -reward_params["w_goal_regress"] * min_goal_near_frac * goal_regress
    prev_goal_near_frac = jnp.clip(
        1.0 - (prev_distance / reward_params["goal_proximity_scale"]),
        0.0,
        1.0,
    )
    goal_near_frac = jnp.clip(
        1.0 - (dist / reward_params["goal_proximity_scale"]),
        0.0,
        1.0,
    )
    r_goal_prox = reward_params["w_goal_proximity"] * (
        jnp.square(goal_near_frac) - jnp.square(prev_goal_near_frac)
    )

    pos = obs_parts["agent_location"]
    out_of_bounds = (
        (jnp.abs(pos[..., 0]) > (reward_params["safety_xy_scale"] * reward_params["xylim"]))
        | (jnp.abs(pos[..., 1]) > (reward_params["safety_xy_scale"] * reward_params["xylim"]))
        | (pos[..., 2] < reward_params["safety_z_low"])
        | (pos[..., 2] > (reward_params["safety_z_high_scale"] * reward_params["zlim"]))
    )
    speed_sq = jnp.sum(jnp.square(obs_parts["agent_vel"]), axis=-1)
    excessive_speed = speed_sq > jnp.square(
        reward_params["safety_speed_scale"] * reward_params["vellim"]
    )
    safety_terminated = out_of_bounds | excessive_speed
    r_safety = jnp.where(safety_terminated, -reward_params["r_collision"], 0.0)
    r_speed = -reward_params["w_speed"] * speed_sq

    hovering_at_goal = (dist <= reward_params["eps_goal"]) & (
        speed_sq <= jnp.square(reward_params["hover_speed_epsilon"])
    )
    next_goal_hold_streak = jnp.where(
        hovering_at_goal,
        goal_hold_streak + jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    success = next_goal_hold_streak >= reward_params["hover_success_steps"]

    reward = (
        r_prog
        + r_obs
        + r_goal_path
        + r_coll
        + r_energy
        + r_smooth
        + r_safety
        + r_speed
        + r_goal_prox
        + r_goal_best
        + r_goal_regress
        + (reward_params["w_goal_hover"] * jnp.where(hovering_at_goal, 1.0, 0.0))
        + jnp.where(success, reward_params["r_goal"], 0.0)
    )

    next_step = step + jnp.asarray(1, dtype=jnp.int32)
    terminated = success | (reward_params["terminate_on_collision"] & collision) | safety_terminated
    truncated = next_step >= reward_params["max_steps"]
    failure_episode_end = (terminated | truncated) & (~success)
    reward = reward + jnp.where(
        failure_episode_end,
        -(reward_params["termination_penalty"] + reward_params["terminal_distance_penalty"] * dist),
        0.0,
    )

    next_rollout_state = {
        "prev_action": action,
        "prev_distance": dist,
        "min_distance_to_goal": next_min_distance,
        "collision_streak": next_collision_streak,
        "goal_hold_streak": next_goal_hold_streak,
        "prev_obstacle_risk": obstacle_risk,
        "prev_goal_path_risk": goal_path_risk,
        "step": next_step,
        "done": terminated | truncated,
    }
    return reward, next_rollout_state


def build_rollout_eval_fn(
    apply_fn,
    ensemble_size: int,
    action_scale: float,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    obs_low: np.ndarray,
    obs_high: np.ndarray,
    reward_params: dict[str, Any],
    num_particles: int,
    gamma: float,
    deterministic_rollouts: bool,
):
    x_mean_j = jnp.asarray(x_mean, dtype=jnp.float32)
    x_std_j = jnp.asarray(x_std, dtype=jnp.float32)
    y_mean_j = jnp.asarray(y_mean, dtype=jnp.float32)
    y_std_j = jnp.asarray(y_std, dtype=jnp.float32)
    obs_low_j = jnp.asarray(obs_low, dtype=jnp.float32)
    obs_high_j = jnp.asarray(obs_high, dtype=jnp.float32)
    action_scale_j = jnp.asarray(float(action_scale), dtype=jnp.float32)
    gamma_j = jnp.asarray(float(gamma), dtype=jnp.float32)
    reward_params_j = {
        "horizontal_lidar_indices": jnp.asarray(
            reward_params["horizontal_lidar_indices"],
            dtype=jnp.int32,
        ),
        "lidar_max_dist": jnp.asarray(reward_params["lidar_max_dist"], dtype=jnp.float32),
        "lidar_softmin_tau": jnp.asarray(reward_params["lidar_softmin_tau"], dtype=jnp.float32),
        "lidar_warn_dist": jnp.asarray(reward_params["lidar_warn_dist"], dtype=jnp.float32),
        "obstacle_safe_dist": jnp.asarray(reward_params["obstacle_safe_dist"], dtype=jnp.float32),
        "lidar_risk_weight": jnp.asarray(reward_params["lidar_risk_weight"], dtype=jnp.float32),
        "true_obstacle_risk_weight": jnp.asarray(
            reward_params["true_obstacle_risk_weight"],
            dtype=jnp.float32,
        ),
        "drone_clearance_radius": jnp.asarray(
            reward_params["drone_clearance_radius"],
            dtype=jnp.float32,
        ),
        "obstacle_radius": jnp.asarray(reward_params["obstacle_radius"], dtype=jnp.float32),
        "goal_path_clearance_margin": jnp.asarray(
            reward_params["goal_path_clearance_margin"],
            dtype=jnp.float32,
        ),
        "w_progress": jnp.asarray(reward_params["w_progress"], dtype=jnp.float32),
        "w_goal_path_clear": jnp.asarray(reward_params["w_goal_path_clear"], dtype=jnp.float32),
        "w_obs": jnp.asarray(reward_params["w_obs"], dtype=jnp.float32),
        "r_collision": jnp.asarray(reward_params["r_collision"], dtype=jnp.float32),
        "w_energy": jnp.asarray(reward_params["w_energy"], dtype=jnp.float32),
        "w_smooth": jnp.asarray(reward_params["w_smooth"], dtype=jnp.float32),
        "w_goal_best_progress": jnp.asarray(
            reward_params["w_goal_best_progress"],
            dtype=jnp.float32,
        ),
        "goal_proximity_scale": jnp.asarray(
            reward_params["goal_proximity_scale"],
            dtype=jnp.float32,
        ),
        "w_goal_regress": jnp.asarray(reward_params["w_goal_regress"], dtype=jnp.float32),
        "w_goal_proximity": jnp.asarray(reward_params["w_goal_proximity"], dtype=jnp.float32),
        "safety_xy_scale": jnp.asarray(reward_params["safety_xy_scale"], dtype=jnp.float32),
        "xylim": jnp.asarray(reward_params["xylim"], dtype=jnp.float32),
        "safety_z_low": jnp.asarray(reward_params["safety_z_low"], dtype=jnp.float32),
        "safety_z_high_scale": jnp.asarray(
            reward_params["safety_z_high_scale"],
            dtype=jnp.float32,
        ),
        "zlim": jnp.asarray(reward_params["zlim"], dtype=jnp.float32),
        "safety_speed_scale": jnp.asarray(
            reward_params["safety_speed_scale"],
            dtype=jnp.float32,
        ),
        "vellim": jnp.asarray(reward_params["vellim"], dtype=jnp.float32),
        "w_speed": jnp.asarray(reward_params["w_speed"], dtype=jnp.float32),
        "eps_goal": jnp.asarray(reward_params["eps_goal"], dtype=jnp.float32),
        "hover_speed_epsilon": jnp.asarray(
            reward_params["hover_speed_epsilon"],
            dtype=jnp.float32,
        ),
        "w_goal_hover": jnp.asarray(reward_params["w_goal_hover"], dtype=jnp.float32),
        "hover_success_steps": jnp.asarray(
            reward_params["hover_success_steps"],
            dtype=jnp.int32,
        ),
        "r_goal": jnp.asarray(reward_params["r_goal"], dtype=jnp.float32),
        "terminate_on_collision": jnp.asarray(
            reward_params["terminate_on_collision"],
            dtype=jnp.bool_,
        ),
        "collision_terminate_steps": jnp.asarray(
            reward_params["collision_terminate_steps"],
            dtype=jnp.int32,
        ),
        "max_steps": jnp.asarray(reward_params["max_steps"], dtype=jnp.int32),
        "termination_penalty": jnp.asarray(
            reward_params["termination_penalty"],
            dtype=jnp.float32,
        ),
        "terminal_distance_penalty": jnp.asarray(
            reward_params["terminal_distance_penalty"],
            dtype=jnp.float32,
        ),
    }

    @jax.jit
    def _evaluate(
        params,
        obs0: jax.Array,
        rollout_state0: dict[str, jax.Array],
        action_sequences: jax.Array,
        rng: jax.Array,
    ) -> jax.Array:
        population = action_sequences.shape[0]
        horizon = action_sequences.shape[1]
        model_key, noise_master_key = jax.random.split(rng)
        obs_particles = jnp.broadcast_to(
            obs0[None, None, :],
            (population, num_particles, OBS_DIM),
        )
        prev_action = jnp.broadcast_to(
            rollout_state0["prev_action"][None, None, :],
            (population, num_particles, ACTION_DIM),
        )
        prev_distance = jnp.broadcast_to(
            rollout_state0["prev_distance"][None, None],
            (population, num_particles),
        )
        min_distance_to_goal = jnp.broadcast_to(
            rollout_state0["min_distance_to_goal"][None, None],
            (population, num_particles),
        )
        collision_streak = jnp.broadcast_to(
            rollout_state0["collision_streak"][None, None],
            (population, num_particles),
        )
        goal_hold_streak = jnp.broadcast_to(
            rollout_state0["goal_hold_streak"][None, None],
            (population, num_particles),
        )
        prev_obstacle_risk = jnp.broadcast_to(
            rollout_state0["prev_obstacle_risk"][None, None],
            (population, num_particles),
        )
        prev_goal_path_risk = jnp.broadcast_to(
            rollout_state0["prev_goal_path_risk"][None, None],
            (population, num_particles),
        )
        step_count = jnp.broadcast_to(
            rollout_state0["step"][None, None],
            (population, num_particles),
        )
        active = jnp.ones((population, num_particles), dtype=jnp.bool_)
        model_ids = jax.random.randint(
            model_key,
            shape=(population, num_particles),
            minval=0,
            maxval=ensemble_size,
            dtype=jnp.int32,
        )
        noise_keys = jax.random.split(noise_master_key, horizon)
        discounts = jnp.power(gamma_j, jnp.arange(horizon, dtype=jnp.float32))
        total_returns = jnp.zeros((population, num_particles), dtype=jnp.float32)

        def _step(carry, inputs):
            (
                current_obs,
                prev_action_c,
                prev_distance_c,
                min_distance_c,
                collision_streak_c,
                goal_hold_streak_c,
                prev_obstacle_risk_c,
                prev_goal_path_risk_c,
                step_count_c,
                active_c,
                returns,
            ) = carry
            action_t, noise_key_t, discount_t = inputs
            raw_action = jnp.broadcast_to(
                action_t[:, None, :],
                (population, num_particles, ACTION_DIM),
            )
            model_action = raw_action * action_scale_j
            obs_flat = current_obs.reshape(population * num_particles, OBS_DIM)
            act_flat = model_action.reshape(population * num_particles, ACTION_DIM)
            model_x = jnp.concatenate([obs_flat, act_flat], axis=-1)
            model_x = (model_x - x_mean_j) / x_std_j
            model_x = jnp.broadcast_to(
                model_x[None, :, :],
                (ensemble_size, model_x.shape[0], model_x.shape[-1]),
            )
            mean_norm, logvar_norm = apply_fn({"params": params}, model_x)
            flat_ids = model_ids.reshape(-1)
            sample_ids = jnp.arange(obs_flat.shape[0], dtype=jnp.int32)
            chosen_mean_norm = mean_norm[flat_ids, sample_ids]
            if deterministic_rollouts:
                chosen_target_norm = chosen_mean_norm
            else:
                chosen_logvar_norm = logvar_norm[flat_ids, sample_ids]
                chosen_std_norm = jnp.exp(0.5 * chosen_logvar_norm)
                noise = jax.random.normal(
                    noise_key_t,
                    shape=chosen_mean_norm.shape,
                    dtype=jnp.float32,
                )
                chosen_target_norm = chosen_mean_norm + (noise * chosen_std_norm)
            chosen_target = (chosen_target_norm * y_std_j) + y_mean_j
            delta = chosen_target[:, :OBS_DIM]
            next_obs_flat = jnp.clip(
                jnp.nan_to_num(obs_flat + delta, nan=0.0, posinf=0.0, neginf=0.0),
                obs_low_j,
                obs_high_j,
            )
            reward_flat, next_rollout_state = _analytic_rollout_reward(
                next_obs=next_obs_flat,
                action=act_flat,
                prev_action=prev_action_c.reshape(-1, ACTION_DIM),
                prev_distance=prev_distance_c.reshape(-1),
                min_distance_to_goal=min_distance_c.reshape(-1),
                collision_streak=collision_streak_c.reshape(-1),
                goal_hold_streak=goal_hold_streak_c.reshape(-1),
                prev_obstacle_risk=prev_obstacle_risk_c.reshape(-1),
                prev_goal_path_risk=prev_goal_path_risk_c.reshape(-1),
                step=step_count_c.reshape(-1),
                reward_params=reward_params_j,
            )

            active_flat = active_c.reshape(-1)
            reward_flat = jnp.where(active_flat, reward_flat, 0.0)
            next_done_flat = jnp.where(active_flat, next_rollout_state["done"], True)
            next_active = (~next_done_flat).reshape(population, num_particles)
            returns = returns + (
                discount_t * reward_flat.reshape(population, num_particles)
            )
            next_obs = next_obs_flat.reshape(population, num_particles, OBS_DIM)
            return (
                next_obs,
                next_rollout_state["prev_action"].reshape(population, num_particles, ACTION_DIM),
                next_rollout_state["prev_distance"].reshape(population, num_particles),
                next_rollout_state["min_distance_to_goal"].reshape(population, num_particles),
                next_rollout_state["collision_streak"].reshape(population, num_particles),
                next_rollout_state["goal_hold_streak"].reshape(population, num_particles),
                next_rollout_state["prev_obstacle_risk"].reshape(population, num_particles),
                next_rollout_state["prev_goal_path_risk"].reshape(population, num_particles),
                next_rollout_state["step"].reshape(population, num_particles),
                next_active,
                returns,
            ), None

        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            total_returns,
        ), _ = jax.lax.scan(
            _step,
            (
                obs_particles,
                prev_action,
                prev_distance,
                min_distance_to_goal,
                collision_streak,
                goal_hold_streak,
                prev_obstacle_risk,
                prev_goal_path_risk,
                step_count,
                active,
                total_returns,
            ),
            (
                action_sequences.swapaxes(0, 1),
                noise_keys,
                discounts,
            ),
        )
        return jnp.mean(total_returns, axis=1)

    return _evaluate


class PETSCEMPlanner:
    def __init__(
        self,
        params,
        apply_fn,
        norm_stats: dict[str, np.ndarray],
        reward_params: dict[str, Any],
        obs_low: np.ndarray,
        obs_high: np.ndarray,
        action_low: np.ndarray,
        action_high: np.ndarray,
        action_scale: float,
        ensemble_size: int,
        horizon: int,
        population_size: int,
        elite_size: int,
        iterations: int,
        num_particles: int,
        init_std: float,
        min_std: float,
        alpha: float,
        gamma: float,
        deterministic_rollouts: bool,
    ) -> None:
        self.params = params
        self.horizon = int(horizon)
        self.population_size = int(population_size)
        self.elite_size = int(elite_size)
        self.iterations = int(iterations)
        self.num_particles = int(num_particles)
        self.alpha = float(alpha)
        self.init_std = float(init_std)
        self.min_std = float(min_std)
        self.action_low = jnp.asarray(action_low, dtype=jnp.float32)
        self.action_high = jnp.asarray(action_high, dtype=jnp.float32)
        self.mean = jnp.zeros((self.horizon, ACTION_DIM), dtype=jnp.float32)
        self.std = jnp.full((self.horizon, ACTION_DIM), self.init_std, dtype=jnp.float32)
        self.evaluate_sequences = build_rollout_eval_fn(
            apply_fn=apply_fn,
            ensemble_size=ensemble_size,
            action_scale=action_scale,
            x_mean=norm_stats["x_mean"],
            x_std=norm_stats["x_std"],
            y_mean=norm_stats["y_mean"],
            y_std=norm_stats["y_std"],
            obs_low=obs_low,
            obs_high=obs_high,
            reward_params=reward_params,
            num_particles=self.num_particles,
            gamma=gamma,
            deterministic_rollouts=deterministic_rollouts,
        )

    def reset(self) -> None:
        self.mean = jnp.zeros((self.horizon, ACTION_DIM), dtype=jnp.float32)
        self.std = jnp.full((self.horizon, ACTION_DIM), self.init_std, dtype=jnp.float32)

    def set_params(self, params) -> None:
        self.params = params

    def plan(
        self,
        obs: jax.Array,
        rollout_state: dict[str, jax.Array],
        rng: jax.Array,
    ) -> tuple[jax.Array, dict[str, float], jax.Array]:
        mean = self.mean
        std = self.std
        best_score = -np.inf

        for _ in range(self.iterations):
            rng, sample_key, eval_key = jax.random.split(rng, 3)
            noise = jax.random.normal(
                sample_key,
                shape=(self.population_size, self.horizon, ACTION_DIM),
                dtype=jnp.float32,
            )
            action_sequences = mean[None, :, :] + (std[None, :, :] * noise)
            action_sequences = jnp.clip(
                action_sequences,
                self.action_low[None, None, :],
                self.action_high[None, None, :],
            )
            scores = self.evaluate_sequences(
                self.params,
                obs,
                rollout_state,
                action_sequences,
                eval_key,
            )
            action_sequences_np = np.asarray(jax.device_get(action_sequences), dtype=np.float32)
            scores_np = np.asarray(jax.device_get(scores), dtype=np.float32)
            elite_idx = np.argpartition(scores_np, -self.elite_size)[-self.elite_size :]
            elite_actions = action_sequences_np[elite_idx]
            elite_mean = np.mean(elite_actions, axis=0, dtype=np.float32)
            elite_std = np.std(elite_actions, axis=0, dtype=np.float32)
            mean = (self.alpha * mean) + ((1.0 - self.alpha) * jnp.asarray(elite_mean))
            std = (self.alpha * std) + ((1.0 - self.alpha) * jnp.asarray(elite_std))
            std = jnp.maximum(std, self.min_std)
            iter_best = float(scores_np[np.argmax(scores_np)])
            best_score = max(best_score, iter_best)

        action = jnp.asarray(mean[0], dtype=jnp.float32)
        shifted_mean = jnp.concatenate(
            [mean[1:], jnp.zeros((1, ACTION_DIM), dtype=jnp.float32)],
            axis=0,
        )
        shifted_std = jnp.concatenate(
            [
                std[1:],
                jnp.full((1, ACTION_DIM), self.init_std, dtype=jnp.float32),
            ],
            axis=0,
        )
        self.mean = shifted_mean
        self.std = shifted_std
        info = {
            "best_return": float(best_score),
            "mean_action_std": float(jnp.mean(std)),
        }
        return action, info, rng


class PETSVisualizer:
    def __init__(
        self,
        env: newDrone,
        render_live: bool,
        render_completed_run: bool,
        real_time: bool,
        print_every: int,
    ) -> None:
        self.env = env
        self.render_live = bool(render_live)
        self.render_completed_run = bool(render_completed_run)
        self.real_time = bool(real_time)
        self.print_every = int(print_every)
        self.viewer = None
        self.viewer_data = None
        self.ctrl_dt = float(getattr(env, "_ctrl_dt", 0.0))
        self.current_episode: EpisodeReplay | None = None
        self.completed_episodes: list[EpisodeReplay] = []

    def _capture_frame(self, state) -> ReplayFrame:
        data = state.data
        mocap_pos = None
        mocap_quat = None
        if self.env.mj_model.nmocap > 0:
            mocap_pos = np.asarray(jax.device_get(data.mocap_pos), dtype=np.float32).copy()
            mocap_quat = np.asarray(jax.device_get(data.mocap_quat), dtype=np.float32).copy()
        return ReplayFrame(
            qpos=np.asarray(jax.device_get(data.qpos), dtype=np.float32).copy(),
            qvel=np.asarray(jax.device_get(data.qvel), dtype=np.float32).copy(),
            ctrl=np.asarray(jax.device_get(data.ctrl), dtype=np.float32).copy(),
            mocap_pos=mocap_pos,
            mocap_quat=mocap_quat,
        )

    def _ensure_viewer(self) -> None:
        if self.viewer_data is None:
            self.viewer_data = mujoco.MjData(self.env.mj_model)
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.env.mj_model, self.viewer_data)
            if self.env._track_camera_id >= 0:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                self.viewer.cam.fixedcamid = self.env._track_camera_id

    def _load_frame(self, frame: ReplayFrame) -> None:
        self._ensure_viewer()
        self.viewer_data.qpos[:] = frame.qpos
        self.viewer_data.qvel[:] = frame.qvel
        self.viewer_data.ctrl[:] = frame.ctrl
        if self.env.mj_model.nmocap > 0 and frame.mocap_pos is not None and frame.mocap_quat is not None:
            self.viewer_data.mocap_pos[:] = frame.mocap_pos
            self.viewer_data.mocap_quat[:] = frame.mocap_quat
        mujoco.mj_forward(self.env.mj_model, self.viewer_data)

    def reset_episode(self, episode_id: int, state) -> bool:
        if self.render_completed_run:
            self.current_episode = EpisodeReplay(
                episode_id=int(episode_id),
                frames=[self._capture_frame(state)],
            )
        else:
            self.current_episode = None
        if not self.render_live:
            return True
        self._ensure_viewer()
        _sync_viewer_data(self.env, self.viewer_data, state)
        self.viewer.sync()
        return self.is_running()

    def sync(self, state) -> bool:
        if self.current_episode is not None:
            self.current_episode.frames.append(self._capture_frame(state))
        if not self.render_live or self.viewer is None or self.viewer_data is None:
            return True
        _sync_viewer_data(self.env, self.viewer_data, state)
        self.viewer.sync()
        return self.is_running()

    def is_running(self) -> bool:
        if self.viewer is None:
            return True
        if hasattr(self.viewer, "is_running"):
            return bool(self.viewer.is_running())
        return True

    def log_step(
        self,
        episode_id: int,
        step_id: int,
        state,
        episode_return: float,
        plan_info: dict[str, float],
        action: jax.Array,
    ) -> None:
        if self.print_every <= 0:
            return
        done = bool(jax.device_get(state.done))
        if (step_id % self.print_every) != 0 and not done:
            return
        info = state.info
        pos = np.asarray(jax.device_get(info["agent_location"]), dtype=np.float32)
        vel = np.asarray(jax.device_get(info["agent_vel"]), dtype=np.float32)
        dist = float(jax.device_get(info["distance"]))
        reward = float(jax.device_get(state.reward))
        action_np = np.asarray(jax.device_get(action), dtype=np.float32)
        print(
            f"ep={episode_id:02d} step={step_id:04d} "
            f"reward={reward: .3f} return={episode_return: .3f} "
            f"dist={dist: .3f} plan={plan_info['best_return']: .3f} "
            f"act={np.round(action_np, 3)} pos={np.round(pos, 3)} vel={np.round(vel, 3)}"
        )

    def maybe_sleep(self, step_start_time: float, *, during_rollout: bool) -> None:
        if not self.real_time or self.ctrl_dt <= 0.0:
            return
        if during_rollout and not self.render_live:
            return
        elapsed = time.perf_counter() - step_start_time
        time.sleep(max(self.ctrl_dt - elapsed, 0.0))

    def finish_episode(self) -> None:
        if self.current_episode is None:
            return
        self.completed_episodes.append(self.current_episode)
        self.current_episode = None

    def replay_completed_episodes(self) -> None:
        if not self.render_completed_run or not self.completed_episodes:
            return
        print(
            f"[replay] Rendering {len(self.completed_episodes)} completed episode(s) after rollout."
        )
        for replay_idx, episode in enumerate(self.completed_episodes, start=1):
            if not episode.frames:
                continue
            print(
                f"[replay {replay_idx}/{len(self.completed_episodes)}] "
                f"episode={episode.episode_id} frames={len(episode.frames)}"
            )
            for frame in episode.frames:
                if not self.is_running():
                    print("Viewer closed during replay.")
                    return
                frame_start_time = time.perf_counter()
                self._load_frame(frame)
                self.viewer.sync()
                self.maybe_sleep(frame_start_time, during_rollout=False)

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self.viewer_data = None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run PETS with MPC + CEM on the pretrained dynamics ensemble and periodic online retraining.",
    )
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--online_data_dir", type=str, default=DEFAULT_ONLINE_DATA_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ensemble_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument(
        "--online_batch_fraction",
        type=float,
        default=0.5,
        help="Fraction of each online update batch drawn from post-rollout data.",
    )
    parser.add_argument("--retrain_updates", type=int, default=250)
    parser.add_argument("--retrain_every", type=int, default=250)
    parser.add_argument("--warmup_online_steps", type=int, default=250)
    parser.add_argument("--val_batch_size", type=int, default=8192)
    parser.add_argument("--val_batches", type=int, default=32)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--plan_horizon", type=int, default=20)
    parser.add_argument("--cem_population", type=int, default=512)
    parser.add_argument("--cem_elites", type=int, default=64)
    parser.add_argument("--cem_iterations", type=int, default=6)
    parser.add_argument("--num_particles", type=int, default=20)
    parser.add_argument("--cem_alpha", type=float, default=0.1)
    parser.add_argument("--init_std", type=float, default=0.6)
    parser.add_argument("--min_std", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--deterministic_rollouts", action="store_true")
    parser.add_argument("--use_final_params", action="store_true")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--jit_step", action="store_true")
    parser.add_argument("--render", action="store_true", help="Open a live MuJoCo viewer.")
    parser.add_argument(
        "--render_completed_run",
        action="store_true",
        help="Record completed episodes and replay them after the run finishes.",
    )
    parser.add_argument(
        "--real_time",
        action="store_true",
        help="Sleep to roughly match ctrl_dt while rendering.",
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=25,
        help="Console logging cadence in env steps. Use 0 to disable.",
    )
    parser.add_argument(
        "--env_override",
        action="append",
        default=[],
        help="Override env config as KEY=VALUE. May be passed multiple times.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.cem_elites <= 0 or args.cem_elites > args.cem_population:
        raise ValueError("--cem_elites must be in [1, --cem_population].")
    if args.cem_population <= 0:
        raise ValueError("--cem_population must be positive.")
    if args.cem_iterations <= 0:
        raise ValueError("--cem_iterations must be positive.")
    if args.num_particles <= 0:
        raise ValueError("--num_particles must be positive.")
    if args.plan_horizon <= 0:
        raise ValueError("--plan_horizon must be positive.")
    if args.retrain_every <= 0:
        raise ValueError("--retrain_every must be positive.")
    if args.batch_size <= 0 or args.retrain_updates <= 0:
        raise ValueError("--batch_size and --retrain_updates must be positive.")
    if not 0.0 <= args.online_batch_fraction <= 1.0:
        raise ValueError("--online_batch_fraction must be in [0, 1].")
    if args.render and args.render_completed_run:
        raise ValueError("Use either --render or --render_completed_run, not both.")

    checkpoint_dir = Path(args.checkpoint_dir)
    save_dir = Path(args.save_dir)
    online_data_dir = Path(args.online_data_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    online_data_dir.mkdir(parents=True, exist_ok=True)

    cfg = default_config()
    env_overrides = _apply_env_overrides(cfg, args.env_override)
    env = newDrone(config=cfg)
    obs_keys = tuple(env.obs_spec.keys())
    obs_low, obs_high = build_flat_obs_bounds(env, obs_keys)
    visualizer = PETSVisualizer(
        env=env,
        render_live=args.render,
        render_completed_run=args.render_completed_run,
        real_time=args.real_time,
        print_every=args.print_every,
    )

    arrays = load_arrays(args.data_path)
    train_idx_path = checkpoint_dir / "train_idx.npy"
    test_idx_path = checkpoint_dir / "test_idx.npy"
    norm_stats_path = checkpoint_dir / "normalization_stats.npz"
    if train_idx_path.exists() and test_idx_path.exists():
        train_idx = np.load(train_idx_path)
        test_idx = np.load(test_idx_path)
    else:
        train_idx, test_idx = split_env_indices(
            num_envs=arrays["obs"].shape[0],
            train_ratio=args.train_ratio,
            seed=args.seed,
        )
    if norm_stats_path.exists():
        norm_stats = load_norm_stats(norm_stats_path)
    else:
        norm_stats = compute_normalization_stats(arrays, train_idx)

    train_transitions = flatten_transition_block(arrays, train_idx)
    val_transitions = flatten_transition_block(arrays, test_idx)
    replay = OnlineReplayBuffer(
        base_obs=train_transitions["obs"],
        base_action=train_transitions["action"],
        base_next_obs=train_transitions["next_obs"],
        base_reward=train_transitions["reward"],
    )

    init_params_name = "final_params.msgpack" if args.use_final_params else "best_params.msgpack"
    init_params_path = checkpoint_dir / init_params_name
    if not init_params_path.exists():
        raise FileNotFoundError(
            f"Missing pretrained PETS checkpoint: {init_params_path}. "
            "Run PETS_Pretrain.py first or point --checkpoint_dir to an existing run."
        )

    trainer = PETSEnsembleTrainer(
        ensemble_size=args.ensemble_size,
        lr=args.lr,
        online_fraction=args.online_batch_fraction,
        seed=args.seed,
        norm_stats=norm_stats,
        init_params_path=init_params_path,
    )
    planner = PETSCEMPlanner(
        params=trainer.state.params,
        apply_fn=trainer.state.apply_fn,
        norm_stats=norm_stats,
        reward_params=build_reward_params(env),
        obs_low=obs_low,
        obs_high=obs_high,
        action_low=np.asarray(env.action_low, dtype=np.float32),
        action_high=np.asarray(env.action_high, dtype=np.float32),
        action_scale=float(env.action_scale),
        ensemble_size=args.ensemble_size,
        horizon=args.plan_horizon,
        population_size=args.cem_population,
        elite_size=args.cem_elites,
        iterations=args.cem_iterations,
        num_particles=args.num_particles,
        init_std=args.init_std,
        min_std=args.min_std,
        alpha=args.cem_alpha,
        gamma=args.gamma,
        deterministic_rollouts=bool(args.deterministic_rollouts),
    )

    step_fn = jax.jit(env.step) if args.jit_step else env.step
    rng = jax.random.PRNGKey(args.seed)

    best_val_loss = float("inf")
    best_episode_return = -float("inf")
    global_step = 0
    last_retrain_step = 0
    retrain_history: list[dict[str, Any]] = []
    episode_history: list[dict[str, Any]] = []
    stop_requested = False
    try:
        if args.jit_step:
            rng, warm_reset_rng, warm_plan_rng = jax.random.split(rng, 3)
            warm_state = env.reset(warm_reset_rng)
            warm_obs = flatten_obs(warm_state.obs, obs_keys)
            warm_rollout_state = build_rollout_reward_state(warm_state.info)
            warm_action, _, warm_plan_rng = planner.plan(
                warm_obs,
                warm_rollout_state,
                warm_plan_rng,
            )
            compile_start = time.perf_counter()
            warm_state = step_fn(warm_state, warm_action)
            jax.block_until_ready(warm_state.reward)
            print(f"Compiled env.step in {time.perf_counter() - compile_start:.2f}s")

        for episode_id in range(args.num_episodes):
            planner.reset()
            rng, reset_rng, plan_rng = jax.random.split(rng, 3)
            state = env.reset(reset_rng)
            if not visualizer.reset_episode(episode_id + 1, state):
                print("Viewer closed, stopping PETS run.")
                stop_requested = True
                break
            episode_return = 0.0
            step_count = 0
            last_plan_info = {"best_return": 0.0, "mean_action_std": float(args.init_std)}
            episode_start_time = time.perf_counter()

            for step_id in range(args.max_steps):
                step_start_time = time.perf_counter()
                obs_flat = flatten_obs(state.obs, obs_keys)
                rollout_state = build_rollout_reward_state(state.info)
                action, plan_info, plan_rng = planner.plan(
                    obs_flat,
                    rollout_state,
                    plan_rng,
                )
                last_plan_info = plan_info
                next_state = step_fn(state, action)
                next_obs_flat = flatten_obs(next_state.obs, obs_keys)
                applied_action = np.asarray(
                    jax.device_get(next_state.info["held_action"]),
                    dtype=np.float32,
                ).reshape(ACTION_DIM)
                reward = float(jax.device_get(next_state.reward))
                replay.add_transition(
                    obs=np.asarray(jax.device_get(obs_flat), dtype=np.float32),
                    action=applied_action,
                    next_obs=np.asarray(jax.device_get(next_obs_flat), dtype=np.float32),
                    reward=reward,
                )
                episode_return += reward
                step_count = step_id + 1
                global_step += 1
                state = next_state

                if not visualizer.sync(state):
                    print("Viewer closed, stopping PETS run.")
                    stop_requested = True
                visualizer.log_step(
                    episode_id=episode_id + 1,
                    step_id=step_count,
                    state=state,
                    episode_return=episode_return,
                    plan_info=plan_info,
                    action=action,
                )

                should_retrain = (
                    replay.online_size >= args.warmup_online_steps
                    and (global_step % args.retrain_every == 0)
                )
                if should_retrain:
                    train_metrics = trainer.train_updates(
                        replay=replay,
                        batch_size=args.batch_size,
                        num_updates=args.retrain_updates,
                    )
                    val_metrics = trainer.evaluate(
                        val_transitions=val_transitions,
                        batch_size=args.val_batch_size,
                        max_batches=args.val_batches,
                    )
                    planner.set_params(trainer.state.params)
                    retrain_record = {
                        "global_step": global_step,
                        "online_transitions": replay.online_size,
                        "train_metrics": train_metrics,
                        "val_metrics": val_metrics,
                    }
                    retrain_history.append(retrain_record)
                    last_retrain_step = global_step
                    latest_params_path = save_dir / "latest_params.msgpack"
                    save_params(latest_params_path, trainer.state.params)
                    if val_metrics is not None and val_metrics["loss"] < best_val_loss:
                        best_val_loss = float(val_metrics["loss"])
                        save_params(save_dir / "best_online_params.msgpack", trainer.state.params)
                    print(
                        f"[retrain] step={global_step} online={replay.online_size} "
                        f"train_loss={train_metrics['loss']:.4f} "
                        f"val_loss={None if val_metrics is None else f'{val_metrics['loss']:.4f}'}"
                    )

                visualizer.maybe_sleep(step_start_time, during_rollout=True)
                if stop_requested or bool(jax.device_get(state.done)):
                    break

            episode_seconds = time.perf_counter() - episode_start_time
            best_episode_return = max(best_episode_return, episode_return)
            done_info = state.info
            episode_record = {
                "episode": episode_id + 1,
                "steps": step_count,
                "return": float(episode_return),
                "seconds": float(episode_seconds),
                "success": bool(jax.device_get(done_info["success"])),
                "terminated": bool(jax.device_get(done_info["terminated"])),
                "truncated": bool(jax.device_get(done_info["truncated"])),
                "final_distance": float(jax.device_get(done_info["distance"])),
                "last_plan_best_return": float(last_plan_info["best_return"]),
                "online_transitions": replay.online_size,
            }
            episode_history.append(episode_record)
            print(
                f"[episode {episode_id + 1}/{args.num_episodes}] "
                f"return={episode_return:.3f} steps={step_count} "
                f"success={episode_record['success']} "
                f"final_dist={episode_record['final_distance']:.3f}"
            )
            if not stop_requested:
                visualizer.finish_episode()
            if stop_requested:
                break
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        visualizer.close()

    has_untrained_online_data = (
        replay.online_size >= args.warmup_online_steps and global_step > last_retrain_step
    )
    if has_untrained_online_data:
        train_metrics = trainer.train_updates(
            replay=replay,
            batch_size=args.batch_size,
            num_updates=args.retrain_updates,
        )
        val_metrics = trainer.evaluate(
            val_transitions=val_transitions,
            batch_size=args.val_batch_size,
            max_batches=args.val_batches,
        )
        planner.set_params(trainer.state.params)
        retrain_record = {
            "global_step": global_step,
            "online_transitions": replay.online_size,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "final_sync": True,
        }
        retrain_history.append(retrain_record)
        if val_metrics is not None and val_metrics["loss"] < best_val_loss:
            best_val_loss = float(val_metrics["loss"])
            save_params(save_dir / "best_online_params.msgpack", trainer.state.params)
        print(
            f"[final retrain] step={global_step} online={replay.online_size} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={None if val_metrics is None else f'{val_metrics['loss']:.4f}'}"
        )

    save_params(save_dir / "final_online_params.msgpack", trainer.state.params)
    save_params(save_dir / "latest_params.msgpack", trainer.state.params)
    replay_meta = {
        "base_data_path": str(args.data_path),
        "checkpoint_dir": str(checkpoint_dir),
        "online_transitions": replay.online_size,
        "seed": int(args.seed),
        "env_overrides": env_overrides,
    }
    replay_path = online_data_dir / (
        f"pets_online_seed{args.seed}_eps{args.num_episodes}_steps{global_step}.npz"
    )
    replay.save_online_only(replay_path, replay_meta)

    metrics_payload = {
        "seed": int(args.seed),
        "data_path": str(args.data_path),
        "checkpoint_dir": str(checkpoint_dir),
        "save_dir": str(save_dir),
        "planner_reward_mode": "analytic_env_shaping",
        "online_batch_fraction": float(args.online_batch_fraction),
        "online_data_path": str(replay_path),
        "env_overrides": env_overrides,
        "global_steps": int(global_step),
        "online_transitions": int(replay.online_size),
        "best_val_loss": None if not np.isfinite(best_val_loss) else float(best_val_loss),
        "best_episode_return": (
            None if not np.isfinite(best_episode_return) else float(best_episode_return)
        ),
        "episode_history": episode_history,
        "retrain_history": retrain_history,
    }
    metrics_path = save_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8")
    norm_save_path = save_dir / "normalization_stats.npz"
    np.savez(norm_save_path, **norm_stats)

    print(f"Saved final params to {save_dir / 'final_online_params.msgpack'}")
    print(f"Saved online transitions to {replay_path}")
    print(f"Saved metrics to {metrics_path}")
    visualizer.replay_completed_episodes()


if __name__ == "__main__":
    main()
