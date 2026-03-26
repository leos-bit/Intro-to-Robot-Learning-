"""Train Drone_env (MJX) with Brax SAC and optional logging/checkpointing."""

from __future__ import annotations

import argparse
import datetime
import functools
import inspect
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac
import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco_playground import wrapper
from mujoco_playground._src import mjx_env
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from jax_implementation.env import default_config, newDrone
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

try:
    from jax_implementation.SAC.sac_config import (
        DEFAULT_LOG_ROOT,
        default_env_overrides,
        default_sac_overrides,
    )
except Exception:
    DEFAULT_LOG_ROOT = "jax_implementation/SAC/artifacts"

    def default_env_overrides() -> config_dict.ConfigDict:
        """Built-in environment overrides when sac_config.py is unavailable."""
        return config_dict.ConfigDict(
            dict(
                xylim=6.0,
                zlim=3.5,
                vellim=1.5,
                yawrate_lim=0.7,
                action_scale=1.0,
                spawn_z_min=0.8,
                target_dist_min=0.8,
                target_dist_max=3.5,
                collision_terminate_steps=12,
                eps_goal=0.35,
                safety_speed_scale=5.0,
                max_active_obstacles=15,
            )
        )

    def default_sac_overrides() -> config_dict.ConfigDict:
        """Built-in SAC defaults when sac_config.py is unavailable."""
        return config_dict.ConfigDict(
            dict(
                num_timesteps=60_000_000,
                num_evals=20,
                reward_scaling=1.0,
                episode_length=2_000,
                normalize_observations=True,
                deterministic_eval=True,
                action_repeat=1,
                discounting=0.99,
                learning_rate=3e-4,
                num_envs=1_024,
                num_eval_envs=128,
                batch_size=2_048,
                tau=0.005,
                min_replay_size=65_536,
                max_replay_size=1_000_000,
                grad_updates_per_step=1,
                run_evals=True,
                network_factory=dict(
                    hidden_layer_sizes=[1024, 1024],
                    policy_network_layer_norm=False,
                    q_network_layer_norm=False,
                ),
            )
        )


def _rscope_summary_fn(full_states, obs, rew, done):
    """Small summary callback used by rscope rollout saver."""
    del full_states, obs
    done_mask = jp.cumsum(done, axis=0)
    valid_rewards = rew * (done_mask == 0)
    ep_rewards = jp.sum(valid_rewards, axis=0)
    print(
        "Collected rscope rollouts with reward "
        f"{float(ep_rewards.mean()):.3f} +- {float(ep_rewards.std()):.3f}"
    )


def _canonicalize_model_assets(assets: dict[str, bytes]) -> dict[str, bytes]:
    """Deduplicate assets by basename for MuJoCo XML-from-string loading."""
    chosen: dict[str, tuple[str, bytes]] = {}
    for key, content in assets.items():
        base = Path(key).name.lower()
        prev = chosen.get(base)
        if prev is None:
            chosen[base] = (key, content)
            continue
        prev_key, _ = prev
        # Prefer relative-path keys over basename aliases.
        if ("/" in key) and ("/" not in prev_key):
            chosen[base] = (key, content)
    return {key: content for key, content in (chosen[b] for b in sorted(chosen))}


def _default_sac_cfg() -> config_dict.ConfigDict:
    """SAC config tuned for stable MJX drone training."""
    cfg = config_dict.ConfigDict(default_sac_overrides())
    cfg.network_factory = config_dict.ConfigDict(dict(cfg.network_factory))
    return cfg


class StateObsWrapper(wrapper.Wrapper):
    """Convert dict observations from Drone_env into a single flat vector."""

    def __init__(self, env: mjx_env.MjxEnv):
        super().__init__(env)
        self._model_assets = env.model_assets
        if hasattr(env, "obs_spec"):
            self._obs_keys = tuple(env.obs_spec.keys())
        else:
            # Fallback order if obs_spec is unavailable.
            self._obs_keys = (
                "agent_pos_xy",
                "agent_pos_z",
                "agent_orientation",
                "agent_vel",
                "target",
                "agent_yawrate",
                "goal_vec",
            )
        self._state_size = int(
            sum(
                int(np.prod(env.obs_spec[k]["shape"])) if hasattr(env, "obs_spec") else 0
                for k in self._obs_keys
            )
        )
        if self._state_size <= 0:
            # Generic fallback if obs_spec is unavailable.
            obs_size = env.observation_size
            if isinstance(obs_size, dict):
                self._state_size = int(
                    sum(
                        int(np.prod(v)) if isinstance(v, tuple) else int(v)
                        for v in obs_size.values()
                    )
                )
            else:
                self._state_size = (
                    int(np.prod(obs_size)) if isinstance(obs_size, tuple) else int(obs_size)
                )

    def _pack_obs(self, obs: dict[str, jax.Array]) -> jax.Array:
        # Flatten each observation block so mixed shapes like obstacle_rel [N, 3]
        # still pack into one SAC-ready state vector.
        return jp.concatenate(
            [jp.asarray(obs[k], dtype=jp.float32).reshape((-1,)) for k in self._obs_keys],
            axis=-1,
        )

    def reset(self, rng: jax.Array) -> mjx_env.State:
        state = self.env.reset(rng)
        return state.replace(obs=self._pack_obs(state.obs))

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        state = self.env.step(state, action)
        return state.replace(obs=self._pack_obs(state.obs))

    @property
    def observation_size(self):
        return self._state_size


def _build_parser() -> argparse.ArgumentParser:
    default_sac_cfg = _default_sac_cfg()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--suffix", type=str, default=None)
    parser.add_argument("--log_root", type=str, default=DEFAULT_LOG_ROOT)
    parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable TensorBoard logging for SAC runs.",
    )
    parser.add_argument(
        "--tensorboard_dir",
        type=str,
        default=None,
        help="Optional TensorBoard output directory (default: <run_logdir>/tensorboard).",
    )
    parser.add_argument("--load_checkpoint_path", type=str, default=None)
    parser.add_argument("--run_evals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress_bar", action=argparse.BooleanOptionalAction, default=True)
    # Training parameters
    parser.add_argument("--num_timesteps", type=int, default=int(default_sac_cfg.num_timesteps))
    parser.add_argument("--num_evals", type=int, default=int(default_sac_cfg.num_evals))
    parser.add_argument("--num_envs", type=int, default=int(default_sac_cfg.num_envs))
    parser.add_argument("--num_eval_envs", type=int, default=int(default_sac_cfg.num_eval_envs))
    parser.add_argument("--episode_length", type=int, default=int(default_sac_cfg.episode_length))
    parser.add_argument("--batch_size", type=int, default=int(default_sac_cfg.batch_size))
    parser.add_argument("--learning_rate", type=float, default=float(default_sac_cfg.learning_rate))
    parser.add_argument("--discounting", type=float, default=float(default_sac_cfg.discounting))
    parser.add_argument("--reward_scaling", type=float, default=float(default_sac_cfg.reward_scaling))
    parser.add_argument("--tau", type=float, default=float(default_sac_cfg.tau))
    parser.add_argument("--min_replay_size", type=int, default=int(default_sac_cfg.min_replay_size))
    parser.add_argument("--max_replay_size", type=int, default=int(default_sac_cfg.max_replay_size))
    parser.add_argument(
        "--grad_updates_per_step",
        type=int,
        default=int(default_sac_cfg.grad_updates_per_step),
    )
    parser.add_argument(
        "--normalize_observations",
        action=argparse.BooleanOptionalAction,
        default=bool(default_sac_cfg.normalize_observations),
    )
    parser.add_argument(
        "--deterministic_eval",
        action=argparse.BooleanOptionalAction,
        default=bool(default_sac_cfg.deterministic_eval),
    )
    parser.add_argument(
        "--hidden_layer_sizes",
        nargs="+",
        type=int,
        default=list(default_sac_cfg.network_factory.hidden_layer_sizes),
    )

    parser.add_argument("--impl", type=str, default="jax")
    parser.add_argument(
        "--env_overrides",
        type=str,
        default=None,
        help="JSON dict of env config overrides, e.g. '{\"action_scale\":0.35}'.",
    )

    parser.add_argument(
        "--rscope_envs",
        type=int,
        default=0,
        help="If >0, dump rollouts for rscope live viewer.",
    )
    parser.add_argument(
        "--deterministic_rscope",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _apply_env_overrides(env_cfg: config_dict.ConfigDict, overrides_json: str | None):
    if not overrides_json:
        return
    overrides = json.loads(overrides_json)
    if not isinstance(overrides, dict):
        raise ValueError("--env_overrides must decode to a JSON object/dict.")
    for key, value in overrides.items():
        env_cfg[key] = _coerce_env_override_value(env_cfg.get(key, None), value)


def _coerce_env_override_value(current_value: Any, value: Any) -> Any:
    """Coerce override values to the existing config field type when safe."""
    if isinstance(current_value, jax.Array):
        return jp.asarray(value, dtype=current_value.dtype)
    if isinstance(current_value, bool):
        return bool(value)
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            return int(value)
        return value
    if isinstance(current_value, float):
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            return float(value)
    return value


def _make_env_cfg(args: argparse.Namespace) -> config_dict.ConfigDict:
    env_cfg = default_config()
    env_cfg.impl = args.impl
    env_cfg.max_steps = args.episode_length
    env_cfg.episode_length = args.episode_length
    for key, value in default_env_overrides().items():
        env_cfg[key] = _coerce_env_override_value(env_cfg.get(key, None), value)
    _apply_env_overrides(env_cfg, args.env_overrides)
    return env_cfg


def _make_sac_cfg(args: argparse.Namespace) -> config_dict.ConfigDict:
    cfg = _default_sac_cfg()
    cfg.num_timesteps = args.num_timesteps
    cfg.num_evals = args.num_evals
    cfg.num_envs = args.num_envs
    cfg.num_eval_envs = args.num_eval_envs
    cfg.episode_length = args.episode_length
    cfg.batch_size = args.batch_size
    cfg.learning_rate = args.learning_rate
    cfg.discounting = args.discounting
    cfg.reward_scaling = args.reward_scaling
    cfg.tau = args.tau
    cfg.min_replay_size = args.min_replay_size
    cfg.max_replay_size = args.max_replay_size
    cfg.grad_updates_per_step = args.grad_updates_per_step
    cfg.normalize_observations = args.normalize_observations
    cfg.deterministic_eval = args.deterministic_eval
    cfg.run_evals = args.run_evals
    cfg.network_factory.hidden_layer_sizes = list(args.hidden_layer_sizes)
    return cfg


def _jsonable_config(value: Any) -> Any:
    """Recursively convert ConfigDict/JAX values into JSON-safe Python types."""
    if isinstance(value, config_dict.ConfigDict):
        return {k: _jsonable_config(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {str(k): _jsonable_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_config(v) for v in value]
    if isinstance(value, jax.Array):
        arr = np.asarray(value)
        if arr.ndim == 0:
            return arr.item()
        return arr.tolist()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _metrics_record(num_steps: int, metrics: dict[str, Any]) -> dict[str, Any]:
    """Convert callback metrics into a JSON-safe row for later plotting."""
    record: dict[str, Any] = {"num_steps": int(num_steps)}
    for key, value in metrics.items():
        scalar = _safe_float_scalar(value)
        record[key] = scalar if scalar is not None else _jsonable_config(value)
    return record


def _safe_float_scalar(value: Any) -> float | None:
    """Best-effort conversion to finite scalar float for TensorBoard."""
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        return None
    if isinstance(value, (int, float, np.number)):
        val = float(value)
        return val if np.isfinite(val) else None
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if arr.size != 1:
        return None
    try:
        scalar = arr.item()
    except Exception:
        return None
    if scalar is None:
        return None
    try:
        val = float(scalar)
    except (TypeError, ValueError):
        return None
    return val if np.isfinite(val) else None


def _validate_train_configs(
    env_cfg: config_dict.ConfigDict,
    sac_cfg: config_dict.ConfigDict,
):
    """Ensure env/train config agreement for rollout and eval."""
    mismatches: list[str] = []
    env_episode_length = int(env_cfg.episode_length)
    env_max_steps = int(env_cfg.max_steps)
    sac_episode_length = int(sac_cfg.episode_length)
    env_action_repeat = int(env_cfg.action_repeat)
    sac_action_repeat = int(sac_cfg.action_repeat)

    if env_episode_length != sac_episode_length:
        mismatches.append(
            f"env episode_length={env_episode_length} != sac episode_length={sac_episode_length}"
        )
    if env_max_steps != sac_episode_length:
        mismatches.append(
            f"env max_steps={env_max_steps} != sac episode_length={sac_episode_length}"
        )
    if env_action_repeat != sac_action_repeat:
        mismatches.append(
            f"env action_repeat={env_action_repeat} != sac action_repeat={sac_action_repeat}"
        )
    if mismatches:
        raise ValueError("Config mismatch detected:\n- " + "\n- ".join(mismatches))


def _checkpoint_rank(metrics: dict[str, Any]) -> tuple[float, float, float]:
    """Rank checkpoints by success, then final goal distance, then eval reward."""
    numerical_issue = _safe_float_scalar(metrics.get("eval/episode_numerical_issue"))
    if numerical_issue is not None and numerical_issue > 0.0:
        return (float("-inf"), float("-inf"), float("-inf"))
    success = _safe_float_scalar(metrics.get("eval/episode_success"))
    final_distance = _safe_float_scalar(metrics.get("eval/episode_final_distance_to_goal"))
    reward = _safe_float_scalar(metrics.get("eval/episode_reward"))
    return (
        float("-inf") if success is None else success,
        float("-inf") if final_distance is None else -final_distance,
        float("-inf") if reward is None else reward,
    )


def _save_best_checkpoint(
    ckpt_dir: Path,
    step: int,
    metrics: dict[str, Any],
    best_dir: Path,
    best_meta_path: Path,
):
    """Copies the current numbered checkpoint into a stable best-checkpoint folder."""
    src_dir = ckpt_dir / f"{step:012d}"
    if step <= 0 or not src_dir.exists():
        return False
    if best_dir.exists():
        shutil.rmtree(best_dir)
    shutil.copytree(src_dir, best_dir)
    best_meta = {
        "source_step": int(step),
        "source_checkpoint": str(src_dir),
        "eval_reward": _safe_float_scalar(metrics.get("eval/episode_reward")),
        "eval_success": _safe_float_scalar(metrics.get("eval/episode_success")),
        "eval_final_distance_to_goal": _safe_float_scalar(
            metrics.get("eval/episode_final_distance_to_goal")
        ),
        "eval_best_distance_to_goal": _safe_float_scalar(
            metrics.get("eval/episode_best_distance_to_goal")
        ),
        "eval_distance_to_goal_per_step": _safe_float_scalar(
            metrics.get("eval/episode_distance_to_goal_per_step")
        ),
        "eval_numerical_issue": _safe_float_scalar(
            metrics.get("eval/episode_numerical_issue")
        ),
    }
    best_meta_path.write_text(json.dumps(best_meta, indent=2), encoding="utf-8")
    return True


class TrainingLogger:
    """SB3-style training logger with progress bar and metrics tracking."""

    def __init__(
        self,
        total_timesteps: int,
        num_envs: int,
        episode_length: int,
        use_progress_bar: bool = True,
        log_interval: int = 1,
    ):
        self.total_timesteps = total_timesteps
        self.num_envs = num_envs
        self.episode_length = episode_length
        self.use_progress_bar = use_progress_bar and tqdm is not None
        self.log_interval = log_interval

        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.last_log_steps = 0
        self.eval_count = 0

        # Metrics history
        self.metrics_history: list[dict] = []

        # Progress bar
        self.pbar = None
        if self.use_progress_bar:
            self.pbar = tqdm(
                total=total_timesteps,
                desc="Training",
                unit="steps",
                dynamic_ncols=True,
                smoothing=0.1,
            )

    def _as_int(self, value: Any) -> int:
        scalar = _safe_float_scalar(value)
        if scalar is None:
            return int(value)
        return int(scalar)

    def _as_float(self, value: Any, default: float = 0.0) -> float:
        scalar = _safe_float_scalar(value)
        if scalar is None:
            return default
        return float(scalar)

    def log_eval(self, num_steps: int, metrics: dict[str, Any]):
        """Log evaluation metrics (SB3-style)."""
        self.eval_count += 1
        current_time = time.time()
        elapsed = current_time - self.start_time
        num_steps_int = self._as_int(num_steps)
        fps = num_steps_int / elapsed if elapsed > 0 else 0

        # Update progress bar
        if self.pbar is not None:
            self.pbar.n = num_steps_int
            self.pbar.refresh()

        # Extract key metrics (Brax uses different keys than SB3)
        eval_reward = self._as_float(metrics.get("eval/episode_reward", 0.0))
        eval_reward_std = self._as_float(metrics.get("eval/episode_reward_std", 0.0))
        eval_goal_dist = self._as_float(
            metrics.get("eval/episode_final_distance_to_goal", float("nan")),
            default=float("nan"),
        )
        eval_best_goal_dist = self._as_float(
            metrics.get("eval/episode_best_distance_to_goal", float("nan")),
            default=float("nan"),
        )
        # Brax uses avg_episode_length, not episode_length
        eval_length = self._as_float(
            metrics.get("eval/avg_episode_length", metrics.get("eval/episode_length", 0.0))
        )

        # Calculate FPS between logs
        steps_since_log = num_steps_int - self.last_log_steps
        time_since_log = current_time - self.last_log_time
        recent_fps = steps_since_log / time_since_log if time_since_log > 0 else fps

        # SB3-style output
        print("-" * 60)
        print(f"| {'rollout/':<20} |")
        print(f"|   {'ep_len_mean':<17} | {eval_length:<20.1f} |")
        print(f"|   {'ep_rew_mean':<17} | {eval_reward:<20.3f} |")
        print(f"|   {'ep_rew_std':<17} | {eval_reward_std:<20.3f} |")
        if not math.isnan(eval_goal_dist):
            print(f"|   {'goal_dist_final':<17} | {eval_goal_dist:<20.3f} |")
        if not math.isnan(eval_best_goal_dist):
            print(f"|   {'goal_dist_best':<17} | {eval_best_goal_dist:<20.3f} |")
        print(f"| {'time/':<20} |")
        print(f"|   {'fps':<17} | {recent_fps:<20.0f} |")
        print(f"|   {'iterations':<17} | {self.eval_count:<20d} |")
        print(f"|   {'time_elapsed':<17} | {elapsed:<20.0f} |")
        print(f"|   {'total_timesteps':<17} | {num_steps_int:<20d} |")

        # Log additional metrics
        if "training/actor_loss" in metrics:
            print(
                f"|   {'actor_loss':<17} | "
                f"{self._as_float(metrics['training/actor_loss']):<20.4f} |"
            )
        if "training/critic_loss" in metrics:
            print(
                f"|   {'critic_loss':<17} | "
                f"{self._as_float(metrics['training/critic_loss']):<20.4f} |"
            )
        if "training/alpha_loss" in metrics:
            print(
                f"|   {'alpha_loss':<17} | "
                f"{self._as_float(metrics['training/alpha_loss']):<20.4f} |"
            )
        if "training/alpha" in metrics:
            print(
                f"|   {'alpha':<17} | "
                f"{self._as_float(metrics['training/alpha']):<20.4f} |"
            )
        if "training/buffer_current_size" in metrics:
            print(
                f"|   {'buffer_size':<17} | "
                f"{self._as_float(metrics['training/buffer_current_size']):<20.0f} |"
            )

        print("-" * 60)

        # Update tracking
        self.last_log_time = current_time
        self.last_log_steps = num_steps_int
        logged_metrics = {}
        for key, value in metrics.items():
            scalar = _safe_float_scalar(value)
            logged_metrics[key] = scalar if scalar is not None else value
        self.metrics_history.append({"num_steps": num_steps_int, **logged_metrics})

    def log_train(self, num_steps: int, metrics: dict[str, Any]):
        """Log training metrics without full eval."""
        num_steps_int = self._as_int(num_steps)
        if self.pbar is not None:
            self.pbar.n = num_steps_int
            # Show training reward in progress bar if available
            train_reward = self._as_float(metrics.get("training/actor_loss"), default=float("nan"))
            if np.isfinite(train_reward):
                self.pbar.set_postfix({"actor_loss": f"{train_reward:.2f}"}, refresh=True)
            else:
                self.pbar.refresh()

    def close(self):
        """Close progress bar and print summary."""
        if self.pbar is not None:
            self.pbar.close()

        elapsed = time.time() - self.start_time
        print(f"\nTraining completed in {elapsed:.2f}s")
        print(f"Total timesteps: {self.total_timesteps}")
        if self.metrics_history:
            final_metrics = self.metrics_history[-1]
            if "eval/episode_reward" in final_metrics:
                print(f"Final eval reward: {final_metrics['eval/episode_reward']:.3f}")


def train(args: argparse.Namespace):
    env_cfg = _make_env_cfg(args)
    sac_cfg = _make_sac_cfg(args)
    _validate_train_configs(env_cfg, sac_cfg)
    env_cfg_json = _jsonable_config(env_cfg)
    sac_cfg_json = _jsonable_config(sac_cfg)

    env_base = newDrone(config=env_cfg)
    canonical_assets = _canonicalize_model_assets(env_base.model_assets)
    env_base._model_assets = canonical_assets
    env = StateObsWrapper(env_base)
    eval_env = None
    if sac_cfg.run_evals:
        eval_env_base = newDrone(config=env_cfg)
        eval_env_base._model_assets = canonical_assets
        eval_env = StateObsWrapper(eval_env_base)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"DroneMJX-{timestamp}"
    if args.suffix:
        exp_name += f"-{args.suffix}"
    logdir = Path(args.log_root).resolve() / exp_name
    ckpt_dir = logdir / "checkpoints"
    best_ckpt_dir = ckpt_dir / "best"
    best_ckpt_meta_path = ckpt_dir / "best_checkpoint.json"
    metrics_path = logdir / "metrics.jsonl"
    latest_metrics_path = logdir / "latest_metrics.json"
    tb_dir = Path(args.tensorboard_dir).resolve() if args.tensorboard_dir else (logdir / "tensorboard")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if args.tensorboard:
        tb_dir.mkdir(parents=True, exist_ok=True)

    with (logdir / "env_config.json").open("w", encoding="utf-8") as fp:
        json.dump(env_cfg_json, fp, indent=2)
    with (logdir / "sac_config.json").open("w", encoding="utf-8") as fp:
        json.dump(sac_cfg_json, fp, indent=2)

    # SB3-style header
    print("=" * 60)
    print(f"| Experiment: {exp_name}")
    print(f"| Log dir: {logdir}")
    print(f"| Checkpoint dir: {ckpt_dir}")
    if args.tensorboard:
        print(f"| TensorBoard dir: {tb_dir}")
    print("=" * 60)
    print("| SAC Hyperparameters:")
    print(f"|   num_timesteps: {sac_cfg.num_timesteps:,}")
    print(f"|   num_envs: {sac_cfg.num_envs}")
    print(f"|   episode_length: {sac_cfg.episode_length}")
    print(f"|   learning_rate: {sac_cfg.learning_rate}")
    print(f"|   discounting (gamma): {sac_cfg.discounting}")
    print(f"|   reward_scaling: {sac_cfg.reward_scaling}")
    print(f"|   tau: {sac_cfg.tau}")
    print(f"|   batch_size: {sac_cfg.batch_size}")
    print(f"|   min_replay_size: {sac_cfg.min_replay_size}")
    print(f"|   max_replay_size: {sac_cfg.max_replay_size}")
    print(f"|   grad_updates_per_step: {sac_cfg.grad_updates_per_step}")
    print(f"|   deterministic_eval: {sac_cfg.deterministic_eval}")
    print(f"|   hidden_layers: {sac_cfg.network_factory.hidden_layer_sizes}")
    print("=" * 60)
    print("| Environment Config:")
    print(f"|   action_scale: {env_cfg.action_scale}")
    print(f"|   target_dist_min: {env_cfg.target_dist_min}")
    print(f"|   target_dist_max: {env_cfg.target_dist_max}")
    print(f"|   eps_goal: {env_cfg.eps_goal}")
    print(f"|   r_goal: {env_cfg.r_goal}")
    print(f"|   w_progress: {env_cfg.w_progress}")
    print(f"|   w_goal_proximity: {env_cfg.w_goal_proximity}")
    print(f"|   goal_proximity_scale: {env_cfg.goal_proximity_scale}")
    print(f"|   w_goal_best_progress: {env_cfg.w_goal_best_progress}")
    print(f"|   w_goal_hover: {env_cfg.w_goal_hover}")
    print(f"|   w_energy: {env_cfg.w_energy}")
    print(f"|   w_smooth: {env_cfg.w_smooth}")
    print(f"|   w_speed: {env_cfg.w_speed}")
    print("=" * 60)

    training_params: dict[str, Any] = dict(sac_cfg)
    network_factory_cfg = dict(training_params.pop("network_factory"))
    # Keep num_eval_envs separate so we can pass it conditionally.
    num_eval_envs = int(training_params.pop("num_eval_envs", sac_cfg.num_eval_envs))
    training_params.pop("run_evals", None)

    # Filter kwargs by installed Brax signature so train-loss metrics are passed when supported.
    try:
        train_sig = inspect.signature(sac.train)
        supported_train_args = set(train_sig.parameters.keys())
    except Exception:
        supported_train_args = set()
    dropped_args = []
    for key in list(training_params.keys()):
        if supported_train_args and key not in supported_train_args:
            dropped_args.append(key)
            del training_params[key]
    if dropped_args:
        print(
            "Dropping unsupported sac.train args for this Brax version: "
            + ", ".join(sorted(dropped_args))
        )

    hidden_layer_sizes = tuple(network_factory_cfg.get("hidden_layer_sizes", (1024, 1024)))
    network_factory = functools.partial(
        sac_networks.make_sac_networks,
        hidden_layer_sizes=hidden_layer_sizes,
        policy_network_layer_norm=bool(
            network_factory_cfg.get("policy_network_layer_norm", False)
        ),
        q_network_layer_norm=bool(network_factory_cfg.get("q_network_layer_norm", False)),
    )

    restore_checkpoint_path = None
    if args.load_checkpoint_path:
        restore_checkpoint_path = str(Path(args.load_checkpoint_path).resolve())
        print(f"Restoring from checkpoint: {restore_checkpoint_path}")

    train_fn_kwargs = dict(
        **training_params,
        network_factory=network_factory,
        seed=args.seed,
        restore_checkpoint_path=restore_checkpoint_path,
        checkpoint_logdir=str(ckpt_dir),
        wrap_env=True,
    )
    if supported_train_args and "wrap_env_fn" not in supported_train_args:
        raise RuntimeError(
            "Installed brax.training SAC does not support custom wrap_env_fn, "
            "which is required for MuJoCo Playground MJX envs."
        )
    train_fn_kwargs["wrap_env_fn"] = wrapper.wrap_for_brax_training
    if (not supported_train_args) or ("num_eval_envs" in supported_train_args):
        train_fn_kwargs["num_eval_envs"] = num_eval_envs
    train_fn = functools.partial(sac.train, **train_fn_kwargs)

    # Initialize logger with progress bar
    logger = TrainingLogger(
        total_timesteps=sac_cfg.num_timesteps,
        num_envs=sac_cfg.num_envs,
        episode_length=sac_cfg.episode_length,
        use_progress_bar=args.progress_bar,
    )
    tb_writer = None
    if args.tensorboard:
        if SummaryWriter is None:
            print("Warning: TensorBoard logging requested but SummaryWriter is unavailable.")
            print("Install with `pip install tensorboard` (or install PyTorch with TensorBoard support).")
        else:
            tb_writer = SummaryWriter(log_dir=str(tb_dir))
            tb_writer.add_text("run/experiment", exp_name, 0)
            tb_writer.add_text("run/logdir", str(logdir), 0)
            tb_writer.add_text("run/env_config", json.dumps(env_cfg_json, indent=2), 0)
            tb_writer.add_text("run/sac_config", json.dumps(sac_cfg_json, indent=2), 0)

    jit_start_time = time.monotonic()
    first_progress_call = [True]
    progress_calls = [0]
    tb_logging_failed = [False]
    best_checkpoint_rank = [(float("-inf"), float("-inf"), float("-inf"))]

    def progress(num_steps, metrics):
        nonlocal jit_start_time, tb_writer
        if first_progress_call[0]:
            jit_time = time.monotonic() - jit_start_time
            print(f"\nJIT compilation time: {jit_time:.2f}s")
            print(f"[debug] first progress callback at step={int(num_steps)}")
            print(f"[debug] metric keys: {sorted(metrics.keys())}")
            if tb_writer is not None:
                tb_writer.add_scalar("time/jit_compile_seconds", jit_time, int(num_steps))
            first_progress_call[0] = False

        if sac_cfg.run_evals and "eval/episode_reward" in metrics:
            logger.log_eval(num_steps, metrics)
            candidate_rank = _checkpoint_rank(metrics)
            if candidate_rank > best_checkpoint_rank[0]:
                step = int(num_steps)
                if _save_best_checkpoint(
                    ckpt_dir=ckpt_dir,
                    step=step,
                    metrics=metrics,
                    best_dir=best_ckpt_dir,
                    best_meta_path=best_ckpt_meta_path,
                ):
                    best_checkpoint_rank[0] = candidate_rank
                    print(
                        "[debug] saved new best checkpoint "
                        f"step={step} success={candidate_rank[0]:.4f} "
                        f"final_dist={-candidate_rank[1]:.3f} reward={candidate_rank[2]:.3f}",
                        flush=True,
                    )
        else:
            logger.log_train(num_steps, metrics)

        record = _metrics_record(int(num_steps), metrics)
        with metrics_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record) + "\n")
        latest_metrics_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

        if tb_writer is not None:
            try:
                step = int(num_steps)
                for key, value in metrics.items():
                    scalar = _safe_float_scalar(value)
                    if scalar is None:
                        continue
                    tb_writer.add_scalar(str(key), scalar, step)
                # Stable aliases for frequently used reward curves.
                train_reward = None
                for reward_key in (
                    "training/episode_reward",
                    "episode/sum_reward",
                    "eval/episode_reward",
                ):
                    if reward_key not in metrics:
                        continue
                    candidate = _safe_float_scalar(metrics[reward_key])
                    if candidate is None:
                        continue
                    train_reward = candidate
                    if reward_key != "eval/episode_reward":
                        break
                if train_reward is not None:
                    tb_writer.add_scalar("train_reward/episode_reward", train_reward, step)
                eval_reward = _safe_float_scalar(metrics.get("eval/episode_reward"))
                if eval_reward is not None:
                    tb_writer.add_scalar("eval_reward/episode_reward", eval_reward, step)
                # Duplicate common loss keys under a stable namespace for quick filtering.
                loss_tag_map = {
                    "training/actor_loss": "train_loss/actor_loss",
                    "training/critic_loss": "train_loss/critic_loss",
                    "training/alpha_loss": "train_loss/alpha_loss",
                    "training/alpha": "train_loss/alpha",
                    "training/buffer_current_size": "train_misc/buffer_current_size",
                }
                for src_key, dst_key in loss_tag_map.items():
                    if src_key not in metrics:
                        continue
                    scalar = _safe_float_scalar(metrics[src_key])
                    if scalar is None:
                        continue
                    tb_writer.add_scalar(dst_key, scalar, step)
                progress_calls[0] += 1
                if progress_calls[0] % 5 == 0:
                    tb_writer.flush()
            except Exception as exc:
                if not tb_logging_failed[0]:
                    print(f"Warning: disabling TensorBoard callback logging due to error: {exc}")
                    tb_logging_failed[0] = True
                tb_writer = None

    if args.rscope_envs > 0:
        raise NotImplementedError(
            "rscope rollouts are not supported in train_sac.py because Brax SAC "
            "does not expose the PPO-style policy_params_fn callback."
        )

    print("\nStarting training...", flush=True)
    print(f"[debug] jax backend: {jax.default_backend()} devices={len(jax.devices())}", flush=True)
    print(f"[debug] env observation_size: {env.observation_size}", flush=True)
    print(f"[debug] env action_size: {env.action_size}", flush=True)
    print(
        "[debug] train summary: "
        f"run_evals={sac_cfg.run_evals}, "
        f"eval_env={'yes' if eval_env is not None else 'no'}, "
        f"num_eval_envs={num_eval_envs}, "
        f"deterministic_eval={train_fn_kwargs.get('deterministic_eval')}, "
        f"normalize_observations={train_fn_kwargs.get('normalize_observations')}",
        flush=True,
    )
    print(
        f"[debug] sac.train kwargs: {sorted(train_fn_kwargs.keys())}",
        flush=True,
    )
    train_start = time.monotonic()
    train_time = 0.0
    try:
        print("[debug] entering brax sac.train() call", flush=True)
        make_inference_fn, params, _ = train_fn(
            environment=env,
            progress_fn=progress,
            eval_env=eval_env,
        )
        print("[debug] brax sac.train() returned", flush=True)
        del make_inference_fn, params
        train_time = time.monotonic() - train_start
    finally:
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()
        logger.close()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"| Total training time: {train_time:.2f}s")
    print(f"| Steps per second: {sac_cfg.num_timesteps / train_time:,.0f}")
    print(f"| Artifacts saved to: {logdir}")
    print("=" * 60)


def main():
    args = _build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
