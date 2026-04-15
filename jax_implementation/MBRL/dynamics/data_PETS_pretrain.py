import argparse
import json
import time
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import numpy as np

import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from jax_implementation.env import _pid_demo_action, default_config, newDrone


def _parse_override(raw_value: str) -> Any:
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return raw_value


def _apply_env_overrides(cfg, override_items: list[str]) -> dict[str, Any]:
    applied: dict[str, Any] = {}
    for item in override_items:
        if "=" not in item:
            raise ValueError(
                f"Invalid --env_override '{item}'. Expected KEY=VALUE."
            )
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --env_override '{item}'. Empty key.")
        value = _parse_override(raw_value.strip())
        cfg[key] = value
        applied[key] = value
    return applied


def _flatten_obs_batch(obs_batch: dict[str, jax.Array], obs_keys: tuple[str, ...]) -> jax.Array:
    return jp.concatenate(
        [
            jp.asarray(obs_batch[key], dtype=jp.float32).reshape((obs_batch[key].shape[0], -1))
            for key in obs_keys
        ],
        axis=-1,
    )


def _mask_where(mask: jax.Array, true_value: jax.Array, false_value: jax.Array) -> jax.Array:
    mask = jp.asarray(mask, dtype=jp.bool_)
    while mask.ndim < true_value.ndim:
        mask = mask[..., None]
    return jp.where(mask, true_value, false_value)


def _build_obs_layout(env: newDrone, obs_keys: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    offset = 0
    layout: dict[str, dict[str, Any]] = {}
    for key in obs_keys:
        shape = tuple(int(dim) for dim in env.obs_spec[key]["shape"])
        size = int(np.prod(shape))
        layout[key] = {
            "shape": list(shape),
            "start": offset,
            "end": offset + size,
        }
        offset += size
    return layout


def _jsonable_value(value: Any) -> Any:
    """Recursively convert JAX/NumPy values into JSON-safe Python types."""
    if isinstance(value, dict):
        return {str(key): _jsonable_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, jax.Array):
        arr = np.asarray(jax.device_get(value))
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


def _make_action_sampler(
    env: newDrone,
    policy: str,
    action_noise_std: float,
):
    action_low = jp.asarray(env.action_low, dtype=jp.float32)
    action_high = jp.asarray(env.action_high, dtype=jp.float32)
    action_shape = tuple(int(dim) for dim in env.action_spec["shape"])
    noise_std_value = max(float(action_noise_std), 0.0)
    noise_std = jp.asarray(noise_std_value, dtype=jp.float32)

    def _sample_random(action_keys: jax.Array) -> jax.Array:
        return jax.vmap(
            lambda key: jax.random.uniform(
                key,
                shape=action_shape,
                minval=action_low,
                maxval=action_high,
                dtype=jp.float32,
            )
        )(action_keys)

    def _sample_pid(states, action_keys: jax.Array) -> jax.Array:
        del action_keys
        return jax.vmap(lambda state: _pid_demo_action(env, state))(states)

    def _sample_pid_noisy(states, action_keys: jax.Array) -> jax.Array:
        base_actions = jax.vmap(lambda state: _pid_demo_action(env, state))(states)
        if noise_std_value <= 0.0:
            return base_actions
        noise = jax.vmap(
            lambda key: jax.random.normal(key, shape=action_shape, dtype=jp.float32)
        )(action_keys)
        return jp.clip(base_actions + (noise_std * noise), action_low, action_high)

    if policy == "random":
        return lambda states, action_keys: _sample_random(action_keys)
    if policy == "pid":
        return _sample_pid
    if policy == "pid_noisy":
        return _sample_pid_noisy
    raise ValueError(f"Unsupported policy '{policy}'.")


def collect_parallel_rollouts(
    env: newDrone,
    num_envs: int,
    steps_per_env: int,
    seed: int,
    policy: str,
    action_noise_std: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any], float]:
    if num_envs <= 0:
        raise ValueError("num_envs must be positive.")
    if steps_per_env <= 0:
        raise ValueError("steps_per_env must be positive.")

    obs_keys = tuple(env.obs_spec.keys())
    obs_layout = _build_obs_layout(env, obs_keys)
    obs_dim = sum(spec["end"] - spec["start"] for spec in obs_layout.values())
    action_dim = int(np.prod(env.action_spec["shape"]))

    batch_reset = jax.vmap(lambda reset_rng: env.reset(reset_rng))
    batch_step = jax.vmap(lambda state, action: env.step(state, action))
    sample_actions = _make_action_sampler(env, policy=policy, action_noise_std=action_noise_std)

    @partial(jax.jit, static_argnames=("num_steps",))
    def _collect(reset_keys: jax.Array, rollout_keys: jax.Array, num_steps: int):
        init_states = batch_reset(reset_keys)
        init_episode_start = jp.ones((reset_keys.shape[0],), dtype=jp.float32)

        def _scan_step(carry, _):
            states, step_keys, episode_start = carry
            key_triplets = jax.vmap(lambda key: jax.random.split(key, 3))(step_keys)
            action_keys = key_triplets[:, 0, :]
            reset_keys = key_triplets[:, 1, :]
            next_step_keys = key_triplets[:, 2, :]

            obs = _flatten_obs_batch(states.obs, obs_keys)
            sampled_action = sample_actions(states, action_keys)
            next_states = batch_step(states, sampled_action)
            next_obs = _flatten_obs_batch(next_states.obs, obs_keys)
            done = jp.asarray(next_states.done, dtype=jp.float32).reshape((-1,))
            done_mask = done > 0.0
            reset_states = batch_reset(reset_keys)
            carry_states = jax.tree_util.tree_map(
                lambda nxt, rst: _mask_where(done_mask, rst, nxt),
                next_states,
                reset_states,
            )
            transition = {
                "obs": obs,
                "action": jp.asarray(sampled_action, dtype=jp.float32),
                "applied_action": jp.asarray(next_states.info["held_action"], dtype=jp.float32),
                "next_obs": next_obs,
                "reward": jp.asarray(next_states.reward, dtype=jp.float32).reshape((-1,)),
                "done": done,
                "terminated": jp.asarray(next_states.info["terminated"], dtype=jp.float32).reshape((-1,)),
                "truncated": jp.asarray(next_states.info["truncated"], dtype=jp.float32).reshape((-1,)),
                "success": jp.asarray(next_states.info["success"], dtype=jp.float32).reshape((-1,)),
                "episode_start": episode_start,
            }
            next_episode_start = done.astype(jp.float32)
            next_carry = (carry_states, next_step_keys, next_episode_start)
            return next_carry, transition

        (_, _, _), transitions = jax.lax.scan(
            _scan_step,
            (init_states, rollout_keys, init_episode_start),
            xs=None,
            length=num_steps,
        )
        return transitions

    seed_key = jax.random.PRNGKey(seed)
    reset_master_key, rollout_master_key = jax.random.split(seed_key)
    reset_keys = jax.random.split(reset_master_key, num_envs)
    rollout_keys = jax.random.split(rollout_master_key, num_envs)

    start_time = time.perf_counter()
    transitions = _collect(reset_keys, rollout_keys, steps_per_env)
    jax.block_until_ready(transitions["obs"])
    elapsed = time.perf_counter() - start_time

    dataset = {
        key: np.swapaxes(np.asarray(jax.device_get(value), dtype=np.float32), 0, 1)
        for key, value in transitions.items()
    }
    metadata = {
        "seed": int(seed),
        "num_envs": int(num_envs),
        "steps_per_env": int(steps_per_env),
        "num_transitions": int(num_envs * steps_per_env),
        "policy": policy,
        "action_noise_std": float(action_noise_std),
        "obs_keys": list(obs_keys),
        "obs_layout": obs_layout,
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "action_low": np.asarray(jax.device_get(env.action_low), dtype=np.float32).tolist(),
        "action_high": np.asarray(jax.device_get(env.action_high), dtype=np.float32).tolist(),
        "lidar_key": "lidar" if "lidar" in obs_layout else None,
        "lidar_slice": obs_layout.get("lidar"),
        "env_config": _jsonable_value(dict(env._config.to_dict())),
    }
    return dataset, metadata, elapsed


def save_dataset(
    dataset: dict[str, np.ndarray],
    metadata: dict[str, Any],
    out_dir: str,
) -> tuple[Path, Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    save_stem = (
        f"pets_pretrain_envB{metadata['num_envs']}_T{metadata['steps_per_env']}"
        f"_{metadata['policy']}_seed{metadata['seed']}"
    )
    dataset_path = out_path / f"{save_stem}.npz"
    meta_path = out_path / f"{save_stem}.json"

    np.savez(dataset_path, **dataset)
    meta_path.write_text(
        json.dumps(_jsonable_value(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return dataset_path, meta_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect PETS pretraining transitions from batched MJX env rollouts."
    )
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--steps_per_env", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--policy",
        choices=("random", "pid", "pid_noisy"),
        default="pid_noisy",
        help="Action source used during pretraining collection.",
    )
    parser.add_argument(
        "--action_noise_std",
        type=float,
        default=0.2,
        help="Gaussian action noise for pid_noisy collection.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="jax_implementation/MBRL/dyn_data",
    )
    parser.add_argument(
        "--env_override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a default_config entry. VALUE is parsed as JSON when possible.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    cfg = default_config()
    applied_overrides = _apply_env_overrides(cfg, args.env_override)
    env = newDrone(config=cfg)

    dataset, metadata, elapsed = collect_parallel_rollouts(
        env=env,
        num_envs=args.num_envs,
        steps_per_env=args.steps_per_env,
        seed=args.seed,
        policy=args.policy,
        action_noise_std=args.action_noise_std,
    )
    if applied_overrides:
        metadata["applied_env_overrides"] = applied_overrides.to_list()

    dataset_path, meta_path = save_dataset(
        dataset=dataset,
        metadata=metadata,
        out_dir=args.out_dir,
    )

    print(
        "Saved PETS pretrain dataset to",
        dataset_path,
    )
    print("Saved metadata to", meta_path)
    print(
        f"Collected {metadata['num_transitions']} transitions "
        f"from {metadata['num_envs']} parallel envs in {elapsed:.2f}s."
    )
    lidar_slice = metadata.get("lidar_slice")
    if lidar_slice is not None:
        print(
            "Flattened lidar slice:",
            f"[{lidar_slice['start']}:{lidar_slice['end']}]",
        )


if __name__ == "__main__":
    main()
