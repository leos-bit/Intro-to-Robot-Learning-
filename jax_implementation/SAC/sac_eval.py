"""Live MuJoCo viewer for trained Brax SAC drone checkpoints."""

from __future__ import annotations

import argparse
import functools
import json
import runpy
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import mujoco.viewer
import numpy as np
from brax.training import networks as brax_networks
from brax.training.acme import running_statistics
from brax.training.agents.sac import checkpoint as sac_checkpoint
from brax.training.agents.sac import networks as sac_networks
from mujoco_playground._src import mjx_env

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from jax_implementation.env import default_config, newDrone


def _load_env_cfg_from_checkpoint(ckpt: Path):
    ckpt = Path(ckpt).resolve()
    env_cfg = default_config()
    run_dir = ckpt.parent.parent
    env_config_path = run_dir / "env_config.json"
    sac_config_path = run_dir / "sac_config.json"

    if env_config_path.exists():
        loaded_env_cfg = json.loads(env_config_path.read_text())
        for key, value in loaded_env_cfg.items():
            current_value = env_cfg.get(key, None)
            if isinstance(current_value, jax.Array):
                value = jp.asarray(value, dtype=current_value.dtype)
            env_cfg[key] = value

    if sac_config_path.exists():
        loaded_sac_cfg = json.loads(sac_config_path.read_text())
        sac_episode_length = int(loaded_sac_cfg["episode_length"])
        sac_action_repeat = int(loaded_sac_cfg["action_repeat"])
        if int(env_cfg.episode_length) != sac_episode_length:
            raise ValueError(
                "Saved env_config and sac_config disagree on episode_length: "
                f"{env_cfg.episode_length} vs {sac_episode_length}"
            )
        if int(env_cfg.max_steps) != sac_episode_length:
            raise ValueError(
                "Saved env_config max_steps does not match sac episode_length: "
                f"{env_cfg.max_steps} vs {sac_episode_length}"
            )
        if int(env_cfg.action_repeat) != sac_action_repeat:
            raise ValueError(
                "Saved env_config and sac_config disagree on action_repeat: "
                f"{env_cfg.action_repeat} vs {sac_action_repeat}"
            )

    return env_cfg


def load_policy(ckpt: Path):
    ns = runpy.run_path(
        str(_REPO_ROOT / "jax_implementation/SAC/train_sac.py"),
        run_name="sac_train_eval_loader",
    )
    StateObsWrapper = ns["StateObsWrapper"]

    ckpt = Path(ckpt).resolve()
    env_cfg = _load_env_cfg_from_checkpoint(ckpt)
    env = StateObsWrapper(newDrone(config=env_cfg))

    params = sac_checkpoint.load(ckpt)
    cfg = json.loads((ckpt / "sac_network_config.json").read_text())
    kwargs = dict(cfg["network_factory_kwargs"])

    if "activation" in kwargs:
        kwargs["activation"] = brax_networks.ACTIVATION[kwargs["activation"]]

    for key in ("policy_network_kernel_init_fn", "q_network_kernel_init_fn"):
        name = kwargs.get(key)
        if name is None:
            kwargs.pop(key, None)
        else:
            kwargs[key] = brax_networks.KERNEL_INITIALIZER[name]

    for key in ("policy_network_kernel_init_kwargs", "q_network_kernel_init_kwargs"):
        if kwargs.get(key) is None:
            kwargs.pop(key, None)

    network_factory = functools.partial(sac_networks.make_sac_networks, **kwargs)
    normalize = running_statistics.normalize if cfg["normalize_observations"] else (lambda x, y: x)
    sac_network = network_factory(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=normalize,
    )
    make_policy = sac_networks.make_inference_fn(sac_network)
    policy = make_policy(params, deterministic=True)
    return policy, env, params


def _sync_viewer_data(env: newDrone, viewer_data: mujoco.MjData, state: mjx_env.State) -> None:
    viewer_data.qpos[:] = np.asarray(state.data.qpos)
    viewer_data.qvel[:] = np.asarray(state.data.qvel)
    viewer_data.ctrl[:] = np.asarray(state.data.ctrl)
    if env.mj_model.nmocap > 0:
        viewer_data.mocap_pos[:] = np.asarray(state.data.mocap_pos)
        viewer_data.mocap_quat[:] = np.asarray(state.data.mocap_quat)
    mujoco.mj_forward(env.mj_model, viewer_data)


def _set_overview_camera(env: newDrone, viewer) -> None:
    arena_half_extent = float(getattr(env, "xylim", 10.0))
    arena_height = float(getattr(env, "zlim", 8.0))
    arena_diagonal = float(np.sqrt(8.0 * (arena_half_extent**2) + (arena_height**2)))
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.fixedcamid = -1
    viewer.cam.lookat[:] = np.array(
        [0.0, 0.0, max(1.5, min(arena_height * 0.4, arena_height - 0.5))],
        dtype=np.float64,
    )
    viewer.cam.distance = max(arena_diagonal * 1.2, arena_half_extent * 2.8)
    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -30.0


def _draw_target_marker(viewer, state: mjx_env.State, radius: float = 0.18) -> None:
    """Draws a viewer-only target marker that physics and lidar cannot see."""
    if viewer is None or not hasattr(viewer, "user_scn"):
        return
    user_scn = viewer.user_scn
    if user_scn.ngeom >= len(user_scn.geoms):
        return
    target = np.asarray(state.info["target"], dtype=np.float64).reshape(3)
    geom = user_scn.geoms[user_scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0], dtype=np.float64),
        target,
        np.eye(3, dtype=np.float64).reshape(-1),
        np.array([0.05, 0.95, 0.15, 0.85], dtype=np.float32),
    )
    user_scn.ngeom += 1


def run_eval_single(
    policy,
    env,
    num_steps: int,
    seed: int,
    render: bool,
    jit_step: bool,
    stop_on_done: bool = False,
) -> None:
    base_env = env.env if hasattr(env, "env") else env
    rng = jax.random.PRNGKey(seed)
    rng, reset_rng = jax.random.split(rng)
    state = env.reset(reset_rng)
    step_fn = jax.jit(env.step) if jit_step else env.step

    viewer = None
    viewer_data = None
    if render:
        viewer_data = mujoco.MjData(base_env.mj_model)
        _sync_viewer_data(base_env, viewer_data, state)
        viewer = mujoco.viewer.launch_passive(base_env.mj_model, viewer_data)
        _set_overview_camera(base_env, viewer)
        viewer.user_scn.ngeom = 0
        _draw_target_marker(viewer, state)
        viewer.sync()

    episodes = 0
    try:
        for step in range(num_steps):
            rng, action_key = jax.random.split(rng)
            action, _ = policy(state.obs, action_key)
            state = step_fn(state, action)

            if render and viewer is not None:
                _sync_viewer_data(base_env, viewer_data, state)
                viewer.user_scn.ngeom = 0
                _draw_target_marker(viewer, state)
                viewer.sync()
                if hasattr(viewer, "is_running") and not viewer.is_running():
                    print("Viewer closed, stopping demo.")
                    break

            if step % 50 == 0:
                dist = float(state.info["distance"])
                pos = np.asarray(state.info["agent_location"])
                vel = np.asarray(state.info["agent_vel"])
                hold_streak = int(state.info["goal_hold_streak"])
                print(
                    f"step={step:04d} reward={float(state.reward): .3f} "
                    f"dist={dist: .3f} hold={hold_streak:03d} "
                    f"pos={np.round(pos, 3)} vel={np.round(vel, 3)}"
                )

            if bool(state.done):
                episodes += 1
                print(
                    "episode ended",
                    f"episode={episodes}",
                    f"step={step}",
                    f"success={bool(state.info['success'])}",
                    f"collision={bool(state.info['collision'])}",
                    f"oob={bool(state.info['out_of_bounds'])}",
                    f"numerical_issue={bool(state.info['numerical_issue'])}",
                )
                if stop_on_done:
                    break
                rng, reset_rng = jax.random.split(rng)
                state = env.reset(reset_rng)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        if viewer is not None:
            viewer.close()


def rollout_episode(policy, env, seed: int, num_steps: int, jit_step: bool = True):
    rng = jax.random.PRNGKey(seed)
    rng, reset_rng = jax.random.split(rng)
    state = env.reset(reset_rng)
    step_fn = jax.jit(env.step) if jit_step else env.step

    total_reward = 0.0
    final_step = 0
    for step in range(num_steps):
        rng, action_key = jax.random.split(rng)
        action, _ = policy(state.obs, action_key)
        state = step_fn(state, action)
        total_reward += float(state.reward)
        final_step = step + 1
        if bool(state.done):
            break

    return {
        "seed": seed,
        "steps": final_step,
        "reward": total_reward,
        "final_distance": float(state.info["distance"]),
        "success": bool(state.info["success"]),
        "collision": bool(state.info["collision"]),
        "out_of_bounds": bool(state.info["out_of_bounds"]),
        "hold_streak": int(state.info["goal_hold_streak"]),
    }


def select_best_episode(policy, env, base_seed: int, episodes: int, num_steps: int, jit_step: bool = True):
    results = []
    for i in range(episodes):
        seed = base_seed + i
        result = rollout_episode(policy, env, seed, num_steps, jit_step=jit_step)
        results.append(result)
        print(
            f"candidate {i + 1:02d}/{episodes:02d} seed={seed} "
            f"success={result['success']} reward={result['reward']:.3f} "
            f"final_dist={result['final_distance']:.3f} steps={result['steps']} "
            f"collision={result['collision']} oob={result['out_of_bounds']}"
        )

    def rank(result):
        return (
            int(result["success"]),
            -int(result["collision"]),
            -int(result["out_of_bounds"]),
            -float(result["final_distance"]),
            float(result["reward"]),
        )

    best = max(results, key=rank)
    print(
        "selected best",
        f"seed={best['seed']}",
        f"success={best['success']}",
        f"reward={best['reward']:.3f}",
        f"final_dist={best['final_distance']:.3f}",
        f"steps={best['steps']}",
    )
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jit_step", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--select_best", action="store_true", help="Evaluate several seeds headlessly, then replay the best one.")
    parser.add_argument("--episodes", type=int, default=20, help="Candidate episodes to scan when --select_best is used.")
    parser.add_argument("--stop_on_done", action="store_true", help="Stop after the first episode ends instead of resetting.")
    args = parser.parse_args()
    seed = int(np.random.randint(0, 1_000_000)) if args.seed is None else args.seed
    policy, env, _ = load_policy(args.checkpoint)
    print(f"checkpoint={Path(args.checkpoint).resolve()}")
    if args.select_best:
        best = select_best_episode(policy, env, seed, args.episodes, args.num_steps, args.jit_step)
        seed = best["seed"]
    print(f"seed={seed}")
    run_eval_single(
        policy,
        env,
        args.num_steps,
        seed,
        args.render,
        args.jit_step,
        stop_on_done=args.stop_on_done or args.select_best,
    )


if __name__ == "__main__":
    main()
