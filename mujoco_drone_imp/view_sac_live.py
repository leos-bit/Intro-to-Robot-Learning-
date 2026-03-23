from __future__ import annotations

import argparse
import time
from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import mujoco as mj
import mujoco.viewer
from stable_baselines3 import SAC

import drone_training  # noqa: F401
from drone_training.configs.sac_config import ENV_CFG


def build_env():
    env_cfg = deepcopy(ENV_CFG)
    env_kwargs = dict(env_cfg.get("eval_kwargs", {}))
    env_kwargs["render_mode"] = None
    return gym.make(env_cfg["id"], **env_kwargs)


def configure_viewer(viewer, model):
    camera_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, "track")
    if camera_id != -1:
        viewer.cam.type = mj.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = camera_id


def run_episode(env, model, viewer, seed: int, deterministic: bool):
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    sim_dt = float(env.unwrapped.model.opt.timestep)

    while viewer.is_running() and not (terminated or truncated):
        loop_start = time.perf_counter()
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1

        viewer.sync()
        elapsed = time.perf_counter() - loop_start
        if sim_dt > elapsed:
            time.sleep(sim_dt - elapsed)

    return {
        "steps": steps,
        "return": total_reward,
        "terminated": terminated,
        "truncated": truncated,
        "distance": info.get("distance"),
        "success": info.get("success"),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Open a live MuJoCo viewer and run a saved SAC policy."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("drone_training/artifacts_sac/best_model/best_model.zip"),
        help="Path to the saved SAC zip file.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to run in the live viewer.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base seed for episode resets.")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions stochastically instead of deterministic playback.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    policy = SAC.load(str(args.model))
    env = build_env()
    unwrapped = env.unwrapped

    print(f"model={args.model}")
    print(f"episodes={args.episodes}")
    print("Close the viewer window to stop early.")

    with mujoco.viewer.launch_passive(unwrapped.model, unwrapped.data) as viewer:
        configure_viewer(viewer, unwrapped.model)
        for episode_idx in range(args.episodes):
            if not viewer.is_running():
                break

            metrics = run_episode(
                env=env,
                model=policy,
                viewer=viewer,
                seed=args.seed + episode_idx,
                deterministic=not args.stochastic,
            )
            print(
                f"episode={episode_idx + 1} "
                f"steps={metrics['steps']} "
                f"return={metrics['return']:.3f} "
                f"distance={metrics['distance']} "
                f"success={metrics['success']} "
                f"terminated={metrics['terminated']} "
                f"truncated={metrics['truncated']}"
            )
            if viewer.is_running():
                time.sleep(0.5)

    env.close()


if __name__ == "__main__":
    main()
