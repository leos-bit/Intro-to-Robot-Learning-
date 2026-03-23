"""Default configuration for off-policy SAC runs on the drone task."""

from __future__ import annotations

from ml_collections import config_dict


DEFAULT_LOG_ROOT = "jax_implementation/SAC/artifacts"


def default_env_overrides() -> config_dict.ConfigDict:
    """Environment overrides for the report-facing SAC baseline."""
    return config_dict.ConfigDict(
        dict(
            xylim=6.0,
            zlim=3.5,
            vellim=1.5,
            yawrate_lim=0.7,
            action_scale=0.35,
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
    """Training defaults for the off-policy SAC baseline."""
    return config_dict.ConfigDict(
        dict(
            num_timesteps=20_000_000,
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
