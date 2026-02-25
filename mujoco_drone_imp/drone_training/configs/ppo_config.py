"""Stage-1 PPO config focused on longer, more stable episodes."""

_ENV_BASE = {
    "model_path": "Drone_MJCFs/skydio_x2/scene.xml",
    "dt": 0.01,
    "max_steps": 2000,
    "xylim": 6.0,
    "zlim": 3.5,
    "vellim": 1.5,
    "yawrate_lim": 0.7,
    "action_scale": 0.35,
    "spawn_z_min": 0.8,
    "target_dist_min": 0.8,
    "target_dist_max": 2.5,
    # Stage-1: allow brief contact but reset if the drone keeps colliding.
    "terminate_on_collision": True,
    "collision_terminate_steps": 12,
    # Reward/control tuning to avoid saturated full-throttle policies.
    "w_progress": 1.0,
    # Significantly stronger penalties on actuator effort/change and speed.
    "w_energy": 0.25,
    "w_smooth": 0.45,
    "w_speed": 0.08,
    "r_collision": 6.0,
    "termination_penalty": 15.0,
    "r_goal": 35.0,
    "eps_goal": 0.35,
    "k_xy": 0.25,
    "k_z": 0.60,
    "k_yaw": 0.08,
    "safety_speed_scale": 5.0,
}

ENV_CFG = {
    "id": "DroneEnv-v0",
    "n_envs": 10,
    "vec_env": "subproc",
    "vec_env_start_method": "forkserver",
    "seed": 0,
    "train_kwargs": {
        **_ENV_BASE,
        "render_mode": None,
    },
    "eval_kwargs": {
        **_ENV_BASE,
        "render_mode": None,
    },
}

PPO_CFG = {
    "policy": "MultiInputPolicy",
    "total_timesteps": 20_000_000,
    "learning_rate": 1e-4,
    "n_steps": 3000,
    "batch_size":5000,
    "n_epochs": 6,
    "gamma": 0.995,
    "gae_lambda": 0.97,
    "clip_range": 0.2,
    "clip_range_vf": 0.2,
    "ent_coef": 0.007,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": 0.02,
    "use_sde": True,
    "sde_sample_freq": 64,
    "policy_kwargs": {"net_arch": {"pi": [1024, 1024], "vf": [1024, 1024]}},
    "tensorboard_log": "drone_training/artifacts/tensorboard",
    "save_freq": 100_000,
    "use_eval": False,
    "eval_freq": 5_000,
    "n_eval_episodes": 10,
    "run_name": "ppo_attempt",
    "verbose": 1,
    "device": "auto",
    "progress_bar": True,
    "curriculum": {
        "enabled": True,
        "schedule": [
            {
                "until_timesteps": 2_000_000,
                "target_dist_min": 0.4,
                "target_dist_max": 1.6,
                "action_scale": 0.25,
                "eps_goal": 0.40,
                "collision_terminate_steps": 14,
            },
            {
                "until_timesteps": 8_000_000,
                "target_dist_min": 0.6,
                "target_dist_max": 2.5,
                "action_scale": 0.32,
                "eps_goal": 0.35,
                "collision_terminate_steps": 12,
            },
            {
                "until_timesteps": 20_000_000,
                "target_dist_min": 0.8,
                "target_dist_max": 3.5,
                "action_scale": 0.40,
                "eps_goal": 0.30,
                "collision_terminate_steps": 10,
            },
        ],
    },
}

OUT_DIR = "drone_training/artifacts"
