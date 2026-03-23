"""Stage-1 SAC config focused on native Windows compatibility."""

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
    "terminate_on_collision": True,
    "collision_terminate_steps": 12,
    "w_progress": 1.0,
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
    "n_envs": 1,
    "vec_env": "dummy",
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

SAC_CFG = {
    "policy": "MultiInputPolicy",
    "total_timesteps": 300_000,
    "learning_rate": 3e-4,
    "buffer_size": 500_000,
    "learning_starts": 10_000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.995,
    "train_freq": 1,
    "gradient_steps": 1,
    "ent_coef": "auto",
    "target_update_interval": 1,
    "target_entropy": "auto",
    "use_sde": False,
    "sde_sample_freq": -1,
    "policy_kwargs": {"net_arch": [256, 256]},
    "tensorboard_log": "drone_training/artifacts/tensorboard",
    "save_freq": 50_000,
    "use_eval": True,
    "eval_freq": 10_000,
    "n_eval_episodes": 5,
    "run_name": "sac_attempt",
    "verbose": 1,
    "device": "auto",
    "progress_bar": True,
}

OUT_DIR = "drone_training/artifacts_sac"
