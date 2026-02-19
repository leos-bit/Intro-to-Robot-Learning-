"""Quick smoke-test configs for PPO training."""

ENV_CFG = {
    "id": "DroneEnv-v0",
    "n_envs": 2,
    "seed": 0,
    "kwargs": {
        "model_path": "Drone_MJCFs/skydio_x2/scene.xml",
        "dt": 0.01,
        "max_steps": 200,
        "render_mode": None,
        "xylim": 2.0,
        "zlim": 2.0,
        "vellim": 1.0,
        "yawrate_lim": 1.0,
    },
}

PPO_CFG = {
    "policy": "MultiInputPolicy",
    "total_timesteps": 10_000,
    "learning_rate": 3e-4,
    "n_steps": 256,
    "batch_size": 64,
    "n_epochs": 5,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "tensorboard_log": "artifacts/tensorboard",
    "save_freq": 5_000,
    "use_eval": True,
    "eval_freq": 5_000,
    "n_eval_episodes": 2,
    "run_name": "ppo_smoke",
    "verbose": 1,
    "device": "auto",
    "progress_bar": False,
}

OUT_DIR = "artifacts"
