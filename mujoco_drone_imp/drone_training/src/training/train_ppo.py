from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import drone_training
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from typing import Optional


def make_env_vec(env_cfg: Optional[dict] = None):
    env_cfg = env_cfg or {}
    env_id = env_cfg.get("id", "DroneEnv-v0")
    n_envs = int(env_cfg.get("n_envs", 1))
    seed = env_cfg.get("seed", 0)
    env_kwargs = env_cfg.get("kwargs", {})
    vec_env = make_vec_env(env_id, n_envs=n_envs, seed=seed, env_kwargs=env_kwargs)
    return vec_env


def build_model(vec_env, ppo_cfg: Optional[dict] = None):
    ppo_cfg = ppo_cfg or {}
    model = PPO(
        policy=ppo_cfg.get("policy", "MultiInputPolicy"),
        env=vec_env,
        learning_rate=ppo_cfg.get("learning_rate", 3e-4),
        n_steps=ppo_cfg.get("n_steps", 2048),
        batch_size=ppo_cfg.get("batch_size", 256),
        n_epochs=ppo_cfg.get("n_epochs", 10),
        gamma=ppo_cfg.get("gamma", 0.99),
        gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
        clip_range=ppo_cfg.get("clip_range", 0.2),
        ent_coef=ppo_cfg.get("ent_coef", 0.0),
        vf_coef=ppo_cfg.get("vf_coef", 0.5),
        max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
        tensorboard_log=ppo_cfg.get("tensorboard_log", "artifacts/tensorboard"),
        seed=ppo_cfg.get("seed", 0),
        verbose=ppo_cfg.get("verbose", 1),
        device=ppo_cfg.get("device", "auto"),
    )

    return model


def build_callbacks(
    env_cfg: Optional[dict] = None,
    ppo_cfg: Optional[dict] = None,
    out_dir: Optional[str | Path] = None,
):
    env_cfg = env_cfg or {}
    ppo_cfg = ppo_cfg or {}
    out = Path(out_dir or "artifacts")
    ckpt_dir = out / "checkpoints"
    eval_dir = out / "eval"
    best_dir = out / "best_model"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    n_envs = env_cfg.get("n_envs", 1)
    env_id = env_cfg["id"]
    env_kwargs = env_cfg["kwargs"]
    seed = env_cfg.get("seed", 0)

    callbacks = []

    save_freq = int(ppo_cfg.get("save_freq", 50_000))
    callbacks.append(
        CheckpointCallback(
            save_freq=max(1, save_freq // n_envs),
            save_path=str(ckpt_dir),
            name_prefix=ppo_cfg.get("run_name", "ppo_drone"),
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
    )

    if ppo_cfg.get("use_eval", True):
        eval_env = make_vec_env(
            env_id,
            n_envs=1,
            env_kwargs=env_kwargs,
            seed=seed + 10_000,
        )
        eval_freq = int(ppo_cfg.get("eval_freq", 25_000))
        callbacks.append(
            EvalCallback(
                eval_env=eval_env,
                best_model_save_path=str(best_dir),
                log_path=str(eval_dir),
                eval_freq=max(1, eval_freq // n_envs),
                n_eval_episodes=int(ppo_cfg.get("n_eval_episodes", 5)),
                deterministic=True,
                render=False,
            )
        )

    return CallbackList(callbacks)


def train(
    env_cfg: Optional[dict] = None,
    ppo_cfg: Optional[dict] = None,
    out_dir: Optional[str | Path] = None,
):
    ppo_cfg = ppo_cfg or {}
    out = Path(out_dir or "artifacts")
    out.mkdir(parents=True, exist_ok=True)

    vec_env = make_env_vec(env_cfg)
    model = build_model(vec_env, ppo_cfg)
    callbacks = build_callbacks(env_cfg, ppo_cfg, out)
    model.learn(
        total_timesteps=int(ppo_cfg.get("total_timesteps", 100_000)),
        callback=callbacks,
        tb_log_name=ppo_cfg.get("run_name", "ppo_drone"),
        progress_bar=ppo_cfg.get("progress_bar", True),
    )
    final_path = out / "ppo_final"
    model.save(str(final_path))
    vec_env.close()
    return model, str(final_path)


if __name__ == "__main__":
    from drone_training.configs.ppo_config import ENV_CFG, OUT_DIR, PPO_CFG

    train(env_cfg=ENV_CFG, ppo_cfg=PPO_CFG, out_dir=OUT_DIR)
