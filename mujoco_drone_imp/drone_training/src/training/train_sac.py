from pathlib import Path
from typing import Optional

import gymnasium as gym
import drone_training
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


def _make_single_env(env_id: str, env_kwargs: dict):
    # Windows spawn workers need to import the package again so Gym sees the env registration.
    import drone_training  # noqa: F401

    return gym.make(env_id, **env_kwargs)


def make_env_vec(env_cfg: Optional[dict] = None):
    env_cfg = env_cfg or {}
    env_id = env_cfg.get("id", "DroneEnv-v0")
    n_envs = int(env_cfg.get("n_envs", 1))
    seed = env_cfg.get("seed", 0)
    env_kwargs = env_cfg.get("train_kwargs", env_cfg.get("kwargs", {}))
    vec_mode = str(env_cfg.get("vec_env", "dummy")).lower()
    if vec_mode == "subproc":
        vec_env_cls = SubprocVecEnv
    elif vec_mode == "dummy":
        vec_env_cls = DummyVecEnv
    else:
        raise ValueError(
            f"Unsupported vec_env mode '{vec_mode}'. Use 'dummy' or 'subproc'."
        )
    vec_env_kwargs = {}
    if vec_mode == "subproc" and env_cfg.get("vec_env_start_method"):
        vec_env_kwargs["start_method"] = env_cfg["vec_env_start_method"]
    return make_vec_env(
        lambda: _make_single_env(env_id, env_kwargs),
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
    )


def build_model(vec_env, sac_cfg: Optional[dict] = None):
    sac_cfg = sac_cfg or {}
    return SAC(
        policy=sac_cfg.get("policy", "MultiInputPolicy"),
        env=vec_env,
        learning_rate=sac_cfg.get("learning_rate", 3e-4),
        buffer_size=int(sac_cfg.get("buffer_size", 1_000_000)),
        learning_starts=int(sac_cfg.get("learning_starts", 10_000)),
        batch_size=int(sac_cfg.get("batch_size", 256)),
        tau=float(sac_cfg.get("tau", 0.005)),
        gamma=float(sac_cfg.get("gamma", 0.99)),
        train_freq=int(sac_cfg.get("train_freq", 1)),
        gradient_steps=int(sac_cfg.get("gradient_steps", 1)),
        action_noise=sac_cfg.get("action_noise", None),
        replay_buffer_class=sac_cfg.get("replay_buffer_class", None),
        replay_buffer_kwargs=sac_cfg.get("replay_buffer_kwargs", None),
        ent_coef=sac_cfg.get("ent_coef", "auto"),
        target_update_interval=int(sac_cfg.get("target_update_interval", 1)),
        target_entropy=sac_cfg.get("target_entropy", "auto"),
        use_sde=bool(sac_cfg.get("use_sde", False)),
        sde_sample_freq=int(sac_cfg.get("sde_sample_freq", -1)),
        use_sde_at_warmup=bool(sac_cfg.get("use_sde_at_warmup", False)),
        stats_window_size=int(sac_cfg.get("stats_window_size", 100)),
        tensorboard_log=sac_cfg.get("tensorboard_log", "drone_training/artifacts/tensorboard"),
        policy_kwargs=sac_cfg.get("policy_kwargs", None),
        verbose=int(sac_cfg.get("verbose", 1)),
        seed=sac_cfg.get("seed", 0),
        device=sac_cfg.get("device", "auto"),
    )


def make_eval_env(env_cfg, video_dir="drone_training/artifacts_sac/videos"):
    del video_dir
    kwargs = dict(env_cfg.get("eval_kwargs", env_cfg.get("kwargs", {})))
    env = gym.make(env_cfg["id"], **kwargs)
    return Monitor(env)


def build_callbacks(
    env_cfg: Optional[dict] = None,
    sac_cfg: Optional[dict] = None,
    out_dir: Optional[str | Path] = None,
):
    env_cfg = env_cfg or {}
    sac_cfg = sac_cfg or {}
    out = Path(out_dir or "drone_training/artifacts_sac")
    ckpt_dir = out / "checkpoints"
    eval_dir = out / "eval"
    best_dir = out / "best_model"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    n_envs = int(env_cfg.get("n_envs", 1))
    callbacks = []
    save_freq = int(sac_cfg.get("save_freq", 50_000))
    callbacks.append(
        CheckpointCallback(
            save_freq=max(1, save_freq // n_envs),
            save_path=str(ckpt_dir),
            name_prefix=sac_cfg.get("run_name", "sac_drone"),
            save_replay_buffer=True,
            save_vecnormalize=False,
        )
    )

    if sac_cfg.get("use_eval", True):
        eval_freq = int(sac_cfg.get("eval_freq", 25_000))
        eval_env = make_eval_env(env_cfg, video_dir=str(out / "videos"))
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=str(best_dir),
                log_path=str(eval_dir),
                eval_freq=max(1, eval_freq // n_envs),
                n_eval_episodes=int(sac_cfg.get("n_eval_episodes", 5)),
                deterministic=True,
                render=False,
            )
        )

    return CallbackList(callbacks)


def train(
    env_cfg: Optional[dict] = None,
    sac_cfg: Optional[dict] = None,
    out_dir: Optional[str | Path] = None,
):
    sac_cfg = sac_cfg or {}
    out = Path(out_dir or "drone_training/artifacts_sac")
    out.mkdir(parents=True, exist_ok=True)

    vec_env = make_env_vec(env_cfg)
    model = build_model(vec_env, sac_cfg)
    callbacks = build_callbacks(env_cfg, sac_cfg, out)
    model.learn(
        total_timesteps=int(sac_cfg.get("total_timesteps", 100_000)),
        callback=callbacks,
        tb_log_name=sac_cfg.get("run_name", "sac_drone"),
        progress_bar=bool(sac_cfg.get("progress_bar", True)),
    )
    final_path = out / "sac_final"
    model.save(str(final_path))
    vec_env.close()
    return model, str(final_path)


if __name__ == "__main__":
    from drone_training.configs.sac_config import ENV_CFG, OUT_DIR, SAC_CFG

    train(env_cfg=ENV_CFG, sac_cfg=SAC_CFG, out_dir=OUT_DIR)
