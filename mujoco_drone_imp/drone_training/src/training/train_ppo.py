from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import drone_training
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from typing import Optional
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


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
    vec_env = make_vec_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
    )
    return vec_env


def build_model(vec_env, ppo_cfg: Optional[dict] = None):
    ppo_cfg = ppo_cfg or {}
    n_steps = int(ppo_cfg.get("n_steps", 2048))
    batch_size = int(ppo_cfg.get("batch_size", 256))
    rollout_size = n_steps * int(getattr(vec_env, "num_envs", 1))
    if batch_size > rollout_size:
        raise ValueError(
            f"batch_size ({batch_size}) must be <= n_steps * n_envs ({rollout_size})."
        )
    model = PPO(
        policy=ppo_cfg.get("policy", "MultiInputPolicy"),
        env=vec_env,
        learning_rate=ppo_cfg.get("learning_rate", 3e-4),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=ppo_cfg.get("n_epochs", 10),
        gamma=ppo_cfg.get("gamma", 0.99),
        gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
        clip_range=ppo_cfg.get("clip_range", 0.2),
        clip_range_vf=ppo_cfg.get("clip_range_vf", None),
        ent_coef=ppo_cfg.get("ent_coef", 0.0),
        vf_coef=ppo_cfg.get("vf_coef", 0.5),
        max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
        target_kl=ppo_cfg.get("target_kl", None),
        use_sde=ppo_cfg.get("use_sde", False),
        sde_sample_freq=ppo_cfg.get("sde_sample_freq", -1),
        policy_kwargs=ppo_cfg.get("policy_kwargs", None),
        tensorboard_log=ppo_cfg.get("tensorboard_log", "drone_training/artifacts/tensorboard"),
        seed=ppo_cfg.get("seed", 0),
        verbose=ppo_cfg.get("verbose", 1),
        device=ppo_cfg.get("device", "auto"),
    )

    return model


def make_eval_env(env_cfg, video_dir="drone_training/artifacts/videos"):
    kwargs = dict(env_cfg.get("eval_kwargs", env_cfg.get("kwargs", {})))
    # if kwargs.get("render_mode") != "rgb_array":
    #     raise ValueError(
    #         "RecordVideo requires eval render_mode='rgb_array' in env config."
    #     )
    env = gym.make(env_cfg["id"], **kwargs)
    env = Monitor(env)
    # env = RecordVideo(
    #     env,
    #     video_folder=video_dir,
    #     episode_trigger=lambda ep: ep % 5 == 0,  # record every 5th eval episode
    #     name_prefix="eval",
    # )
    return env


def _set_env_attr(env, name: str, value):
    """Set attr for VecEnv, wrapped VecEnv, or regular Gym env."""
    if hasattr(env, "set_attr"):
        env.set_attr(name, value)
        return
    inner = getattr(env, "venv", None)
    if inner is not None:
        _set_env_attr(inner, name, value)
        return
    base = getattr(env, "unwrapped", env)
    if not hasattr(base, name):
        raise AttributeError(f"Env has no attribute '{name}'")
    setattr(base, name, value)


class CurriculumCallback(BaseCallback):
    """Applies stage-wise env parameter updates during training."""

    def __init__(
        self,
        schedule: list[dict],
        eval_env=None,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.schedule = sorted(
            [dict(s) for s in schedule if "until_timesteps" in s],
            key=lambda s: int(s["until_timesteps"]),
        )
        self._stage_idx = -1

    def _pick_stage(self, timesteps: int):
        for idx, stage in enumerate(self.schedule):
            if timesteps <= int(stage["until_timesteps"]):
                return idx, stage
        return len(self.schedule) - 1, self.schedule[-1]

    def _apply_stage(self, stage: dict):
        for k, v in stage.items():
            if k == "until_timesteps":
                continue
            _set_env_attr(self.training_env, k, v)
            if self.eval_env is not None:
                _set_env_attr(self.eval_env, k, v)
            self.logger.record(f"curriculum/{k}", float(v))

    def _on_training_start(self) -> None:
        if not self.schedule:
            return
        idx, stage = self._pick_stage(self.num_timesteps)
        self._stage_idx = idx
        self._apply_stage(stage)

    def _on_step(self) -> bool:
        if not self.schedule:
            return True
        idx, stage = self._pick_stage(self.num_timesteps)
        if idx != self._stage_idx:
            self._stage_idx = idx
            self._apply_stage(stage)
            if self.verbose > 0:
                print(
                    f"[curriculum] step={self.num_timesteps} stage={idx + 1}/{len(self.schedule)}"
                )
        else:
            for k, v in stage.items():
                if k != "until_timesteps":
                    self.logger.record(f"curriculum/{k}", float(v))
        return True


def build_callbacks(
    env_cfg: Optional[dict] = None,
    ppo_cfg: Optional[dict] = None,
    out_dir: Optional[str | Path] = None,
):
    env_cfg = env_cfg or {}
    ppo_cfg = ppo_cfg or {}
    out = Path(out_dir or "drone_training/artifacts")
    ckpt_dir = out / "checkpoints"
    eval_dir = out / "eval"
    best_dir = out / "best_model"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    n_envs = env_cfg.get("n_envs", 1)
    env_id = env_cfg["id"]
    env_kwargs = env_cfg.get("eval_kwargs", env_cfg.get("kwargs", {}))
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

    eval_env = None
    eval_freq = int(ppo_cfg.get("eval_freq", 25_000))
    if ppo_cfg.get("use_eval", True):
        eval_env = make_eval_env(env_cfg, video_dir=str(out / "videos"))
    curriculum_cfg = ppo_cfg.get("curriculum", {})
    if curriculum_cfg.get("enabled", False):
        callbacks.append(
            CurriculumCallback(
                schedule=list(curriculum_cfg.get("schedule", [])),
                eval_env=eval_env,
                verbose=1 if int(ppo_cfg.get("verbose", 0)) > 0 else 0,
            )
        )

    if eval_env is not None:
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
    out = Path(out_dir or "drone_training/artifacts")
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
