import argparse
import json
from functools import partial
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from flax.training import train_state
import optax

OBS_DIM = 96
ACTION_DIM = 4
MAX_OBSTACLES = 15
TARGET_DIM = OBS_DIM + 1
STATS_CHUNK_ENVS = 8
DEFAULT_DATA_PATH = (
    "jax_implementation/MBRL/dyn_data/"
    "pets_pretrain_envB512_T10000_pid_noisy_seed0.npz"
)
LOGVAR_REG_COEF = 1e-2
GRAD_CLIP_NORM = 100.0
LOGVAR_LOWER_BOUND = -10.0
LOGVAR_UPPER_BOUND = 0.5
LOGVAR_BOUND_EPS = 1e-6


class MLP(nn.Module):
    widths: tuple[int, ...] = (512, 512)

    @nn.compact
    def __call__(self, x):
        for w in self.widths:
            x = nn.Dense(w)(x)
            x = nn.silu(x)
        return x


class DirectionAwareLidarEncoder(nn.Module):
    max_dist: float = 6.0
    out_dim: int = 32

    @nn.compact
    def __call__(self, lidar):
        lidar = jnp.clip(lidar, 0.0, self.max_dist) / self.max_dist
        prox = 1.0 - lidar

        axis = prox[..., 0:6]
        horiz_diag = prox[..., 6:10]
        up_diag = prox[..., 10:14]
        down_diag = prox[..., 14:18]

        sym = jnp.stack(
            [
                axis[..., 0] - axis[..., 2],
                axis[..., 1] - axis[..., 3],
                axis[..., 4] - axis[..., 5],
                horiz_diag[..., 0] - horiz_diag[..., 2],
                horiz_diag[..., 1] - horiz_diag[..., 3],
                up_diag[..., 2] - up_diag[..., 3],
                down_diag[..., 2] - down_diag[..., 3],
            ],
            axis=-1,
        )

        stats = jnp.stack(
            [
                jnp.min(prox, axis=-1),
                jnp.max(prox, axis=-1),
                jnp.mean(prox, axis=-1),
                jnp.mean(axis, axis=-1),
                jnp.mean(horiz_diag, axis=-1),
                jnp.mean(up_diag, axis=-1),
                jnp.mean(down_diag, axis=-1),
            ],
            axis=-1,
        )

        axis_feat = MLP((32, 16))(axis)
        horiz_feat = MLP((32, 16))(horiz_diag)
        up_feat = MLP((32, 16))(up_diag)
        down_feat = MLP((32, 16))(down_diag)
        sym_feat = MLP((16,))(sym)
        stats_feat = MLP((16,))(stats)

        feat = jnp.concatenate(
            [axis_feat, horiz_feat, up_feat, down_feat, sym_feat, stats_feat],
            axis=-1,
        )
        return MLP((64, self.out_dim))(feat)


def _safe_l2_norm(x, axis=None, keepdims=False, eps: float = 1e-8):
    x = jnp.asarray(x, dtype=jnp.float32)
    return jnp.sqrt(jnp.sum(jnp.square(x), axis=axis, keepdims=keepdims) + eps)


class MaskedObstacleEncoder(nn.Module):
    out_dim: int = 64
    max_obstacles: int = MAX_OBSTACLES

    @nn.compact
    def __call__(self, obstacle_rel, obstacle_mask, num_active):
        mask = jnp.asarray(obstacle_mask, dtype=jnp.float32)[..., None]
        rel = jnp.asarray(obstacle_rel, dtype=jnp.float32)
        dist = _safe_l2_norm(rel, axis=-1, keepdims=True)

        per_obstacle = jnp.concatenate([rel, dist], axis=-1)
        per_obstacle = MLP((64, 64))(per_obstacle)
        per_obstacle = per_obstacle * mask

        valid_count = jnp.sum(mask, axis=-2)
        denom = jnp.maximum(valid_count, 1.0)
        mean_pool = jnp.sum(per_obstacle, axis=-2) / denom

        max_fill = jnp.full_like(per_obstacle, -1e9)
        max_pool = jnp.max(jnp.where(mask > 0.0, per_obstacle, max_fill), axis=-2)
        has_any = valid_count > 0.0
        max_pool = jnp.where(has_any, max_pool, jnp.zeros_like(max_pool))

        flat_dist = jnp.squeeze(dist, axis=-1)
        large = jnp.full_like(flat_dist, 1e6)
        closest = jnp.min(
            jnp.where(obstacle_mask > 0.0, flat_dist, large),
            axis=-1,
            keepdims=True,
        )
        closest = jnp.where(has_any, closest, jnp.zeros_like(closest))

        density = jnp.asarray(num_active, dtype=jnp.float32) / float(self.max_obstacles)
        pooled = jnp.concatenate([mean_pool, max_pool, closest, density], axis=-1)
        return MLP((128, self.out_dim))(pooled)


def split_model_input(x):
    obs = x[..., :OBS_DIM]
    action = x[..., OBS_DIM:]
    return obs, action


def split_observation(obs):
    obstacle_rel = obs[..., 35:80].reshape(*obs.shape[:-1], MAX_OBSTACLES, 3)
    obstacle_mask = obs[..., 80:95]
    return {
        "core": obs[..., :17],
        "lidar": obs[..., 17:35],
        "obstacle_rel": obstacle_rel,
        "obstacle_mask": obstacle_mask,
        "num_active": obs[..., 95:96],
    }


class Dynamics_Model(nn.Module):
    target_dim: int = 97
    trunk_widths: tuple[int, ...] = (256, 256, 256, 256)
    lidar_max_dist: float = 6.0

    @nn.compact
    def __call__(self, x):
        obs, action = split_model_input(x)
        obs_parts = split_observation(obs)

        core_feat = MLP((64, 64))(obs_parts["core"])
        lidar_feat = DirectionAwareLidarEncoder(
            max_dist=self.lidar_max_dist,
            out_dim=32,
        )(obs_parts["lidar"])
        obstacle_feat = MaskedObstacleEncoder(
            out_dim=64,
            max_obstacles=MAX_OBSTACLES,
        )(
            obs_parts["obstacle_rel"],
            obs_parts["obstacle_mask"],
            obs_parts["num_active"],
        )

        h = jnp.concatenate([core_feat, lidar_feat, obstacle_feat, action], axis=-1)
        h = MLP(self.trunk_widths)(h)

        mean = nn.Dense(self.target_dim, name="mean_head")(h)
        raw_logvar = nn.Dense(self.target_dim, name="logvar_head")(h)

        max_logvar = self.param(
            "max_logvar", nn.initializers.constant(0.5), (self.target_dim,)
        )
        min_logvar = self.param(
            "min_logvar", nn.initializers.constant(-10.0), (self.target_dim,)
        )
        max_logvar = jnp.clip(
            max_logvar,
            LOGVAR_LOWER_BOUND + LOGVAR_BOUND_EPS,
            LOGVAR_UPPER_BOUND,
        )
        min_logvar = jnp.clip(
            min_logvar,
            LOGVAR_LOWER_BOUND,
            LOGVAR_UPPER_BOUND - LOGVAR_BOUND_EPS,
        )
        max_logvar = jnp.maximum(max_logvar, min_logvar + LOGVAR_BOUND_EPS)

        logvar = max_logvar - nn.softplus(max_logvar - raw_logvar)
        logvar = min_logvar + nn.softplus(logvar - min_logvar)
        return mean, logvar


def load_arrays(data_path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(data_path)
    return {
        "obs": data["obs"],
        "applied_action": data["applied_action"],
        "next_obs": data["next_obs"],
        "reward": data["reward"],
    }


def split_env_indices(
    num_envs: int,
    train_ratio: float = 0.9,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if num_envs <= 1:
        raise ValueError("num_envs must be greater than 1.")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1).")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_envs)
    n_train = int(num_envs * train_ratio)
    n_train = min(max(n_train, 1), num_envs - 1)
    return perm[:n_train], perm[n_train:]


def _finalize_std(var: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.sqrt(np.maximum(var, eps)).astype(np.float64)


def compute_normalization_stats(
    arrays: dict[str, np.ndarray],
    env_idx: np.ndarray,
    chunk_envs: int = STATS_CHUNK_ENVS,
) -> dict[str, np.ndarray]:
    if chunk_envs <= 0:
        raise ValueError("chunk_envs must be positive.")

    obs_sum = np.zeros((OBS_DIM,), dtype=np.float64)
    obs_sumsq = np.zeros((OBS_DIM,), dtype=np.float64)
    act_sum = np.zeros((ACTION_DIM,), dtype=np.float64)
    act_sumsq = np.zeros((ACTION_DIM,), dtype=np.float64)
    delta_sum = np.zeros((OBS_DIM,), dtype=np.float64)
    delta_sumsq = np.zeros((OBS_DIM,), dtype=np.float64)
    reward_sum = np.zeros((1,), dtype=np.float64)
    reward_sumsq = np.zeros((1,), dtype=np.float64)
    total_count = 0

    for start in range(0, len(env_idx), chunk_envs):
        chunk_idx = env_idx[start : start + chunk_envs]
        obs = np.asarray(arrays["obs"][chunk_idx], dtype=np.float64)
        action = np.asarray(arrays["applied_action"][chunk_idx], dtype=np.float64)
        next_obs = np.asarray(arrays["next_obs"][chunk_idx], dtype=np.float64)
        reward = np.asarray(arrays["reward"][chunk_idx], dtype=np.float64)[..., None]
        delta = next_obs - obs

        obs_sum += obs.sum(axis=(0, 1))
        obs_sumsq += np.square(obs).sum(axis=(0, 1))
        act_sum += action.sum(axis=(0, 1))
        act_sumsq += np.square(action).sum(axis=(0, 1))
        delta_sum += delta.sum(axis=(0, 1))
        delta_sumsq += np.square(delta).sum(axis=(0, 1))
        reward_sum += reward.sum(axis=(0, 1))
        reward_sumsq += np.square(reward).sum(axis=(0, 1))
        total_count += obs.shape[0] * obs.shape[1]

    if total_count <= 0:
        raise ValueError("No transitions available to compute normalization stats.")

    obs_mean = obs_sum / total_count
    act_mean = act_sum / total_count
    delta_mean = delta_sum / total_count
    reward_mean = reward_sum / total_count

    obs_var = (obs_sumsq / total_count) - np.square(obs_mean)
    act_var = (act_sumsq / total_count) - np.square(act_mean)
    delta_var = (delta_sumsq / total_count) - np.square(delta_mean)
    reward_var = (reward_sumsq / total_count) - np.square(reward_mean)

    x_mean = np.concatenate([obs_mean, act_mean], axis=0).astype(np.float32)
    x_std = np.concatenate(
        [_finalize_std(obs_var), _finalize_std(act_var)],
        axis=0,
    ).astype(np.float32)
    y_mean = np.concatenate([delta_mean, reward_mean], axis=0).astype(np.float32)
    y_std = np.concatenate(
        [_finalize_std(delta_var), _finalize_std(reward_var)],
        axis=0,
    ).astype(np.float32)

    return {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }


class TransitionBatchLoader:
    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        env_idx: np.ndarray,
        batch_size: int,
        shuffle: bool,
        seed: int,
        norm_stats: dict[str, np.ndarray],
    ) -> None:
        self.obs = np.asarray(arrays["obs"], dtype=np.float32)
        self.action = np.asarray(arrays["applied_action"], dtype=np.float32)
        self.next_obs = np.asarray(arrays["next_obs"], dtype=np.float32)
        self.reward = np.asarray(arrays["reward"], dtype=np.float32)
        self.env_idx = np.asarray(env_idx, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.rng = np.random.default_rng(seed)
        self.x_mean = np.asarray(norm_stats["x_mean"], dtype=np.float32)
        self.x_std = np.asarray(norm_stats["x_std"], dtype=np.float32)
        self.y_mean = np.asarray(norm_stats["y_mean"], dtype=np.float32)
        self.y_std = np.asarray(norm_stats["y_std"], dtype=np.float32)

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.obs.ndim != 3 or self.action.ndim != 3 or self.next_obs.ndim != 3:
            raise ValueError("obs, action, and next_obs must all have shape (B, H, D).")
        if self.reward.shape != self.obs.shape[:2]:
            raise ValueError("reward must have shape (B, H).")

        self.horizon = int(self.obs.shape[1])
        self.size = int(self.env_idx.shape[0] * self.horizon)
        self.num_batches = int(np.ceil(self.size / self.batch_size))
        self.input_dim = int(self.obs.shape[-1] + self.action.shape[-1])
        self.target_dim = int(self.obs.shape[-1] + 1)

    def __len__(self) -> int:
        return self.num_batches

    def _gather_batch(self, batch_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        env_offsets = batch_ids // self.horizon
        step_idx = batch_ids % self.horizon
        env_ids = self.env_idx[env_offsets]

        obs = self.obs[env_ids, step_idx]
        action = self.action[env_ids, step_idx]
        next_obs = self.next_obs[env_ids, step_idx]
        reward = self.reward[env_ids, step_idx][:, None]

        x = np.concatenate([obs, action], axis=-1)
        y = np.concatenate([next_obs - obs, reward], axis=-1)

        x = (x - self.x_mean) / self.x_std
        y = (y - self.y_mean) / self.y_std
        return x.astype(np.float32, copy=False), y.astype(np.float32, copy=False)

    def sample_batch(self) -> tuple[np.ndarray, np.ndarray]:
        batch_ids = self.rng.integers(0, self.size, size=self.batch_size, dtype=np.int64)
        return self._gather_batch(batch_ids)

    def __iter__(self):
        if self.shuffle:
            # Avoid materializing a full permutation over millions of transitions.
            for _ in range(self.num_batches):
                batch_ids = self.rng.integers(0, self.size, size=self.batch_size, dtype=np.int64)
                yield self._gather_batch(batch_ids)
            return

        for start in range(0, self.size, self.batch_size):
            stop = min(start + self.batch_size, self.size)
            batch_ids = np.arange(start, stop, dtype=np.int64)
            yield self._gather_batch(batch_ids)


def build_ensemble_loaders(
    arrays: dict[str, np.ndarray],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    ensemble_size: int,
    batch_size: int,
    seed: int,
    norm_stats: dict[str, np.ndarray],
) -> tuple[list[TransitionBatchLoader], TransitionBatchLoader]:
    if ensemble_size <= 0:
        raise ValueError("ensemble_size must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    rng = np.random.default_rng(seed)
    train_loaders: list[TransitionBatchLoader] = []

    for member_id in range(ensemble_size):
        member_idx = rng.choice(train_idx, size=len(train_idx), replace=True)
        loader = TransitionBatchLoader(
            arrays=arrays,
            env_idx=member_idx,
            batch_size=batch_size,
            shuffle=True,
            seed=seed + member_id + 1,
            norm_stats=norm_stats,
        )
        train_loaders.append(loader)

    test_loader = TransitionBatchLoader(
        arrays=arrays,
        env_idx=test_idx,
        batch_size=batch_size,
        shuffle=False,
        seed=seed + 10_000,
        norm_stats=norm_stats,
    )
    return train_loaders, test_loader


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare PETS ensemble training loaders.")
    parser.add_argument(
        "--data_path",
        type=str,
        default=DEFAULT_DATA_PATH,
    )
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ensemble_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--steps_per_epoch", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--val_batches", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--save_dir",
        type=str,
        default="jax_implementation/MBRL/checkpoints/pets_pretrain",
    )
    return parser

##Ensembling
class EnsembleDynamics(nn.Module):
    ensemble_size: int = 8

    @nn.compact
    def __call__(self, x):
        VmappedModel = nn.vmap(
            Dynamics_Model,
            variable_axes={"params": 0},  
            split_rngs={"params": True},  
            in_axes=0,                     
            out_axes=0,
            axis_size=self.ensemble_size,
        )
        return VmappedModel()(x)


def _logvar_reg(params) -> jnp.ndarray:
    model_params = params["VmapDynamics_Model_0"]
    max_logvar = model_params["max_logvar"]
    min_logvar = model_params["min_logvar"]
    return LOGVAR_REG_COEF * (jnp.sum(max_logvar) - jnp.sum(min_logvar))


@jax.jit
def train_step(state, batch_x, batch_y, y_std):
    def loss_fn(params):
        mean, logvar = state.apply_fn({"params": params}, batch_x)  # (E,B,97), (E,B,97)
        inv_var = jnp.exp(-logvar)
        nll = jnp.sum((mean - batch_y) ** 2 * inv_var + logvar, axis=-1)
        loss_per_model = jnp.mean(nll, axis=-1)   # (E,)
        loss = jnp.mean(loss_per_model)           # scalar
        logvar_reg = _logvar_reg(params)
        objective = loss + logvar_reg
        mse_per_model = jnp.mean(jnp.square(mean - batch_y), axis=(1, 2))
        raw_error = (mean - batch_y) * y_std[None, None, :]
        raw_mse_per_model = jnp.mean(jnp.square(raw_error), axis=(1, 2))
        raw_delta_mse_per_model = jnp.mean(jnp.square(raw_error[..., :OBS_DIM]), axis=(1, 2))
        raw_reward_mse_per_model = jnp.mean(jnp.square(raw_error[..., OBS_DIM:]), axis=(1, 2))
        metrics = {
            "objective": objective,
            "loss": loss,
            "logvar_reg": logvar_reg,
            "loss_per_model": loss_per_model,
            "mse": jnp.mean(mse_per_model),
            "raw_mse": jnp.mean(raw_mse_per_model),
            "raw_delta_mse": jnp.mean(raw_delta_mse_per_model),
            "raw_reward_mse": jnp.mean(raw_reward_mse_per_model),
        }
        return objective, metrics

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, metrics


@partial(jax.jit, static_argnames=("apply_fn",))
def eval_step(params, apply_fn, batch_x, batch_y, y_std):
    mean, logvar = apply_fn({"params": params}, batch_x)
    inv_var = jnp.exp(-logvar)
    nll = jnp.sum((mean - batch_y) ** 2 * inv_var + logvar, axis=-1)
    loss_per_model = jnp.mean(nll, axis=-1)
    loss = jnp.mean(loss_per_model)
    logvar_reg = _logvar_reg(params)
    mse_per_model = jnp.mean(jnp.square(mean - batch_y), axis=(1, 2))
    raw_error = (mean - batch_y) * y_std[None, None, :]
    raw_mse_per_model = jnp.mean(jnp.square(raw_error), axis=(1, 2))
    raw_delta_mse_per_model = jnp.mean(jnp.square(raw_error[..., :OBS_DIM]), axis=(1, 2))
    raw_reward_mse_per_model = jnp.mean(jnp.square(raw_error[..., OBS_DIM:]), axis=(1, 2))
    metrics = {
        "objective": loss + logvar_reg,
        "loss": loss,
        "logvar_reg": logvar_reg,
        "loss_per_model": loss_per_model,
        "mse": jnp.mean(mse_per_model),
        "raw_mse": jnp.mean(raw_mse_per_model),
        "raw_delta_mse": jnp.mean(raw_delta_mse_per_model),
        "raw_reward_mse": jnp.mean(raw_reward_mse_per_model),
    }
    return metrics


def save_params(path: Path, params) -> None:
    path.write_bytes(serialization.to_bytes(params))


def to_float_list(x) -> list[float]:
    return [float(v) for v in np.asarray(x, dtype=np.float32).tolist()]


def assert_finite_metrics(metrics: dict[str, object], context: str) -> None:
    bad = []
    for key, value in metrics.items():
        arr = np.asarray(value)
        if not np.all(np.isfinite(arr)):
            bad.append(f"{key}={arr}")
    if bad:
        raise FloatingPointError(f"Non-finite metrics at {context}: " + ", ".join(bad))


if __name__ == "__main__":
    args = _build_parser().parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    arrays = load_arrays(args.data_path)
    num_envs, horizon, obs_dim = arrays["obs"].shape
    action_dim = arrays["applied_action"].shape[-1]

    train_idx, test_idx = split_env_indices(
        num_envs=num_envs,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    norm_stats = compute_normalization_stats(arrays, train_idx)
    train_loaders, test_loader = build_ensemble_loaders(
        arrays=arrays,
        train_idx=train_idx,
        test_idx=test_idx,
        ensemble_size=args.ensemble_size,
        batch_size=args.batch_size,
        seed=args.seed,
        norm_stats=norm_stats,
    )

    print("dataset obs shape:", arrays["obs"].shape)
    print("dataset action shape:", arrays["applied_action"].shape)
    print("train_idx shape:", train_idx.shape)
    print("test_idx shape:", test_idx.shape)
    print("logical train input shape:", (train_idx.shape[0], horizon, obs_dim + action_dim))
    print("logical train target shape:", (train_idx.shape[0], horizon, obs_dim + 1))
    print("ensemble loaders:", len(train_loaders))

    first_batch_x, first_batch_y = next(iter(train_loaders[0]))
    print("first member batch x:", first_batch_x.shape)
    print("first member batch y:", first_batch_y.shape)

    test_batch_x, test_batch_y = next(iter(test_loader))
    print("test batch x:", test_batch_x.shape)
    print("test batch y:", test_batch_y.shape)
    E = args.ensemble_size
    model = EnsembleDynamics(ensemble_size=E)
    y_std = jnp.asarray(norm_stats["y_std"], dtype=jnp.float32)

    if args.steps_per_epoch <= 0:
        raise ValueError("--steps_per_epoch must be positive.")
    if args.eval_every <= 0:
        raise ValueError("--eval_every must be positive.")
    
    dummy_x = jnp.zeros((E, 1, obs_dim + action_dim), dtype=jnp.float32)
    params = model.init(jax.random.PRNGKey(0), dummy_x)["params"]

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.chain(
            optax.clip_by_global_norm(GRAD_CLIP_NORM),
            optax.adam(args.lr),
        ),
    )
    np.save(save_dir / "train_idx.npy", train_idx)
    np.save(save_dir / "test_idx.npy", test_idx)
    np.savez(save_dir / "normalization_stats.npz", **norm_stats)

    best_val_loss = float("inf")
    best_epoch = -1
    history: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss_sum = 0.0
        train_loss_per_model_sum = np.zeros((E,), dtype=np.float64)
        train_mse_sum = 0.0
        train_raw_mse_sum = 0.0
        train_raw_delta_mse_sum = 0.0
        train_raw_reward_mse_sum = 0.0
        num_train_batches = 0

        for _ in range(args.steps_per_epoch):
            member_batches = [loader.sample_batch() for loader in train_loaders]
            batch_x = jnp.asarray(np.stack([bx for bx, _ in member_batches], axis=0), dtype=jnp.float32)
            batch_y = jnp.asarray(np.stack([by for _, by in member_batches], axis=0), dtype=jnp.float32)

            state, metrics = train_step(state, batch_x, batch_y, y_std)
            metrics = jax.device_get(metrics)
            assert_finite_metrics(metrics, f"train epoch={epoch} step={num_train_batches + 1}")
            train_loss_sum += float(metrics["loss"])
            train_loss_per_model_sum += np.asarray(metrics["loss_per_model"], dtype=np.float64)
            train_mse_sum += float(metrics["mse"])
            train_raw_mse_sum += float(metrics["raw_mse"])
            train_raw_delta_mse_sum += float(metrics["raw_delta_mse"])
            train_raw_reward_mse_sum += float(metrics["raw_reward_mse"])
            num_train_batches += 1

        mean_train_loss = train_loss_sum / max(num_train_batches, 1)
        mean_train_loss_per_model = train_loss_per_model_sum / max(num_train_batches, 1)
        mean_train_mse = train_mse_sum / max(num_train_batches, 1)
        mean_train_raw_mse = train_raw_mse_sum / max(num_train_batches, 1)
        mean_train_raw_delta_mse = train_raw_delta_mse_sum / max(num_train_batches, 1)
        mean_train_raw_reward_mse = train_raw_reward_mse_sum / max(num_train_batches, 1)

        should_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        mean_val_loss = None
        mean_val_loss_per_model = None
        mean_val_mse = None
        mean_val_raw_mse = None
        mean_val_raw_delta_mse = None
        mean_val_raw_reward_mse = None

        if should_eval:
            val_loss_sum = 0.0
            val_loss_per_model_sum = np.zeros((E,), dtype=np.float64)
            val_mse_sum = 0.0
            val_raw_mse_sum = 0.0
            val_raw_delta_mse_sum = 0.0
            val_raw_reward_mse_sum = 0.0
            num_val_batches = 0

            for batch_id, (bx, by) in enumerate(test_loader, start=1):
                bx = jnp.asarray(bx, dtype=jnp.float32)
                by = jnp.asarray(by, dtype=jnp.float32)

                bx = jnp.broadcast_to(bx[None], (E, *bx.shape))
                by = jnp.broadcast_to(by[None], (E, *by.shape))

                val_metrics = eval_step(state.params, state.apply_fn, bx, by, y_std)
                val_metrics = jax.device_get(val_metrics)
                assert_finite_metrics(val_metrics, f"val epoch={epoch} batch={batch_id}")
                val_loss_sum += float(val_metrics["loss"])
                val_loss_per_model_sum += np.asarray(val_metrics["loss_per_model"], dtype=np.float64)
                val_mse_sum += float(val_metrics["mse"])
                val_raw_mse_sum += float(val_metrics["raw_mse"])
                val_raw_delta_mse_sum += float(val_metrics["raw_delta_mse"])
                val_raw_reward_mse_sum += float(val_metrics["raw_reward_mse"])
                num_val_batches += 1

                if args.val_batches > 0 and batch_id >= args.val_batches:
                    break

            mean_val_loss = val_loss_sum / max(num_val_batches, 1)
            mean_val_loss_per_model = val_loss_per_model_sum / max(num_val_batches, 1)
            mean_val_mse = val_mse_sum / max(num_val_batches, 1)
            mean_val_raw_mse = val_raw_mse_sum / max(num_val_batches, 1)
            mean_val_raw_delta_mse = val_raw_delta_mse_sum / max(num_val_batches, 1)
            mean_val_raw_reward_mse = val_raw_reward_mse_sum / max(num_val_batches, 1)

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(mean_train_loss),
            "train_loss_per_model": to_float_list(mean_train_loss_per_model),
            "train_mse": float(mean_train_mse),
            "train_raw_mse": float(mean_train_raw_mse),
            "train_raw_delta_mse": float(mean_train_raw_delta_mse),
            "train_raw_reward_mse": float(mean_train_raw_reward_mse),
            "val_loss": None if mean_val_loss is None else float(mean_val_loss),
            "val_loss_per_model": None if mean_val_loss_per_model is None else to_float_list(mean_val_loss_per_model),
            "val_mse": None if mean_val_mse is None else float(mean_val_mse),
            "val_raw_mse": None if mean_val_raw_mse is None else float(mean_val_raw_mse),
            "val_raw_delta_mse": None if mean_val_raw_delta_mse is None else float(mean_val_raw_delta_mse),
            "val_raw_reward_mse": None if mean_val_raw_reward_mse is None else float(mean_val_raw_reward_mse),
        }
        history.append(epoch_record)

        if should_eval:
            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"train_nll={mean_train_loss:.4f} | "
                f"val_nll={mean_val_loss:.4f} | "
                f"train_raw_mse={mean_train_raw_mse:.6f} | "
                f"val_raw_mse={mean_val_raw_mse:.6f} | "
                f"train_raw_delta_mse={mean_train_raw_delta_mse:.6f} | "
                f"val_raw_delta_mse={mean_val_raw_delta_mse:.6f}"
            )

            if mean_val_loss < best_val_loss:
                best_val_loss = float(mean_val_loss)
                best_epoch = epoch
                save_params(save_dir / "best_params.msgpack", state.params)
        else:
            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"train_nll={mean_train_loss:.4f} | "
                f"train_raw_mse={mean_train_raw_mse:.6f} | "
                f"train_raw_delta_mse={mean_train_raw_delta_mse:.6f} | "
                f"val=skipped"
            )

    save_params(save_dir / "final_params.msgpack", state.params)
    metrics_payload = {
        "data_path": str(args.data_path),
        "train_ratio": float(args.train_ratio),
        "seed": int(args.seed),
        "ensemble_size": int(args.ensemble_size),
        "batch_size": int(args.batch_size),
        "steps_per_epoch": int(args.steps_per_epoch),
        "epochs": int(args.epochs),
        "eval_every": int(args.eval_every),
        "val_batches": int(args.val_batches),
        "lr": float(args.lr),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "history": history,
    }
    metrics_path = save_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8")

    print(f"Saved best params to {save_dir / 'best_params.msgpack'}")
    print(f"Saved final params to {save_dir / 'final_params.msgpack'}")
    print(f"Saved metrics to {metrics_path}")
