import argparse
import json
from functools import partial
from pathlib import Path

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from torch.utils.data import DataLoader, Dataset
from flax import serialization
from flax.training import train_state
import jax
OBS_DIM = 96
ACTION_DIM = 4
MAX_OBSTACLES = 15

import optax
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


class MaskedObstacleEncoder(nn.Module):
    out_dim: int = 64
    max_obstacles: int = MAX_OBSTACLES

    @nn.compact
    def __call__(self, obstacle_rel, obstacle_mask, num_active):
        mask = jnp.asarray(obstacle_mask, dtype=jnp.float32)[..., None]
        rel = jnp.asarray(obstacle_rel, dtype=jnp.float32)
        dist = jnp.linalg.norm(rel, axis=-1, keepdims=True)

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


class TransitionDataset(Dataset):
    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        env_idx: np.ndarray,
    ) -> None:
        self.obs = np.asarray(arrays["obs"], dtype=np.float32)
        self.action = np.asarray(arrays["applied_action"], dtype=np.float32)
        self.next_obs = np.asarray(arrays["next_obs"], dtype=np.float32)
        self.reward = np.asarray(arrays["reward"], dtype=np.float32)
        self.env_idx = np.asarray(env_idx, dtype=np.int64)

        if self.obs.ndim != 3:
            raise ValueError(f"Expected obs shape (B, H, D), got {self.obs.shape}.")
        if self.action.ndim != 3:
            raise ValueError(
                f"Expected applied_action shape (B, H, A), got {self.action.shape}."
            )
        if self.next_obs.shape != self.obs.shape:
            raise ValueError(
                "next_obs must have the same shape as obs: "
                f"{self.next_obs.shape} vs {self.obs.shape}."
            )
        if self.reward.shape != self.obs.shape[:2]:
            raise ValueError(
                "reward must have shape (B, H): "
                f"{self.reward.shape} vs {self.obs.shape[:2]}."
            )

        self.horizon = int(self.obs.shape[1])
        self.size = int(self.env_idx.shape[0] * self.horizon)
        self.input_dim = int(self.obs.shape[-1] + self.action.shape[-1])
        self.target_dim = int(self.obs.shape[-1] + 1)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        env_offset = idx // self.horizon
        step_idx = idx % self.horizon
        env_idx = int(self.env_idx[env_offset])

        obs = self.obs[env_idx, step_idx]
        action = self.action[env_idx, step_idx]
        next_obs = self.next_obs[env_idx, step_idx]
        reward = np.asarray([self.reward[env_idx, step_idx]], dtype=np.float32)

        x = np.concatenate([obs, action], axis=-1).astype(np.float32, copy=False)
        y = np.concatenate([next_obs - obs, reward], axis=-1).astype(np.float32, copy=False)
        return x, y


def collate_fn(batch: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    x, y = zip(*batch)
    return np.stack(x, axis=0), np.stack(y, axis=0)


def build_ensemble_loaders(
    arrays: dict[str, np.ndarray],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    ensemble_size: int,
    batch_size: int,
    seed: int,
) -> tuple[list[DataLoader], DataLoader]:
    if ensemble_size <= 0:
        raise ValueError("ensemble_size must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    rng = np.random.default_rng(seed)
    train_loaders: list[DataLoader] = []

    for _ in range(ensemble_size):
        member_idx = rng.choice(train_idx, size=len(train_idx), replace=True)
        member_dataset = TransitionDataset(arrays, env_idx=member_idx)
        loader = DataLoader(
            member_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
        )
        train_loaders.append(loader)

    test_dataset = TransitionDataset(arrays, env_idx=test_idx)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    return train_loaders, test_loader


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare PETS ensemble training loaders.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="jax_implementation/MBRL/dyn_data/pets_pretrain_envB1024_T10000_pid_noisy_seed0.npz",
    )
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ensemble_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=20)
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
@jax.jit
def train_step(state, batch_x, batch_y):


    def loss_fn(params):
        mean, logvar =    state.apply_fn({"params": params}, batch_x)  # (E,B,97), (E,B,97)
        
        inv_var = jnp.exp(-logvar)
        nll = jnp.sum((mean - batch_y) ** 2 *inv_var+logvar, axis=-1)
        loss_per_model = jnp.mean(nll, axis=-1)   # (E,)
        loss = jnp.mean(loss_per_model)           # scalar
        return loss, loss_per_model
    
    (loss, loss_per_model), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, {"loss": loss, "loss_per_model": loss_per_model}


@partial(jax.jit, static_argnames=("apply_fn",))
def eval_step(params, apply_fn, batch_x, batch_y):
    mean, logvar = apply_fn({"params": params}, batch_x)
    inv_var = jnp.exp(-logvar)
    nll = jnp.sum((mean - batch_y) ** 2 * inv_var + logvar, axis=-1)
    loss_per_model = jnp.mean(nll, axis=-1)
    loss = jnp.mean(loss_per_model)
    return loss, loss_per_model


def save_params(path: Path, params) -> None:
    path.write_bytes(serialization.to_bytes(params))


def to_float_list(x) -> list[float]:
    return [float(v) for v in np.asarray(x, dtype=np.float32).tolist()]

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
    train_loaders, test_loader = build_ensemble_loaders(
        arrays=arrays,
        train_idx=train_idx,
        test_idx=test_idx,
        ensemble_size=args.ensemble_size,
        batch_size=args.batch_size,
        seed=args.seed,
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
    
    dummy_x = jnp.zeros((E, 1, obs_dim + action_dim), dtype=jnp.float32)
    params = model.init(jax.random.PRNGKey(0), dummy_x)["params"]

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(args.lr),
    )
    np.save(save_dir / "train_idx.npy", train_idx)
    np.save(save_dir / "test_idx.npy", test_idx)

    best_val_loss = float("inf")
    best_epoch = -1
    best_params = state.params
    history: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss_sum = 0.0
        train_loss_per_model_sum = np.zeros((E,), dtype=np.float64)
        num_train_batches = 0

        for member_batches in zip(*train_loaders):
            batch_x = jnp.stack([jnp.asarray(bx) for bx, _ in member_batches], axis=0)
            batch_y = jnp.stack([jnp.asarray(by) for _, by in member_batches], axis=0)

            state, metrics = train_step(state, batch_x, batch_y)
            train_loss_sum += float(metrics["loss"])
            train_loss_per_model_sum += np.asarray(metrics["loss_per_model"], dtype=np.float64)
            num_train_batches += 1

        mean_train_loss = train_loss_sum / max(num_train_batches, 1)
        mean_train_loss_per_model = train_loss_per_model_sum / max(num_train_batches, 1)

        val_loss_sum = 0.0
        val_loss_per_model_sum = np.zeros((E,), dtype=np.float64)
        num_val_batches = 0

        for bx, by in test_loader:
            bx = jnp.asarray(bx)
            by = jnp.asarray(by)

            bx = jnp.broadcast_to(bx[None], (E, *bx.shape))
            by = jnp.broadcast_to(by[None], (E, *by.shape))

            val_loss, val_loss_per_model = eval_step(state.params, state.apply_fn, bx, by)
            val_loss_sum += float(val_loss)
            val_loss_per_model_sum += np.asarray(val_loss_per_model, dtype=np.float64)
            num_val_batches += 1

        mean_val_loss = val_loss_sum / max(num_val_batches, 1)
        mean_val_loss_per_model = val_loss_per_model_sum / max(num_val_batches, 1)

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(mean_train_loss),
            "train_loss_per_model": to_float_list(mean_train_loss_per_model),
            "val_loss": float(mean_val_loss),
            "val_loss_per_model": to_float_list(mean_val_loss_per_model),
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={mean_train_loss:.4f} | "
            f"val_loss={mean_val_loss:.4f}"
        )

        if mean_val_loss < best_val_loss:
            best_val_loss = float(mean_val_loss)
            best_epoch = epoch
            best_params = state.params
            save_params(save_dir / "best_params.msgpack", best_params)

    save_params(save_dir / "final_params.msgpack", state.params)
    metrics_payload = {
        "data_path": str(args.data_path),
        "train_ratio": float(args.train_ratio),
        "seed": int(args.seed),
        "ensemble_size": int(args.ensemble_size),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
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
