import argparse
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Dataset


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
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

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
