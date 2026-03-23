from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_eval_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    timesteps = np.asarray(data["timesteps"], dtype=np.float64)
    results = np.asarray(data["results"], dtype=np.float64)
    return timesteps, results


def summarize(results: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = results.mean(axis=1)
    std = results.std(axis=1)
    return mean, std


def main():
    parser = argparse.ArgumentParser(
        description="Plot PPO vs SAC mean return curves from saved evaluation artifacts."
    )
    parser.add_argument(
        "--ppo",
        type=Path,
        default=Path("drone_training/artifacts/eval/evaluations.npz"),
        help="Path to PPO evaluations.npz",
    )
    parser.add_argument(
        "--sac",
        type=Path,
        default=Path("drone_training/artifacts_sac/eval/evaluations.npz"),
        help="Path to SAC evaluations.npz",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("drone_training/artifacts/ppo_vs_sac_comparison.png"),
        help="Output path for the comparison figure",
    )
    parser.add_argument(
        "--random-baseline",
        type=float,
        default=-120.0,
        help="Constant random-agent return to draw as a horizontal baseline",
    )
    args = parser.parse_args()

    ppo_steps, ppo_results = load_eval_npz(args.ppo)
    sac_steps, sac_results = load_eval_npz(args.sac)

    ppo_mean, ppo_std = summarize(ppo_results)
    sac_mean, sac_std = summarize(sac_results)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    plt.plot(ppo_steps, ppo_mean, label="PPO (on-policy)", color="#0f766e", linewidth=2.5)
    plt.fill_between(
        ppo_steps,
        ppo_mean - ppo_std,
        ppo_mean + ppo_std,
        color="#0f766e",
        alpha=0.18,
    )

    plt.plot(sac_steps, sac_mean, label="SAC (off-policy)", color="#b45309", linewidth=2.5)
    plt.fill_between(
        sac_steps,
        sac_mean - sac_std,
        sac_mean + sac_std,
        color="#b45309",
        alpha=0.18,
    )

    plt.axhline(
        args.random_baseline,
        color="#475569",
        linestyle="--",
        linewidth=2.0,
        label=f"Random baseline ({args.random_baseline:.1f})",
    )

    plt.title("Mean Return vs Training Environment Steps")
    plt.xlabel("Training environment steps")
    plt.ylabel("Mean return")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)

    print(f"Saved comparison plot to: {args.out.resolve()}")
    print(f"PPO mean returns: {np.round(ppo_mean, 3)}")
    print(f"SAC mean returns: {np.round(sac_mean, 3)}")


if __name__ == "__main__":
    main()
