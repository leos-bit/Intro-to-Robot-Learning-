import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RANDOM_POLICY_MEAN_RETURN = -392.51


def _find_csv(base_dir: Path, candidates: tuple[str, ...]) -> Path:
    for name in candidates:
        path = base_dir / name
        if path.exists():
            return path

    by_name = {path.name.casefold(): path for path in base_dir.glob("*.csv")}
    for name in candidates:
        path = by_name.get(name.casefold())
        if path is not None:
            return path

    raise FileNotFoundError(f"Missing CSV. Tried: {', '.join(candidates)}")


def _load_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows: list[tuple[float, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append((float(row["Step"]), float(row["Value"])))

    if not rows:
        raise ValueError(f"{path.name} is empty.")

    rows.sort(key=lambda item: item[0])
    steps, rewards = zip(*rows)
    return np.asarray(steps, dtype=float), np.asarray(rewards, dtype=float)


def _match_final_step(steps: np.ndarray, target_final_step: float) -> np.ndarray:
    final_step = float(steps[-1])
    if final_step <= 0:
        return steps.copy()
    return steps * (target_final_step / final_step)


def main() -> None:
    base_dir = Path(__file__).parent
    ppo_path = _find_csv(base_dir, ("pppo.csv", "ppo.csv"))
    sac_path = _find_csv(base_dir, ("sac.csv", "SAc.csv"))

    ppo_steps, ppo_rewards = _load_curve(ppo_path)
    sac_steps, sac_rewards = _load_curve(sac_path)

    matched_final_step = max(float(ppo_steps[-1]), float(sac_steps[-1]))
    ppo_steps = _match_final_step(ppo_steps, matched_final_step)
    sac_steps = _match_final_step(sac_steps, matched_final_step)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ppo_steps, ppo_rewards, label="PPO", linewidth=2.5, color="#1f77b4")
    ax.plot(sac_steps, sac_rewards, label="SAC", linewidth=2.5, color="#d62728")
    ax.axhline(
        RANDOM_POLICY_MEAN_RETURN,
        label="Random Policy",
        linewidth=2.0,
        linestyle="--",
        color="#2ca02c",
    )
    ax.set_title("Mean Return  vs Steps")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Return ")
    ax.grid(True, alpha=0.3)
    ax.legend()

    output_path = base_dir / "reward_curves.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)

    print(f"PPO source: {ppo_path.name}")
    print(f"SAC source: {sac_path.name}")
    print(f"Random-policy baseline: {RANDOM_POLICY_MEAN_RETURN:.2f}")
    print(f"Matched final step: {int(matched_final_step)}")
    print(f"Saved plot to {output_path.name}")

    plt.show()


if __name__ == "__main__":
    main()
