"""Bai 15 - Theo dõi một state cụ thể"""
import os, sys
os.environ.setdefault("MPLBACKEND", "Agg")            # luu hinh, khong mo cua so
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from mc_utils import *
from experiments import *


def main():
    env = make_env("Blackjack-v1")
    target_state = (20, 10, False)
    checkpoints = [100, 500, 1000, 5000, 10000]
    total, count, estimates = 0.0, 0, []
    for i in range(1, 10001):
        ep = generate_episode(env, stick_on_20_policy, seed=0 if i == 1 else None)
        G = compute_returns([r for _, _, r in ep], gamma=1.0)
        for (state, _, _), g in zip(ep, G):
            if state == target_state:
                total += g
                count += 1
        if i in checkpoints:
            estimates.append(total / count if count else np.nan)
            print(f"sau {i:5d} episode: V{target_state} ≈ {estimates[-1]:+.4f}  (n={count})")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(checkpoints, estimates, "o-")
    ax.set_xscale("log"); ax.set_xlabel("số episode"); ax.set_ylabel("V(s) ước lượng")
    ax.set_title(f"Hội tụ của V{target_state}"); ax.grid(alpha=0.3)
    save_fig(fig, "convergence_target_state.png")
    env.close()


if __name__ == "__main__":
    main()
