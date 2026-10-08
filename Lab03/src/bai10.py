"""Bai 10 - So sánh gamma"""
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
    rng = np.random.default_rng(0)
    random_policy = make_random_policy(env.action_space.n, rng)
    all_rewards = []
    for i in range(5000):
        ep = generate_episode(env, random_policy, seed=0 if i == 0 else None)
        all_rewards.append([r for _, _, r in ep])

    gammas = [0.5, 0.8, 0.9, 0.99, 1.0]
    mean_g0 = []
    for gamma in gammas:
        g0 = [compute_returns(rw, gamma)[0] for rw in all_rewards]
        mean_g0.append(np.mean(g0))
        print(f"gamma={gamma:4}: mean G_0 = {mean_g0[-1]:.4f}")

    # Reward chỉ xuất hiện ở bước cuối nên G_0 = gamma^(T-1) * R_T. Episode Blackjack rất ngắn
    # (1-3 bước) nên gamma ảnh hưởng ít; gamma nhỏ chỉ làm return "co" về 0.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(gammas, mean_g0, "o-")
    ax.set_xlabel("gamma"); ax.set_ylabel("mean G_0"); ax.set_title("Mean G_0 theo gamma (random policy)")
    ax.grid(alpha=0.3)
    save_fig(fig, "gamma_comparison.png")
    env.close()


if __name__ == "__main__":
    main()
