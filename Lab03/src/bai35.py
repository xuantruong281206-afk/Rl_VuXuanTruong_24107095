"""Bai 35 - Learning curve"""
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
    Q, policy, episode_rewards = load_or_train_control(env, n_episodes=300_000, epsilon=0.1, seed=42)
    window = 1000
    ma = moving_average(episode_rewards, window)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(len(ma)) + window, ma, label="moving average (window=1000)")
    ax.axhline(np.mean(episode_rewards[:window]), color="r", ls=":", label="1000 episode đầu")
    ax.set_xlabel("episode"); ax.set_ylabel("mean reward")
    ax.set_title("On-policy MC control: learning curve (epsilon=0.1)")
    ax.legend(); ax.grid(alpha=0.3)
    save_fig(fig, "mc_convergence.png")
    print(f"1000 episode đầu : {np.mean(episode_rewards[:1000]):+.3f}")
    print(f"1000 episode cuối: {np.mean(episode_rewards[-1000:]):+.3f}   (lúc train vẫn còn explore ε=0.1)")
    env.close()


if __name__ == "__main__":
    main()
