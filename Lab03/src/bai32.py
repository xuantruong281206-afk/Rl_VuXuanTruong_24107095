"""Bai 32 - On-policy First-Visit MC Control"""
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
    Q, policy, episode_rewards = on_policy_mc_control(env, n_episodes=20_000, gamma=1.0, epsilon=0.1, seed=42)
    print("Số state đã học   :", len(policy))
    print("Mean reward 1000 episode cuối (lúc train, có explore):", np.mean(episode_rewards[-1000:]))
    env.close()


if __name__ == "__main__":
    main()
