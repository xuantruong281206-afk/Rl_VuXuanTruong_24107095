"""Bai 33 - Học policy cho Blackjack"""
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
    Q, policy, episode_rewards = load_or_train_control(env, n_episodes=300_000, gamma=1.0, epsilon=0.1, seed=42)
    print("Số state đã học:", len(policy))
    for s in [(20, 10, False), (18, 6, False), (13, 2, False), (18, 6, True)]:
        if s in policy:
            print(f"state={s}: Q={np.round(Q[s], 3)} -> {ACTION_NAMES[policy[s]]}")
        else:
            print(f"state={s}: chưa gặp trong lúc train")
    env.close()


if __name__ == "__main__":
    main()
