"""Bai 25 - Ước lượng Q(s,a)"""
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
    random_policy = make_random_policy(env.action_space.n, np.random.default_rng(1))
    Q, N = mc_action_value_prediction(env, random_policy, n_episodes=100_000, gamma=1.0, seed=1)
    print("Số state có Q:", len(Q))
    for s in [(20, 10, False), (13, 2, False), (18, 6, True), (12, 6, False)]:
        print(f"Q{s} = [stick={Q[s][0]:+.3f}, hit={Q[s][1]:+.3f}]   N = {N[s]}")
    env.close()


if __name__ == "__main__":
    main()
