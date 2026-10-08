"""Bai 24 - Lưu state-action-return"""
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
    sa_returns = defaultdict(list)
    for i in range(20000):
        ep = generate_episode(env, random_policy, seed=1 if i == 0 else None)
        G = compute_returns([r for _, _, r in ep], gamma=1.0)
        seen = set()
        for t, (state, action, _) in enumerate(ep):
            if (state, action) not in seen:          # first visit trên cặp (s, a)
                seen.add((state, action))
                sa_returns[(state, action)].append(G[t])
    print("Số cặp (state, action):", len(sa_returns))
    for key in [((20, 10, False), 0), ((20, 10, False), 1), ((13, 2, False), 1)]:
        r = sa_returns[key]
        print(f"{key}: n={len(r)}, mean return = {np.mean(r):+.3f}")
    env.close()


if __name__ == "__main__":
    main()
