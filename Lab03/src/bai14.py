"""Bai 14 - Ước lượng V(s)"""
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
    returns = defaultdict(list)
    for i in range(10000):
        ep = generate_episode(env, stick_on_20_policy, seed=0 if i == 0 else None)
        G = compute_returns([r for _, _, r in ep], gamma=1.0)
        for (state, _, _), g in zip(ep, G):
            returns[state].append(g)
    V = {state: np.mean(g_list) for state, g_list in returns.items()}
    top10 = sorted(returns, key=lambda s: len(returns[s]), reverse=True)[:10]
    for s in top10:
        print(f"V{s} = {V[s]:+.3f}   (từ {len(returns[s])} return)")
    env.close()


if __name__ == "__main__":
    main()
