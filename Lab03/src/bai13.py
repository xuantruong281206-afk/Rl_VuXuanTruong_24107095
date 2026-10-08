"""Bai 13 - Thu thập return theo state"""
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
    print("Số state đã gặp:", len(returns))
    example = (20, 10, False)
    print(f"returns[{example}]: {len(returns[example])} mẫu, 10 mẫu đầu = {returns[example][:10]}")
    env.close()


if __name__ == "__main__":
    main()
