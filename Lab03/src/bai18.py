"""Bai 18 - Chạy First-Visit MC"""
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
    for n in [100, 1000, 10000, 50000]:
        V, returns_count = first_visit_mc_prediction(env, stick_on_20_policy, n_episodes=n, seed=0)
        print(f"{n:6d} episode -> {len(V):3d} state được ước lượng, state ít mẫu nhất có n={min(returns_count.values())}")
    env.close()


if __name__ == "__main__":
    main()
