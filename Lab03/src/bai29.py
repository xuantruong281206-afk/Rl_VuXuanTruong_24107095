"""Bai 29 - So sánh epsilon"""
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
    rows = epsilon_experiment(env, epsilons=(0.01, 0.05, 0.10, 0.20, 0.50), n_train=100_000, n_eval=20_000)
    print(f"{'epsilon':>8s} {'greedy_freq':>12s} {'lý thuyết':>10s} {'mean reward (learned)':>22s}")
    for r in rows:
        print(f"{r['epsilon']:8.2f} {r['greedy_freq']:12.3f} {r['theory']:10.3f} {r['mean_reward']:14.4f} ± {1.96 * r['std_error']:.4f}")
    env.close()


if __name__ == "__main__":
    main()
