"""Bai 27 - Epsilon-greedy policy"""
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
    rng = np.random.default_rng(0)
    Q = defaultdict(lambda: np.zeros(2))
    Q[(18, 6, False)] = np.array([0.15, -0.20])      # stick tốt hơn => greedy action = 0
    for eps in (0.0, 0.3, 1.0):
        actions = [epsilon_greedy_action(Q, (18, 6, False), 2, eps, rng) for _ in range(20)]
        print(f"epsilon={eps}: {actions}")


if __name__ == "__main__":
    main()
