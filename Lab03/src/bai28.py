"""Bai 28 - Kiểm tra epsilon-greedy"""
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
    Q = {"s": np.array([1.0, 5.0])}
    epsilon, n = 0.1, 10_000
    actions = np.array([epsilon_greedy_action(Q, "s", 2, epsilon, rng) for _ in range(n)])
    counts = np.bincount(actions, minlength=2)
    for a in range(2):
        print(f"action {a}: {counts[a]:5d} lần  ({counts[a] / n:.3f})")
    print(f"Lý thuyết: greedy = 1 - ε + ε/2 = {1 - epsilon + epsilon / 2:.3f}, còn lại = ε/2 = {epsilon / 2:.3f}")
    # Nhận xét: action 1 (greedy) được chọn ~95%, action 0 ~5% => agent vẫn thỉnh thoảng khám phá.


if __name__ == "__main__":
    main()
