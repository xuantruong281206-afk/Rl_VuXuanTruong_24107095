"""Bai 31 - Incremental mean"""
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
    Q = defaultdict(lambda: np.zeros(2))
    N = defaultdict(lambda: np.zeros(2))
    state, action = (18, 6, False), 1
    returns_seen = []
    for G in [1, -1, 1, 1, -1, -1, 1]:
        returns_seen.append(G)
        update_q_incremental(Q, N, state, action, G)
        assert np.isclose(Q[state][action], np.mean(returns_seen))
        print(f"G={G:+d}: Q={Q[state][action]:+.4f}  mean(returns)={np.mean(returns_seen):+.4f}  N={int(N[state][action])}")
    print("Incremental mean trùng với trung bình đầy đủ, nhưng không cần lưu toàn bộ returns.")


if __name__ == "__main__":
    main()
