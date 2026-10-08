"""Bai 30 - Khởi tạo Q"""
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
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))      # N(s,a)
    print("Q[(18, 6, False)] khi chưa học :", Q[(18, 6, False)])
    print("N[(18, 6, False)] khi chưa học :", N[(18, 6, False)])
    print("Số state hiện có trong Q       :", len(Q))
    env.close()


if __name__ == "__main__":
    main()
