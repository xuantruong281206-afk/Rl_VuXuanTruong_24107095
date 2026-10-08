"""Bai 23 - Biểu đồ hội tụ"""
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
    blackjack = make_env("Blackjack-v1")
    frozen = make_env("FrozenLake-v1")
    diffs = compare_first_every_convergence(blackjack, frozen, n_episodes=10_000, seed=0)
    print("Mean absolute difference:", diffs)
    blackjack.close(); frozen.close()


if __name__ == "__main__":
    main()
