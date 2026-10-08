"""Bai 19 - Visualize value"""
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
    V, _ = first_visit_mc_prediction(env, stick_on_20_policy, n_episodes=100_000, seed=0)
    plot_value_heatmap(V, False, "blackjack_value_no_usable_ace.png", "V(s), stick_on_20, không usable ace")
    plot_value_heatmap(V, True, "blackjack_value_usable_ace.png", "V(s), stick_on_20, có usable ace")
    # Bảng số cho usable_ace=False (hàng: player_sum 12..21, cột: dealer A,2..10)
    grid = value_grid(V, usable_ace=False)
    print(np.round(grid, 2))
    env.close()


if __name__ == "__main__":
    main()
