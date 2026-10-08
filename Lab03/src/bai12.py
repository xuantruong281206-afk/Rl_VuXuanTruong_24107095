"""Bai 12 - Sinh 100 episode"""
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
    counts = {"win": 0, "loss": 0, "draw": 0}
    for i in range(100):
        ep = generate_episode(env, stick_on_20_policy, seed=0 if i == 0 else None)
        final_reward = ep[-1][2]
        if final_reward > 0:   counts["win"] += 1
        elif final_reward < 0: counts["loss"] += 1
        else:                  counts["draw"] += 1
    for k, v in counts.items():
        print(f"{k:5s}: {v:3d}  ({v / 100:.0%})")
    env.close()


if __name__ == "__main__":
    main()
