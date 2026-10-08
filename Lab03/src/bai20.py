"""Bai 20 - Every-Visit MC Prediction"""
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
    V, returns_count = every_visit_mc_prediction(env, stick_on_20_policy, n_episodes=5000, gamma=1.0, seed=0)
    print("Số state có estimate:", len(V))
    for s in sorted(V, key=lambda s: -returns_count[s])[:8]:
        print(f"V{s} = {V[s]:+.3f}  (n={returns_count[s]})")
    env.close()


if __name__ == "__main__":
    main()
