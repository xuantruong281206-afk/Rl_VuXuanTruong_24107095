"""Bai 21 - So sánh First-Visit và Every-Visit"""
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
    V_first, _ = first_visit_mc_prediction(env, stick_on_20_policy, 50_000, gamma=1.0, seed=0)
    V_every, cnt = every_visit_mc_prediction(env, stick_on_20_policy, 50_000, gamma=1.0, seed=0)
    print(f"{'state':18s} {'V_first':>9s} {'V_every':>9s} {'|diff|':>8s}")
    for s in sorted(V_every, key=lambda s: -cnt[s])[:10]:
        print(f"{str(s):18s} {V_first[s]:9.4f} {V_every[s]:9.4f} {abs(V_first[s] - V_every[s]):8.5f}")
    env.close()


if __name__ == "__main__":
    main()
