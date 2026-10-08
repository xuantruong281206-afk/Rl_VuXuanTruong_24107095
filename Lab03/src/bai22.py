"""Bai 22 - Sai khác giữa hai phương pháp"""
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
    def mean_abs_diff(V_a, V_b):
        common = set(V_a) & set(V_b)
        return np.mean([abs(V_a[s] - V_b[s]) for s in common]), len(common)

    env = make_env("Blackjack-v1")
    Vf, _ = first_visit_mc_prediction(env, stick_on_20_policy, 50_000, seed=0)
    Ve, _ = every_visit_mc_prediction(env, stick_on_20_policy, 50_000, seed=0)
    d, n = mean_abs_diff(Vf, Ve)
    print(f"Blackjack : mean |V_first - V_every| = {d:.6f} trên {n} state chung")
    # = 0 vì state không lặp lại trong 1 episode => first visit và every visit chọn đúng cùng các return.

    frozen = make_env("FrozenLake-v1")
    pf = make_random_policy(4, np.random.default_rng(0)); Vf, _ = first_visit_mc_prediction(frozen, pf, 20_000, seed=0)
    pe = make_random_policy(4, np.random.default_rng(0)); Ve, _ = every_visit_mc_prediction(frozen, pe, 20_000, seed=0)
    d, n = mean_abs_diff(Vf, Ve)
    print(f"FrozenLake: mean |V_first - V_every| = {d:.6f} trên {n} state chung")
    # > 0 vì agent có thể quay lại cùng một ô trong một episode.
    env.close(); frozen.close()


if __name__ == "__main__":
    main()
