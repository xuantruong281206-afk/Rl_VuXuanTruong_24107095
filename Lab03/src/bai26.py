"""Bai 26 - Greedy action từ Q"""
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
    random_policy = make_random_policy(env.action_space.n, np.random.default_rng(1))
    Q, _ = mc_action_value_prediction(env, random_policy, n_episodes=100_000, seed=1)
    for s in [(20, 10, False), (13, 2, False), (18, 6, True), (12, 6, False)]:
        a = greedy_action(Q, s, env.action_space.n)
        print(f"state={s}: Q={np.round(Q[s], 3)} -> greedy action = {a} ({ACTION_NAMES[a]})")
    # Lưu ý: đây là greedy theo Q của RANDOM policy, chưa phải policy tối ưu.
    env.close()


if __name__ == "__main__":
    main()
