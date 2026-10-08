"""Bai 34 - Đánh giá policy đã học"""
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
    Q, policy, _ = load_or_train_control(env, n_episodes=300_000, epsilon=0.1, seed=42)
    n_eval = 50_000
    results = {
        "random": evaluate_policy(env, make_random_policy(2, np.random.default_rng(0)), n_eval, seed=123),
        "fixed (stick>=20)": evaluate_policy(env, stick_on_20_policy, n_eval, seed=123),
        "fixed (stick>=17)": evaluate_policy(env, stick_on_17_policy, n_eval, seed=123),
        "MC learned": evaluate_policy(env, make_policy_fn(policy), n_eval, seed=123),
    }
    print(f"{'policy':20s} {'win':>7s} {'loss':>7s} {'draw':>7s} {'mean reward':>12s}")
    for name, r in results.items():
        print(f"{name:20s} {r['win_rate']:7.3f} {r['loss_rate']:7.3f} {r['draw_rate']:7.3f} {r['mean_reward']:12.4f}")
    plot_policy_performance(results)
    env.close()


if __name__ == "__main__":
    main()
