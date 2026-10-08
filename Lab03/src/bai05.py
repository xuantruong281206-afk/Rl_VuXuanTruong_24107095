"""Bai 05 - Lưu episode"""
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
    episode = sample_long_episode(env, random_policy, min_len=3)   # tìm episode đủ dài để minh họa
    print("Số bước trong episode:", len(episode))
    for t, (state, action, reward) in enumerate(episode):
        print(f"t={t}: state={state}, action={ACTION_NAMES[action]}, reward={reward}")
    env.close()


if __name__ == "__main__":
    main()
