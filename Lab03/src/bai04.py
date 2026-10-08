"""Bai 04 - Một episode ngẫu nhiên"""
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
    rng = np.random.default_rng(0)
    state, _ = env.reset(seed=0)
    step = 0
    while True:
        action = int(rng.integers(env.action_space.n))            # random policy
        next_state, reward, terminated, truncated, _ = env.step(action)
        print(f"step {step}: state={state} action={ACTION_NAMES[action]} reward={reward} "
              f"next_state={next_state} terminated={terminated} truncated={truncated}")
        state, step = next_state, step + 1
        if terminated or truncated:
            break
    env.close()


if __name__ == "__main__":
    main()
