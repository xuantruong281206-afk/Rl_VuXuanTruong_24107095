"""Bai 01 - Tạo Blackjack-v1"""
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
    observation, info = env.reset(seed=42)
    print("observation      :", observation)
    print("info             :", info)
    print("observation_space:", env.observation_space)
    print("action_space     :", env.action_space)
    env.close()


if __name__ == "__main__":
    main()
