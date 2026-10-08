"""Bai 08 - Kiểm tra return"""
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
    rewards = [0, 0, 1]
    for gamma in (1.0, 0.9, 0.5):
        G = compute_returns(rewards, gamma)
        expected = [gamma ** 2, gamma, 1.0]
        assert np.allclose(G, expected), (G, expected)
        print(f"gamma={gamma}: G = {G}   (kỳ vọng {expected})")
    print("Tất cả kiểm tra đều đúng.")


if __name__ == "__main__":
    main()
