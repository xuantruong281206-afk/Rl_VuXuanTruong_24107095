"""Bai 03 - Phân tích action"""
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
    print("action_space:", env.action_space, "-> số action =", env.action_space.n)
    # 0 = stick: dừng rút bài, dealer chơi nốt và ván kết thúc
    # 1 = hit  : rút thêm một lá bài, ván tiếp tục (tổng > 21 thì bust và thua)
    ACTION_NAMES = {0: "stick", 1: "hit"}
    for a, name in ACTION_NAMES.items():
        print(f"action {a} = {name}")
    env.close()


if __name__ == "__main__":
    main()
