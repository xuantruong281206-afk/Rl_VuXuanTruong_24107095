"""Bai 02 - Phân tích observation"""
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
    # observation = (player_sum, dealer_card, usable_ace)
    #   player_sum  : tổng điểm bài của người chơi (Ace tính 11 nếu không làm quá 21)
    #   dealer_card : lá bài dealer đang lật ngửa (1 = Ace, 2..10; J/Q/K tính 10)
    #   usable_ace  : True nếu người chơi có Ace đang được tính là 11 mà chưa bị bust
    for episode_id in range(12):
        obs, _ = env.reset(seed=episode_id)      # mỗi episode = một ván bài mới
        print(f"episode {episode_id:2d}: player_sum={obs[0]:2d}, dealer_card={obs[1]:2d}, usable_ace={obs[2]}")
    env.close()


if __name__ == "__main__":
    main()
