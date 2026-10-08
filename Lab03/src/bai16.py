"""Bai 16 - Phát hiện first visit"""
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
    def find_first_visits(episode):
        """Trả về dict {state: chỉ số timestep xuất hiện ĐẦU TIÊN}."""
        first_visit_index = {}
        for t, (state, _, _) in enumerate(episode):
            if state not in first_visit_index:
                first_visit_index[state] = t
        return first_visit_index

    env = make_env("Blackjack-v1")
    random_policy = make_random_policy(env.action_space.n, np.random.default_rng(1))
    episode = sample_long_episode(env, random_policy, min_len=3)
    print("Blackjack:", find_first_visits(episode))
    print("-> Trong Blackjack mỗi lần hit đều đổi (player_sum, usable_ace) và không bao giờ quay lại state cũ => state KHÔNG lặp lại trong 1 episode.")

    # FrozenLake: agent có thể quay lại cùng một ô => state lặp lại, first visit khác every visit
    frozen = make_env("FrozenLake-v1")
    frozen_policy = make_random_policy(4, np.random.default_rng(0))
    ep = sample_long_episode(frozen, frozen_policy, min_len=10)
    states = [s for s, _, _ in ep]
    print("\nFrozenLake states   :", states)
    print("first-visit index   :", find_first_visits(ep))
    repeated = sorted({s for s in states if states.count(s) > 1})
    print("state xuất hiện >1 lần:", repeated)
    env.close(); frozen.close()


if __name__ == "__main__":
    main()
