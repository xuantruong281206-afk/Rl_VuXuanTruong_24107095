"""
Bai 14 - Deterministic policy cho MDP nho.
"""

import numpy as np

from mdp_utils import print_policy

ACTION_NAMES = {0: "Thu dong", 1: "Chu dong"}


def main():
    policy = np.array([1, 0])  # state 0 -> chu dong, state 1 -> thu dong

    print("Deterministic policy:")
    print_policy(policy, action_names=ACTION_NAMES)


if __name__ == "__main__":
    main()
