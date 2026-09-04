"""
Bai 26 - Xay dung greedy policy tu state-value function V.
"""

import numpy as np
import gymnasium as gym

from mdp_utils import greedy_policy_from_value


def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    env.reset(seed=42)

    V = np.zeros(env.observation_space.n)  # V ban dau (chua evaluate gi ca)
    policy = greedy_policy_from_value(env, V, gamma=0.99)

    print("Greedy policy tu V=0 (chi mang tinh minh hoa buoc lam):")
    print(policy.reshape(4, 4))

    env.close()


if __name__ == "__main__":
    main()
