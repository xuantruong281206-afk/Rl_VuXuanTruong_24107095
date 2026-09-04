"""
Bai 27 - Hien thi policy FrozenLake tren luoi 4x4 bang mui ten.
"""

import numpy as np
import gymnasium as gym

from mdp_utils import policy_evaluation, greedy_policy_from_value, print_frozenlake_policy


def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    env.reset(seed=42)

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    uniform_policy = np.ones((n_states, n_actions)) / n_actions

    V, _ = policy_evaluation(env, uniform_policy, gamma=0.99)
    greedy_policy = greedy_policy_from_value(env, V, gamma=0.99)

    print("Policy dang luoi 4x4 (greedy tu V cua uniform policy):")
    print_frozenlake_policy(env, greedy_policy)

    env.close()


if __name__ == "__main__":
    main()
