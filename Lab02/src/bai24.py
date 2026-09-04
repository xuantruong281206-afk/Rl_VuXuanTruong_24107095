"""
Bai 24 - Iterative Policy Evaluation hoan chinh.
"""

import numpy as np
import gymnasium as gym

from mdp_utils import policy_evaluation


def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    env.reset(seed=42)

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    policy = np.ones((n_states, n_actions)) / n_actions  # uniform random policy

    V, n_iterations = policy_evaluation(env, policy, gamma=0.99, theta=1e-8)

    print(f"Hoi tu sau {n_iterations} iteration.")
    print("State-value function V (dang luoi 4x4):")
    print(V.reshape(4, 4))

    env.close()


if __name__ == "__main__":
    main()
