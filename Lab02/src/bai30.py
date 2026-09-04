"""
Bai 30 - Kiem tra policy stability trong Policy Iteration, tu lap trinh
(khong goi thang policy_iteration() cua mdp_utils de minh hoa cach kiem tra).
"""

import numpy as np
import gymnasium as gym

from mdp_utils import policy_evaluation, greedy_policy_from_value


def my_policy_iteration(env, gamma=0.99, theta=1e-8, max_iterations=1000):
    n_states = env.observation_space.n
    policy = np.zeros(n_states, dtype=int)

    for i in range(1, max_iterations + 1):
        V, _ = policy_evaluation(env, policy, gamma=gamma, theta=theta)
        new_policy = greedy_policy_from_value(env, V, gamma=gamma)

        policy_stable = np.array_equal(new_policy, policy)
        policy = new_policy

        if policy_stable:
            print(f"Policy Iteration converged after {i} iterations.")
            return policy, V, i

    print(f"Policy Iteration khong hoi tu sau {max_iterations} vong lap.")
    return policy, V, max_iterations


def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    env.reset(seed=42)

    my_policy_iteration(env, gamma=0.99, theta=1e-8)

    env.close()


if __name__ == "__main__":
    main()
