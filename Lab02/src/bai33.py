"""
Bai 33 - Trich xuat optimal policy tu Value Iteration va hien thi dang luoi.
"""

import gymnasium as gym

from mdp_utils import value_iteration, greedy_policy_from_value, print_frozenlake_policy


def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    env.reset(seed=42)

    gamma = 0.99
    V, n_iterations, deltas = value_iteration(env, gamma=gamma, theta=1e-8)
    optimal_policy = greedy_policy_from_value(env, V, gamma=gamma)

    print(f"Value Iteration hoi tu sau {n_iterations} iteration.\n")

    print("Optimal state values:")
    print(V.reshape(4, 4))

    print("\nOptimal policy (chi so action):")
    print(optimal_policy.reshape(4, 4))

    print("\nOptimal policy (dang luoi mui ten):")
    print_frozenlake_policy(env, optimal_policy)

    env.close()


if __name__ == "__main__":
    main()
