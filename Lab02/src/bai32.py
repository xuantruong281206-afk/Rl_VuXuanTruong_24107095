"""
Bai 32 - Value Iteration hoan chinh (goi lai tu mdp_utils).
"""

import gymnasium as gym

from mdp_utils import value_iteration


def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    env.reset(seed=42)

    V, n_iterations, deltas = value_iteration(env, gamma=0.99, theta=1e-8)

    print(f"Value Iteration hoi tu sau {n_iterations} iteration.")
    print("State values:")
    print(V.reshape(4, 4))

    env.close()


if __name__ == "__main__":
    main()
