"""
Bai 28 - Mot buoc Policy Improvement: Evaluation + Improvement, dem so state
doi action.
"""

import numpy as np
import gymnasium as gym

from mdp_utils import policy_evaluation, greedy_policy_from_value


def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    env.reset(seed=42)

    n_states = env.observation_space.n
    old_policy = np.zeros(n_states, dtype=int)  # policy ban dau: luon LEFT

    V, n_iterations = policy_evaluation(env, old_policy, gamma=0.99)
    new_policy = greedy_policy_from_value(env, V, gamma=0.99)

    n_changed = np.sum(old_policy != new_policy)

    print(f"Policy Evaluation hoi tu sau {n_iterations} iteration.")
    print("Old policy:", old_policy.reshape(4, 4))
    print("New policy (sau Improvement):", new_policy.reshape(4, 4))
    print(f"So state doi action: {n_changed} / {n_states}")

    env.close()


if __name__ == "__main__":
    main()
