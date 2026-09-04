"""
Bai 34 - Danh gia policy bang simulation: so sanh random policy, policy tu
Value Iteration va policy tu Policy Iteration.
"""

import numpy as np
import gymnasium as gym

from mdp_utils import (
    evaluate_policy_by_simulation,
    greedy_policy_from_value,
    policy_iteration,
    value_iteration,
)


def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    env.reset(seed=42)

    n_states = env.observation_space.n
    gamma = 0.99

    # 1) Random policy (deterministic ngau nhien, co dinh de tai lap duoc)
    rng = np.random.default_rng(0)
    random_policy = rng.integers(0, env.action_space.n, size=n_states)

    # 2) Policy tu Value Iteration
    V_vi, _, _ = value_iteration(env, gamma=gamma, theta=1e-8)
    vi_policy = greedy_policy_from_value(env, V_vi, gamma=gamma)

    # 3) Policy tu Policy Iteration
    pi_policy, _, _ = policy_iteration(env, gamma=gamma, theta=1e-8)

    results = {
        "Random policy": evaluate_policy_by_simulation(env, random_policy, n_episodes=1000, seed=42),
        "Value Iteration policy": evaluate_policy_by_simulation(env, vi_policy, n_episodes=1000, seed=42),
        "Policy Iteration policy": evaluate_policy_by_simulation(env, pi_policy, n_episodes=1000, seed=42),
    }

    print(f"{'Policy':<25s}{'Success rate':>15s}{'Mean reward':>15s}{'Mean length':>15s}")
    for name, stats in results.items():
        print(
            f"{name:<25s}{stats['success_rate']:>15.3f}"
            f"{stats['mean_reward']:>15.3f}{stats['mean_length']:>15.2f}"
        )

    env.close()


if __name__ == "__main__":
    main()
