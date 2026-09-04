"""
Bai 25 - Theo doi hoi tu cua Policy Evaluation, ve bieu do delta theo iteration.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym

from mdp_utils import policy_evaluation_sweep

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def policy_evaluation_with_history(env, policy, gamma=0.99, theta=1e-8, max_iterations=10000):
    n_states = env.observation_space.n
    V = np.zeros(n_states)
    deltas = []

    for i in range(max_iterations):
        new_V = policy_evaluation_sweep(env, policy, V, gamma)
        delta = np.max(np.abs(new_V - V))
        deltas.append(delta)
        V = new_V
        if delta < theta:
            break

    return V, deltas


def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    env.reset(seed=42)

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    policy = np.ones((n_states, n_actions)) / n_actions

    V, deltas = policy_evaluation_with_history(env, policy, gamma=0.99, theta=1e-8)
    print(f"So iteration: {len(deltas)}")

    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(deltas) + 1), deltas, color="tab:orange")
    plt.yscale("log")
    plt.title("Hoi tu cua Policy Evaluation (uniform random policy)")
    plt.xlabel("Iteration")
    plt.ylabel("delta (log scale)")
    plt.grid(True)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, "policy_iteration_convergence.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Da luu bieu do vao: {out_path}")

    env.close()


if __name__ == "__main__":
    main()
