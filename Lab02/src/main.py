"""
Bai 36 - Mini-project: Dynamic Programming Solver hoan chinh cho FrozenLake-v1.

Chuong trinh nay tong hop toan bo cac ham cua Lab02 thanh mot pipeline:
    create_environment -> Value Iteration / Policy Iteration
    -> evaluate_policy_by_simulation -> plot_convergence -> so sanh

Ho tro is_slippery=False va is_slippery=True, co tham so gamma, theta,
max_iterations. Day cung chinh la noi dung cua Lab02/src/main.py.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym

from mdp_utils import (
    evaluate_policy_by_simulation,
    greedy_policy_from_value,
    policy_iteration,
    print_frozenlake_policy,
    timed_run,
    value_iteration,
)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def create_environment(is_slippery=True, map_name="4x4"):
    """Tao FrozenLake-v1 environment."""
    env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery)
    env.reset(seed=42)
    return env


def plot_convergence(deltas_vi, out_name="value_iteration_convergence.png"):
    """Ve va luu bieu do hoi tu (delta theo iteration) cua Value Iteration."""
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(deltas_vi) + 1), deltas_vi, color="tab:red")
    plt.yscale("log")
    plt.title("Hoi tu cua Value Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("delta (log scale)")
    plt.grid(True)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, out_name)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def run_solver(is_slippery, gamma, theta, max_iterations, n_episodes):
    print("=" * 70)
    print(f"FrozenLake-v1 | is_slippery={is_slippery} | gamma={gamma} | theta={theta}")
    print("=" * 70)

    env = create_environment(is_slippery=is_slippery)

    # --- Value Iteration ---
    (V_vi, n_iter_vi, deltas_vi), time_vi = timed_run(
        value_iteration, env, gamma=gamma, theta=theta, max_iterations=max_iterations
    )
    vi_policy = greedy_policy_from_value(env, V_vi, gamma=gamma)

    # --- Policy Iteration ---
    (pi_policy, V_pi, n_iter_pi), time_pi = timed_run(
        policy_iteration, env, gamma=gamma, theta=theta, max_iterations=max_iterations
    )

    print("\n--- Value Iteration ---")
    print(f"So iteration: {n_iter_vi}, thoi gian: {time_vi:.4f}s")
    print("Value table:")
    print(V_vi.reshape(4, 4))
    print("Policy:")
    print_frozenlake_policy(env, vi_policy)

    print("\n--- Policy Iteration ---")
    print(f"So vong lap: {n_iter_pi}, thoi gian: {time_pi:.4f}s")
    print("Value table:")
    print(V_pi.reshape(4, 4))
    print("Policy:")
    print_frozenlake_policy(env, pi_policy)

    # --- Danh gia bang simulation (>= 1000 episode) ---
    stats_vi = evaluate_policy_by_simulation(env, vi_policy, n_episodes=n_episodes, seed=42)
    stats_pi = evaluate_policy_by_simulation(env, pi_policy, n_episodes=n_episodes, seed=42)

    print(f"\n--- Danh gia bang {n_episodes} episode ---")
    print(f"Value Iteration : success_rate={stats_vi['success_rate']:.3f}, "
          f"mean_reward={stats_vi['mean_reward']:.3f}")
    print(f"Policy Iteration: success_rate={stats_pi['success_rate']:.3f}, "
          f"mean_reward={stats_pi['mean_reward']:.3f}")

    # --- Bieu do hoi tu ---
    out_path = plot_convergence(deltas_vi)
    print(f"\nDa luu bieu do hoi tu vao: {out_path}")

    env.close()

    return {
        "value_iteration": {"n_iterations": n_iter_vi, "time": time_vi, "stats": stats_vi},
        "policy_iteration": {"n_iterations": n_iter_pi, "time": time_pi, "stats": stats_pi},
    }


def main():
    parser = argparse.ArgumentParser(description="Dynamic Programming Solver cho FrozenLake-v1")
    parser.add_argument("--is-slippery", action="store_true", default=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--theta", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=10000)
    parser.add_argument("--n-episodes", type=int, default=1000)
    args = parser.parse_args()

    # Chay ca hai truong hop deterministic va stochastic de so sanh
    run_solver(False, args.gamma, args.theta, args.max_iterations, args.n_episodes)
    run_solver(True, args.gamma, args.theta, args.max_iterations, args.n_episodes)


if __name__ == "__main__":
    main()
