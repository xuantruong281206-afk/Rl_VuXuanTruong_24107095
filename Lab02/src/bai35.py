"""
Bai 35 - So sanh Value Iteration va Policy Iteration: so vong lap, thoi gian
chay, success rate, mean reward. Ve va luu bieu do so sanh.
"""

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
    timed_run,
    value_iteration,
)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    env.reset(seed=42)

    gamma = 0.99
    theta = 1e-8

    (V_vi, n_iter_vi, deltas_vi), time_vi = timed_run(
        value_iteration, env, gamma=gamma, theta=theta
    )
    vi_policy = greedy_policy_from_value(env, V_vi, gamma=gamma)

    (pi_policy, V_pi, n_iter_pi), time_pi = timed_run(
        policy_iteration, env, gamma=gamma, theta=theta
    )

    stats_vi = evaluate_policy_by_simulation(env, vi_policy, n_episodes=1000, seed=42)
    stats_pi = evaluate_policy_by_simulation(env, pi_policy, n_episodes=1000, seed=42)

    print(f"{'Thuat toan':<20s}{'So vong lap':>15s}{'Thoi gian (s)':>16s}"
          f"{'Success rate':>15s}{'Mean reward':>15s}")
    print(f"{'Value Iteration':<20s}{n_iter_vi:>15d}{time_vi:>16.4f}"
          f"{stats_vi['success_rate']:>15.3f}{stats_vi['mean_reward']:>15.3f}")
    print(f"{'Policy Iteration':<20s}{n_iter_pi:>15d}{time_pi:>16.4f}"
          f"{stats_pi['success_rate']:>15.3f}{stats_pi['mean_reward']:>15.3f}")

    # Bieu do so sanh
    labels = ["Value Iteration", "Policy Iteration"]
    mean_rewards = [stats_vi["mean_reward"], stats_pi["mean_reward"]]
    runtimes = [time_vi, time_pi]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].bar(labels, mean_rewards, color=["tab:blue", "tab:green"])
    axes[0].set_title("Mean reward")
    axes[0].set_ylabel("Mean reward")
    axes[0].grid(True, axis="y")

    axes[1].bar(labels, runtimes, color=["tab:blue", "tab:green"])
    axes[1].set_title("Runtime")
    axes[1].set_ylabel("Thoi gian (giay)")
    axes[1].grid(True, axis="y")

    fig.suptitle("So sanh Value Iteration va Policy Iteration tren FrozenLake-v1")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, "algorithm_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nDa luu bieu do vao: {out_path}")

    # Nhan xet (>= 8 dong):
    # 1. Ca hai thuat toan deu hoi tu ve cung mot optimal value function va
    #    optimal policy (vi cung giai mot MDP voi model day du, khong xap xi).
    # 2. Value Iteration thuong can nhieu sweep hon nhung moi sweep re hon,
    #    vi chi lam mot Bellman optimality backup cho moi state.
    # 3. Policy Iteration can it vong lap ngoai hon (moi vong la 1 lan
    #    Policy Evaluation + 1 lan Policy Improvement), nhung ban than
    #    Policy Evaluation ben trong lai la mot vong lap con co the chay
    #    kha nhieu iteration truoc khi hoi tu.
    # 4. Vi vay tong thoi gian chay giua hai thuat toan co the khac nhau
    #    tuy vao gia tri theta va toc do hoi tu cua Policy Evaluation.
    # 5. Tren mot moi truong nho nhu FrozenLake 4x4, ca hai deu chay rat
    #    nhanh (duoi 1 giay), nen su khac biet thoi gian chua the hien ro
    #    nhu tren cac moi truong lon hon (vi du 8x8 hoac lon hon).
    # 6. Success rate va mean reward khi danh gia bang simulation gan nhu
    #    giong nhau giua hai policy, vi ca hai deu la optimal policy (co
    #    the khac nhau ve action tai vai state co nhieu action toi uu dong
    #    thoi, nhung gia tri ky vong thi nhu nhau).
    # 7. Nhin chung, Value Iteration don gian hon de cai dat (chi mot vong
    #    lap), trong khi Policy Iteration can quan ly hai buoc long nhau
    #    nhung thuong hoi tu voi it vong lap ngoai hon.
    # 8. Lua chon thuat toan nao trong thuc te phu thuoc vao kich thuoc
    #    state/action space va chi phi cua moi lan Policy Evaluation day du.

    env.close()


if __name__ == "__main__":
    main()
