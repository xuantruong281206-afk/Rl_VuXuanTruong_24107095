"""
main.py - Chay toan bo Lab03 (mini-project Bai 36) va tao cac hinh trong figures/.

Cach chay:  python main.py
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")      # khong can giao dien do hoa

import numpy as np
import matplotlib.pyplot as plt

from mc_utils import *
from experiments import *


# === MINI_PROJECT_START ===
def run_mini_project(n_pred=50_000, n_train=300_000, n_eval=50_000, seed=42):
    """Bai 36: Monte Carlo agent hoan chinh (20 buoc) tren Blackjack-v1 (+ FrozenLake-v1 lam moi truong phu)."""
    print("=== 1. Environment ===")
    env = make_env("Blackjack-v1")
    print("Môi trường chính: Blackjack-v1")

    print("\n=== 2. Observation / action space ===")
    print("observation_space:", env.observation_space)
    print("action_space     :", env.action_space, "->", ACTION_NAMES)

    print("\n=== 3-4. Episode mẫu & trajectory ===")
    random_policy = make_random_policy(env.action_space.n, np.random.default_rng(seed))
    episode = sample_long_episode(env, random_policy, min_len=2)
    for t, (s, a, r) in enumerate(episode):
        print(f"  t={t}: state={s}, action={ACTION_NAMES[a]}, reward={r}")

    print("\n=== 5-7. Reward, return, discounted return ===")
    rewards = [r for _, _, r in episode]
    print("rewards:", rewards)
    for gamma in (1.0, 0.9):
        print(f"  gamma={gamma}: G_t =", [round(g, 3) for g in compute_returns(rewards, gamma)])

    print("\n=== 8. Fixed policy (stick khi tổng >= 20) ===")
    fixed = evaluate_policy(env, stick_on_20_policy, 10_000, seed=1)
    print("  ", {k: round(v, 4) for k, v in fixed.items()})

    print("\n=== 9-11. First-Visit vs Every-Visit MC prediction ===")
    V_f, cnt_f = first_visit_mc_prediction(env, stick_on_20_policy, n_pred, seed=seed)
    V_e, cnt_e = every_visit_mc_prediction(env, stick_on_20_policy, n_pred, seed=seed)
    common = set(V_f) & set(V_e)
    mad = np.mean([abs(V_f[s] - V_e[s]) for s in common])
    print(f"  số state ước lượng: first={len(V_f)}, every={len(V_e)}; mean |V_first - V_every| = {mad:.6f}")
    print("  (Blackjack: state không lặp lại trong 1 episode => hai phương pháp trùng nhau)")
    plot_value_heatmap(V_f, False, "blackjack_value_no_usable_ace.png", "V(s) stick_on_20, không có usable ace")
    plot_value_heatmap(V_f, True, "blackjack_value_usable_ace.png", "V(s) stick_on_20, có usable ace")

    print("\n=== 12. Q(s,a) của random policy ===")
    Q_rand, N_rand = mc_action_value_prediction(env, random_policy, 100_000, seed=seed)
    for s in [(20, 10, False), (13, 2, False), (18, 6, True)]:
        print(f"  Q{s} = {np.round(Q_rand[s], 3)}  (stick, hit)")

    print("\n=== 13. Epsilon-greedy ===")
    rng = np.random.default_rng(seed)
    acts = [epsilon_greedy_action({"s": np.array([1.0, 5.0])}, "s", 2, 0.1, rng) for _ in range(10_000)]
    print(f"  epsilon=0.1, Q=[1,5]: tần suất greedy = {np.mean(np.array(acts) == 1):.3f} (lý thuyết 0.95)")

    print("\n=== 14-15. On-policy MC control + learning curve ===")
    Q, policy, ep_rewards = load_or_train_control(env, n_train, epsilon=0.1, seed=seed)
    print(f"  đã train {n_train:,} episode, học được policy cho {len(policy)} state")
    ma = moving_average(ep_rewards, 1000)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(len(ma)) + 1000, ma)
    ax.set_xlabel("episode"); ax.set_ylabel("mean reward (moving avg, window=1000)")
    ax.set_title("On-policy MC control: learning curve (epsilon=0.1)"); ax.grid(alpha=0.3)
    save_fig(fig, "mc_convergence.png")

    print("\n=== 16. Learned policy ===")
    learned_fn = make_policy_fn(policy)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    plot_policy_heatmap(policy, False, axes[0], "Learned policy, không usable ace (S=stick, H=hit)")
    plot_policy_heatmap(policy, True, axes[1], "Learned policy, có usable ace")
    fig.tight_layout()
    save_fig(fig, "blackjack_policy.png")

    print("\n=== 17-18. Evaluation & so sánh policy ===")
    results = {
        "random": evaluate_policy(env, make_random_policy(2, np.random.default_rng(seed)), n_eval, seed=123),
        "fixed (stick>=20)": evaluate_policy(env, stick_on_20_policy, n_eval, seed=123),
        "fixed (stick>=17)": evaluate_policy(env, stick_on_17_policy, n_eval, seed=123),
        "MC learned": evaluate_policy(env, learned_fn, n_eval, seed=123),
    }
    print(f"  {'policy':20s} {'win':>7s} {'loss':>7s} {'draw':>7s} {'mean reward':>12s}")
    for name, r in results.items():
        print(f"  {name:20s} {r['win_rate']:7.3f} {r['loss_rate']:7.3f} {r['draw_rate']:7.3f} {r['mean_reward']:12.4f}")
    plot_policy_performance(results)

    print("\n=== Môi trường phụ: FrozenLake-v1 (generate episode, return, MC prediction) ===")
    frozen = make_env("FrozenLake-v1")
    frozen_policy = make_random_policy(frozen.action_space.n, np.random.default_rng(seed))
    ep = generate_episode(frozen, frozen_policy, seed=seed)
    print(f"  episode mẫu dài {len(ep)} bước, return G_0 = {compute_returns([r for _, _, r in ep], 0.99)[0]:.3f}")
    Vf_f, _ = first_visit_mc_prediction(frozen, make_random_policy(4, np.random.default_rng(seed)), 20_000, seed=seed)
    Ve_f, _ = every_visit_mc_prediction(frozen, make_random_policy(4, np.random.default_rng(seed)), 20_000, seed=seed)
    mad_f = np.mean([abs(Vf_f[s] - Ve_f[s]) for s in set(Vf_f) & set(Ve_f)])
    print(f"  V(0) first={Vf_f[0]:.4f}, every={Ve_f[0]:.4f}; mean |diff| = {mad_f:.5f} (FrozenLake có state lặp => khác nhau)")

    print("\n=== 19. Nhận xét ===")
    best = results["MC learned"]["mean_reward"]
    print(f"  - MC learned (mean reward {best:.3f}) vượt xa random ({results['random']['mean_reward']:.3f})"
          f" và fixed stick>=20 ({results['fixed (stick>=20)']['mean_reward']:.3f}).")
    print("  - Blackjack có lợi thế nhà cái nên mean reward tối ưu vẫn hơi âm (khoảng -0.05).")
    print("  - Learned policy có nhiễu ở các state hiếm; cần nhiều episode hơn để ổn định.")

    print("\n=== 20. Kết luận ===")
    print("  Chỉ từ các episode (không dùng model), Monte Carlo control với epsilon-greedy học được policy")
    print("  gần tối ưu cho Blackjack. Hạn chế: phải chờ hết episode và phương sai return lớn => Lab04: TD.")
    env.close(); frozen.close()
    return results
# === MINI_PROJECT_END ===


def main():
    env = make_env("Blackjack-v1")
    frozen = make_env("FrozenLake-v1")
    run_mini_project()
    print("\n--- Bài 23: First vs Every visit ---")
    print(compare_first_every_convergence(env, frozen, n_episodes=10_000, seed=0))
    print("\n--- Bài 29: so sánh epsilon ---")
    for row in epsilon_experiment(env):
        print({k: round(v, 4) for k, v in row.items()})
    env.close(); frozen.close()


if __name__ == "__main__":
    main()
