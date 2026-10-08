"""
experiments.py - Cac thi nghiem / ham ve hinh cap cao cho Lab03.
Dung cac ham da tu cai dat trong mc_utils.py.
"""
import numpy as np
import matplotlib.pyplot as plt

from mc_utils import *


def sample_long_episode(env, policy, min_len=3, max_tries=1000, seed0=0):
    """Tim mot episode co it nhat `min_len` buoc de minh hoa trajectory (thu lan luot cac seed)."""
    episode = []
    for k in range(max_tries):
        episode = generate_episode(env, policy, seed=seed0 + k)
        if len(episode) >= min_len:
            return episode
    return episode


def compare_first_every_convergence(blackjack_env, frozen_env, n_episodes=10000, seed=0):
    """
    Bai 23: ve V(s) cua First-Visit va Every-Visit theo so episode cho 3 state,
    tren Blackjack (hang tren) va FrozenLake (hang duoi). Luu first_vs_every_visit.png.
    Tra ve dict mean absolute difference cua tung env.
    """
    checkpoints = [c for c in (100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000) if c <= n_episodes]
    configs = [
        ("Blackjack-v1, stick_on_20", blackjack_env, lambda: stick_on_20_policy,
         [(20, 10, False), (18, 6, False), (13, 2, False)]),
        ("FrozenLake-v1, random policy", frozen_env,
         lambda: make_random_policy(frozen_env.action_space.n, np.random.default_rng(seed)),
         [0, 4, 9]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    diffs = {}
    for row, (title, env, policy_factory, states) in enumerate(configs):
        hist_f = {"checkpoints": set(checkpoints), "track": states}
        hist_e = {"checkpoints": set(checkpoints), "track": states}
        # policy_factory() tao policy moi (rng moi) => hai phuong phap thay CUNG chuoi episode
        V_f, _ = first_visit_mc_prediction(env, policy_factory(), n_episodes, seed=seed, history=hist_f)
        V_e, _ = every_visit_mc_prediction(env, policy_factory(), n_episodes, seed=seed, history=hist_e)
        common = set(V_f) & set(V_e)
        diffs[title] = float(np.mean([abs(V_f[s] - V_e[s]) for s in common]))
        for col, s in enumerate(states):
            ax = axes[row, col]
            ax.plot(checkpoints, hist_f[s], "o-", label="First-Visit")
            ax.plot(checkpoints, hist_e[s], "s--", label="Every-Visit")
            ax.set_xscale("log"); ax.set_title(f"{title}\nstate = {s}", fontsize=9)
            ax.set_xlabel("số episode"); ax.set_ylabel("V(s)"); ax.grid(alpha=0.3)
            if row == 0 and col == 0:
                ax.legend()
    fig.suptitle("First-Visit vs Every-Visit MC: hội tụ của V(s)")
    fig.tight_layout()
    save_fig(fig, "first_vs_every_visit.png")
    return diffs


def epsilon_experiment(env, epsilons=(0.01, 0.05, 0.10, 0.20, 0.50), n_samples=10000,
                       n_train=100_000, n_eval=20_000, seed=0):
    """
    Bai 29: (a) tan suat chon greedy action cua epsilon-greedy voi Q gia [1, 5];
            (b) chat luong policy hoc duoc (mean reward khi danh gia) theo epsilon luc train.
    Luu epsilon_comparison.png. Tra ve list cac dong ket qua.
    """
    rng = np.random.default_rng(seed)
    Q_toy = {"s": np.array([1.0, 5.0])}       # action 1 la greedy
    rows = []
    for eps in epsilons:
        actions = np.array([epsilon_greedy_action(Q_toy, "s", 2, eps, rng) for _ in range(n_samples)])
        freq = float(np.mean(actions == 1))
        _, policy, _ = on_policy_mc_control(env, n_train, epsilon=eps, seed=seed)
        ev = evaluate_policy(env, make_policy_fn(policy), n_eval, seed=123)
        rows.append({"epsilon": eps, "greedy_freq": freq, "theory": 1 - eps + eps / 2,
                     "mean_reward": ev["mean_reward"], "std_error": ev["std_error"]})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = [r["epsilon"] for r in rows]
    axes[0].plot(x, [r["greedy_freq"] for r in rows], "o-", label="thực nghiệm")
    axes[0].plot(x, [r["theory"] for r in rows], "k--", label="lý thuyết 1-ε+ε/2")
    axes[0].set_xlabel("epsilon"); axes[0].set_ylabel("tần suất chọn greedy action")
    axes[0].set_title("Epsilon-greedy: tần suất chọn greedy"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].errorbar(x, [r["mean_reward"] for r in rows], yerr=[1.96 * r["std_error"] for r in rows],
                     fmt="o-", capsize=3)
    axes[1].set_xlabel("epsilon (lúc train)"); axes[1].set_ylabel("mean reward của policy học được")
    axes[1].set_title(f"MC control ({n_train:,} episode) theo epsilon"); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "epsilon_comparison.png")
    return rows


def plot_policy_performance(results, name="policy_performance.png"):
    """
    Bai 34: ve so sanh cac policy. `results` = {ten_policy: dict tu evaluate_policy}.
    Trai: mean reward (+- 95% CI). Phai: ti le win/draw/loss.
    """
    names = list(results)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    means = [results[n]["mean_reward"] for n in names]
    errs = [1.96 * results[n]["std_error"] for n in names]
    axes[0].bar(names, means, yerr=errs, capsize=4, color="tab:blue")
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_ylabel("mean reward"); axes[0].set_title("Mean reward (±95% CI)")
    axes[0].tick_params(axis="x", rotation=20)
    bottom = np.zeros(len(names))
    for key, color in (("win_rate", "tab:green"), ("draw_rate", "tab:gray"), ("loss_rate", "tab:red")):
        vals = np.array([results[n][key] for n in names])
        axes[1].bar(names, vals, bottom=bottom, label=key.replace("_rate", ""), color=color)
        bottom += vals
    axes[1].set_title("Tỉ lệ win / draw / loss"); axes[1].legend()
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    save_fig(fig, name)
