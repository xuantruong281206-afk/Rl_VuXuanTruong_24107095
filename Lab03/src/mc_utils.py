"""
mc_utils.py - Cac ham dung chung cho Lab03: Monte Carlo Methods.

Tat ca thuat toan (prediction, control, epsilon-greedy) deu tu cai dat,
khong dung thu vien RL co san. Chi dung gymnasium de lay environment.
"""
import os
import pickle
from collections import defaultdict

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- hang so
ACTION_NAMES = {0: "stick", 1: "hit"}   # 0 = dung lai, 1 = rut them bai
STICK, HIT = 0, 1

try:                                   # chay tu file .py
    _BASE = os.path.dirname(os.path.abspath(__file__))
    FIG_DIR = os.path.join(_BASE, "..", "figures")
    DATA_DIR = os.path.join(_BASE, "..", "data")
except NameError:                      # chay trong notebook
    FIG_DIR, DATA_DIR = "figures", "data"


def make_env(name="Blackjack-v1", **kwargs):
    """Tao environment gymnasium theo ten."""
    return gym.make(name, **kwargs)


# ---------------------------------------------------------------- policy
def make_random_policy(n_actions, rng):
    """Tra ve policy ngau nhien deu: state -> action."""
    return lambda state: int(rng.integers(n_actions))


def stick_on_20_policy(state):
    """Policy co dinh cua Bai 11: stick (0) neu tong >= 20, nguoc lai hit (1)."""
    player_sum, dealer_card, usable_ace = state
    if player_sum >= 20:
        return STICK
    return HIT


def stick_on_17_policy(state):
    """Policy co dinh bo sung (giong dealer): stick neu tong >= 17."""
    return STICK if state[0] >= 17 else HIT


def make_policy_fn(policy_dict, default_action=STICK):
    """Doi dict {state: action} thanh ham policy (state la khoa thieu -> default_action)."""
    return lambda state: policy_dict.get(state, default_action)


# ---------------------------------------------------------------- episode
def generate_episode(env, policy, seed=None, max_steps=1000):
    """
    Sinh MOT episode bang cach cho agent tuong tac voi env theo `policy`.

    Tham so
    -------
    env : environment gymnasium
    policy : ham state -> action
    seed : seed cho env.reset (None = tiep tuc luong ngau nhien cua env)
    max_steps : gioi han so buoc (phong env khong ket thuc)

    Tra ve
    ------
    episode : list cac tuple (state, action, reward);
              reward la R_(t+1) nhan duoc sau khi lam action o state S_t.
    """
    state, _ = env.reset(seed=seed)
    episode = []
    for _ in range(max_steps):
        action = policy(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if terminated or truncated:
            break
    return episode


def compute_returns(rewards, gamma=1.0):
    """
    Tinh return G_t cho moi timestep, duyet NGUOC tu cuoi episode:
        G_t = R_(t+1) + gamma * G_(t+1)

    Tra ve list [G_0, G_1, ..., G_(T-1)] cung do dai voi `rewards`.
    """
    returns = [0.0] * len(rewards)
    g = 0.0
    for t in reversed(range(len(rewards))):
        g = rewards[t] + gamma * g
        returns[t] = g
    return returns


# ---------------------------------------------------------------- prediction
def _mc_prediction(env, policy, n_episodes, gamma, first_visit, seed, history):
    """Ham loi dung chung cho First-Visit va Every-Visit MC prediction."""
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    for i in range(n_episodes):
        episode = generate_episode(env, policy, seed=seed if i == 0 else None)
        states = [s for s, _, _ in episode]
        G = compute_returns([r for _, _, r in episode], gamma)
        seen = set()
        for t, state in enumerate(states):
            if first_visit:
                if state in seen:          # chi lay lan xuat hien dau tien
                    continue
                seen.add(state)
            returns_sum[state] += G[t]
            returns_count[state] += 1
        if history is not None and (i + 1) in history["checkpoints"]:
            for s in history["track"]:
                c = returns_count.get(s, 0)
                history.setdefault(s, []).append(returns_sum[s] / c if c else np.nan)
    V = {s: returns_sum[s] / returns_count[s] for s in returns_count}
    return V, dict(returns_count)


def first_visit_mc_prediction(env, policy, n_episodes, gamma=1.0, seed=None, history=None):
    """
    First-Visit MC prediction: V(s) = trung binh return tai LAN DAU s xuat hien trong episode.

    history (tuy chon): dict {"checkpoints": [...], "track": [states]} se duoc dien them
    estimate V(s) tai cac checkpoint de ve duong hoi tu.
    Tra ve (V, returns_count).
    """
    return _mc_prediction(env, policy, n_episodes, gamma, True, seed, history)


def every_visit_mc_prediction(env, policy, n_episodes, gamma=1.0, seed=None, history=None):
    """Every-Visit MC prediction: dung return o MOI lan s xuat hien. Tra ve (V, returns_count)."""
    return _mc_prediction(env, policy, n_episodes, gamma, False, seed, history)


def mc_action_value_prediction(env, policy, n_episodes, gamma=1.0, seed=None):
    """
    Uoc luong Q(s,a) cua `policy` bang first-visit MC tren cap (s,a).
    Tra ve (Q, N): Q[state][action] va N[state][action] (so lan cap duoc cap nhat).
    """
    n_actions = env.action_space.n
    returns_sum = defaultdict(lambda: np.zeros(n_actions))
    N = defaultdict(lambda: np.zeros(n_actions))
    for i in range(n_episodes):
        episode = generate_episode(env, policy, seed=seed if i == 0 else None)
        G = compute_returns([r for _, _, r in episode], gamma)
        seen = set()
        for t, (s, a, _) in enumerate(episode):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            returns_sum[s][a] += G[t]
            N[s][a] += 1
    Q = defaultdict(lambda: np.zeros(n_actions))
    for s in N:
        Q[s] = np.divide(returns_sum[s], N[s], out=np.zeros(n_actions), where=N[s] > 0)
    return Q, N


# ---------------------------------------------------------------- epsilon-greedy
def greedy_action(Q, state, n_actions):
    """Action tot nhat theo Q (np.argmax). State chua gap -> action 0."""
    if state not in Q:
        return 0
    return int(np.argmax(Q[state][:n_actions]))


def epsilon_greedy_action(Q, state, n_actions, epsilon, rng):
    """
    Epsilon-greedy: voi xac suat epsilon chon ngau nhien (deu) trong moi action,
    nguoc lai chon greedy. => P(greedy) = 1 - epsilon + epsilon/n_actions.
    """
    if rng.random() < epsilon:
        return int(rng.integers(n_actions))
    return greedy_action(Q, state, n_actions)


# ---------------------------------------------------------------- control
def update_q_incremental(Q, N, state, action, G):
    """
    Cap nhat trung binh tang dan (khong can luu moi return):
        N(s,a) += 1
        Q(s,a) += (G - Q(s,a)) / N(s,a)
    """
    N[state][action] += 1
    Q[state][action] += (G - Q[state][action]) / N[state][action]


def on_policy_mc_control(env, n_episodes, gamma=1.0, epsilon=0.1, seed=42):
    """
    On-policy First-Visit MC control voi epsilon-greedy.

    Moi episode: sinh bang epsilon-greedy(Q hien tai) -> tinh return ->
    first-visit update Q(s,a) -> policy (epsilon-greedy theo Q) tu dong cai thien.

    Tra ve
    ------
    Q : defaultdict state -> mang Q(s, .)
    policy : dict {state: greedy action} cho cac state da gap
    episode_rewards : list tong reward cua tung episode luc train
    """
    n_actions = env.action_space.n
    rng = np.random.default_rng(seed)
    Q = defaultdict(lambda: np.zeros(n_actions))
    N = defaultdict(lambda: np.zeros(n_actions))
    behavior = lambda s: epsilon_greedy_action(Q, s, n_actions, epsilon, rng)
    episode_rewards = []
    for i in range(n_episodes):
        episode = generate_episode(env, behavior, seed=seed if i == 0 else None)
        rewards = [r for _, _, r in episode]
        G = compute_returns(rewards, gamma)
        episode_rewards.append(sum(rewards))
        seen = set()
        for t, (s, a, _) in enumerate(episode):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            update_q_incremental(Q, N, s, a, G[t])
    policy = {s: int(np.argmax(q)) for s, q in Q.items()}
    return Q, policy, episode_rewards


def load_or_train_control(env, n_episodes=300_000, gamma=1.0, epsilon=0.1, seed=42):
    """Train MC control, luu ket qua vao data/ de cac bai sau dung lai (tiet kiem thoi gian)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"mc_control_{n_episodes}_{gamma}_{epsilon}_{seed}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            Q, policy, rewards = pickle.load(f)
        Q = defaultdict(lambda: np.zeros(env.action_space.n), Q)
        return Q, policy, rewards
    Q, policy, rewards = on_policy_mc_control(env, n_episodes, gamma, epsilon, seed)
    with open(path, "wb") as f:
        pickle.dump((dict(Q), policy, rewards), f)
    return Q, policy, rewards


# ---------------------------------------------------------------- evaluation
def evaluate_policy(env, policy, n_episodes=10000, seed=123):
    """
    Danh gia policy (khong hoc, khong explore) tren episode MOI.
    Tra ve dict: win_rate, loss_rate, draw_rate, mean_reward, std_error.
    """
    totals = []
    for i in range(n_episodes):
        ep = generate_episode(env, policy, seed=seed if i == 0 else None)
        totals.append(sum(r for _, _, r in ep))
    totals = np.array(totals)
    return {
        "win_rate": float(np.mean(totals > 0)),
        "loss_rate": float(np.mean(totals < 0)),
        "draw_rate": float(np.mean(totals == 0)),
        "mean_reward": float(totals.mean()),
        "std_error": float(totals.std() / np.sqrt(n_episodes)),
    }


# ---------------------------------------------------------------- ve hinh
def moving_average(x, window=1000):
    """Trung binh truot (kieu 'valid')."""
    x = np.asarray(x, dtype=float)
    return np.convolve(x, np.ones(window) / window, mode="valid")


def save_fig(fig, name):
    """Luu hinh vao figures/ roi hien thi."""
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print("Da luu:", os.path.normpath(path))


def value_grid(V, usable_ace, sums=range(12, 22), dealers=range(1, 11)):
    """Bang V(player_sum, dealer_card) cho Blackjack; o chua gap = NaN."""
    return np.array([[V.get((p, d, usable_ace), np.nan) for d in dealers] for p in sums])


def plot_value_heatmap(V, usable_ace, name, title=None):
    """Heatmap V(s) theo player sum (12-21) x dealer showing (A,2..10)."""
    sums, dealers = list(range(12, 22)), list(range(1, 11))
    grid = value_grid(V, usable_ace, sums, dealers)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid, origin="lower", cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(10)); ax.set_xticklabels(["A"] + [str(d) for d in dealers[1:]])
    ax.set_yticks(range(10)); ax.set_yticklabels(sums)
    ax.set_xlabel("Dealer showing"); ax.set_ylabel("Player sum")
    ax.set_title(title or f"V(s), usable_ace={usable_ace}")
    fig.colorbar(im, label="V(s)")
    save_fig(fig, name)


def plot_policy_heatmap(policy, usable_ace, ax, title):
    """Ve policy hoc duoc (0=stick, 1=hit) tren mot truc ax."""
    sums, dealers = list(range(12, 22)), list(range(1, 11))
    grid = np.array([[policy.get((p, d, usable_ace), np.nan) for d in dealers] for p in sums])
    ax.imshow(grid, origin="lower", cmap="coolwarm", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(10)); ax.set_xticklabels(["A"] + [str(d) for d in dealers[1:]])
    ax.set_yticks(range(10)); ax.set_yticklabels(sums)
    ax.set_xlabel("Dealer showing"); ax.set_ylabel("Player sum"); ax.set_title(title)
    for i in range(10):
        for j in range(10):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, "H" if v == 1 else "S", ha="center", va="center", fontsize=8)
