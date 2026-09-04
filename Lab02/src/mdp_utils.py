"""
mdp_utils.py

Thư viện tiện ích cho Lab02 - Markov Decision Process va Dynamic Programming.

Module nay tap hop lai toan bo cac ham dung chung duoc cai dat trong cac
bai tap tu Bai 21 den Bai 36, de cac file bai*.py va main.py co the import
lai thay vi copy-paste code (dung theo yeu cau cua de bai).

Quy uoc:
- env: mot Gymnasium environment rieng roi (vi du FrozenLake-v1), truy cap
  model bang env.unwrapped.P[state][action] -> list cac tuple
  (probability, next_state, reward, terminated)
- policy: co the la
    * deterministic: mang 1 chieu, kich thuoc (n_states,), moi phan tu la
      chi so action (vi du policy[s] = 2)
    * stochastic: mang 2 chieu, kich thuoc (n_states, n_actions), moi hang
      la mot phan phoi xac suất tren cac action (tong = 1)
"""

from time import perf_counter

import numpy as np


# ---------------------------------------------------------------------------
# PHAN A - Markov chain utilities (dung o Bai 01-06)
# ---------------------------------------------------------------------------

def validate_transition_matrix(P, tol=1e-10):
    """Kiem tra P co phai la mot transition matrix hop le hay khong.

    Dieu kien:
        1. P la ma tran vuong.
        2. Moi phan tu thuoc [0, 1].
        3. Tong moi hang xap xi 1.
    """
    P = np.asarray(P, dtype=float)

    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        return False

    if np.any(P < -tol) or np.any(P > 1 + tol):
        return False

    row_sums = P.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=tol):
        return False

    return True


def state_distribution(p0, P, n_steps):
    """Tinh phan phoi trang thai sau n_steps buoc cua Markov chain.

    p_(t+1) = p_t @ P
    """
    p0 = np.asarray(p0, dtype=float)
    P = np.asarray(P, dtype=float)

    p = p0.copy()
    for _ in range(n_steps):
        p = p @ P
    return p


def sample_next_state(current_state, P, rng):
    """Sinh mot trang thai ke tiep bang cach lay mau tu hang current_state."""
    P = np.asarray(P, dtype=float)
    n_states = P.shape[0]
    probabilities = P[current_state]
    return int(rng.choice(n_states, p=probabilities))


# ---------------------------------------------------------------------------
# PHAN B - Return va discount factor (dung o Bai 07-11)
# ---------------------------------------------------------------------------

def compute_return(rewards, gamma):
    """Tinh discounted return G_0 tu mot chuoi reward.

    G_0 = R_1 + gamma * R_2 + gamma^2 * R_3 + ...
    """
    G = 0.0
    for t, r in enumerate(rewards):
        G += (gamma ** t) * r
    return G


def discounted_returns(rewards, gamma):
    """Tinh return tai moi timestep G_0, G_1, ..., G_(T-1).

    Duyet tu cuoi episode ve dau de tan dung cong thuc de quy:
        G_t = R_(t+1) + gamma * G_(t+1)
    """
    G = 0.0
    returns = [0.0] * len(rewards)
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns


# ---------------------------------------------------------------------------
# PHAN C - MDP model nho tu xay dung (dung o Bai 12-15)
# ---------------------------------------------------------------------------

def validate_mdp(P, n_states, n_actions):
    """Kiem tra tong xac suat transition cua tung (state, action) bang 1."""
    for s in range(n_states):
        for a in range(n_actions):
            total_prob = sum(item[0] for item in P[s][a])
            if not np.isclose(total_prob, 1.0):
                print(f"Invalid transition at state={s}, action={a}")
                return False
    return True


def print_policy(policy, action_names=None):
    """In deterministic policy dang danh sach state -> action."""
    for s, a in enumerate(policy):
        label = action_names[a] if action_names is not None else a
        print(f"State {s}: {label}")


# ---------------------------------------------------------------------------
# PHAN D - Kham pha model FrozenLake (dung o Bai 16-20)
# ---------------------------------------------------------------------------

def describe_state(env, state):
    """In toan bo transition (moi action) xuat phat tu mot state."""
    P = env.unwrapped.P
    n_actions = env.action_space.n
    print(f"--- State {state} ---")
    for a in range(n_actions):
        print(f"  Action {a}:")
        for probability, next_state, reward, terminated in P[state][a]:
            print(
                f"    probability={probability:.3f}, "
                f"next_state={next_state}, reward={reward}, "
                f"terminated={terminated}"
            )


# ---------------------------------------------------------------------------
# PHAN E - Bellman backup va Policy Evaluation (Bai 21-25)
# ---------------------------------------------------------------------------

def q_from_v(env, V, state, action, gamma):
    """Bellman backup: tinh Q(s, a) tu state-value function V.

    Q(s, a) = sum_{s', r} p(s', r | s, a) * [r + gamma * V(s')]

    Luu y: khi transition la terminal (terminated=True), gia tri V(s') sau
    do khong duoc cong dao vi khong co hanh dong nao thuc su duoc thuc hien
    tiep tu trang thai ket thuc. Trong FrozenLake, V tai cac trang thai
    terminal (Hole/Goal) luon duoc giu bang 0 trong suot qua trinh, nhung
    ta van nhan tuong minh voi (not terminated) de code dung ca khi
    V(terminal) bi khoi tao khac 0.
    """
    P = env.unwrapped.P
    q = 0.0
    for probability, next_state, reward, terminated in P[state][action]:
        future = 0.0 if terminated else V[next_state]
        q += probability * (reward + gamma * future)
    return q


def action_values(env, V, state, gamma):
    """Tra ve vector Q(s, .) cho tat ca action tai mot state."""
    n_actions = env.action_space.n
    return np.array([q_from_v(env, V, state, a, gamma) for a in range(n_actions)])


def policy_evaluation_sweep(env, policy, V, gamma):
    """Thuc hien MOT sweep (mot lan duyet toan bo state) cua Policy Evaluation."""
    n_states = env.observation_space.n
    new_V = np.zeros(n_states)

    for s in range(n_states):
        if np.asarray(policy).ndim == 1:
            action = int(policy[s])
            new_V[s] = q_from_v(env, V, s, action, gamma)
        else:
            value = 0.0
            for a, action_prob in enumerate(policy[s]):
                if action_prob > 0:
                    value += action_prob * q_from_v(env, V, s, a, gamma)
            new_V[s] = value

    return new_V


def policy_evaluation(env, policy, gamma=0.99, theta=1e-8, max_iterations=10000):
    """Iterative Policy Evaluation: lap sweep den khi hoi tu.

    Tra ve (V, n_iterations).
    """
    n_states = env.observation_space.n
    V = np.zeros(n_states)

    for i in range(1, max_iterations + 1):
        new_V = policy_evaluation_sweep(env, policy, V, gamma)
        delta = np.max(np.abs(new_V - V))
        V = new_V
        if delta < theta:
            return V, i

    return V, max_iterations


# ---------------------------------------------------------------------------
# PHAN F - Policy Improvement va Policy Iteration (Bai 26-30)
# ---------------------------------------------------------------------------

def greedy_policy_from_value(env, V, gamma=0.99):
    """Xay dung deterministic greedy policy tu state-value function V."""
    n_states = env.observation_space.n
    policy = np.zeros(n_states, dtype=int)

    for s in range(n_states):
        q_values = action_values(env, V, s, gamma)
        policy[s] = int(np.argmax(q_values))

    return policy


def print_frozenlake_policy(env, policy, action_symbols=None):
    """In policy FrozenLake duoi dang luoi mui ten, danh dau Hole/Goal."""
    if action_symbols is None:
        action_symbols = {0: "<", 1: "v", 2: ">", 3: "^"}

    desc = env.unwrapped.desc
    n_rows, n_cols = desc.shape
    P = env.unwrapped.P

    lines = []
    for r in range(n_rows):
        row_symbols = []
        for c in range(n_cols):
            s = r * n_cols + c
            cell = desc[r, c].decode("utf-8")
            if cell == "H":
                row_symbols.append("H")
            elif cell == "G":
                row_symbols.append("G")
            else:
                row_symbols.append(action_symbols[int(policy[s])])
        lines.append(" ".join(row_symbols))

    print("\n".join(lines))


def policy_iteration(env, gamma=0.99, theta=1e-8, max_iterations=1000):
    """Policy Iteration: xen ke Policy Evaluation va Policy Improvement.

    Tra ve (policy, V, n_policy_iterations).
    """
    n_states = env.observation_space.n
    policy = np.zeros(n_states, dtype=int)  # khoi tao: luon chon action 0

    for i in range(1, max_iterations + 1):
        V, _ = policy_evaluation(env, policy, gamma=gamma, theta=theta)
        new_policy = greedy_policy_from_value(env, V, gamma=gamma)

        policy_stable = np.array_equal(new_policy, policy)
        policy = new_policy

        if policy_stable:
            return policy, V, i

    return policy, V, max_iterations


# ---------------------------------------------------------------------------
# PHAN G - Value Iteration (Bai 31-33)
# ---------------------------------------------------------------------------

def value_iteration_sweep(env, V, gamma):
    """Thuc hien mot sweep cua Value Iteration (Bellman optimality backup)."""
    n_states = env.observation_space.n
    new_V = np.zeros(n_states)

    for s in range(n_states):
        q_values = action_values(env, V, s, gamma)
        new_V[s] = np.max(q_values)

    return new_V


def value_iteration(env, gamma=0.99, theta=1e-8, max_iterations=10000):
    """Value Iteration hoan chinh.

    Tra ve (V, n_iterations, deltas) trong do deltas la lich su hoi tu.
    """
    n_states = env.observation_space.n
    V = np.zeros(n_states)
    deltas = []

    for i in range(1, max_iterations + 1):
        new_V = value_iteration_sweep(env, V, gamma)
        delta = np.max(np.abs(new_V - V))
        deltas.append(delta)
        V = new_V
        if delta < theta:
            return V, i, deltas

    return V, max_iterations, deltas


# ---------------------------------------------------------------------------
# PHAN H - Danh gia va so sanh (Bai 34-36)
# ---------------------------------------------------------------------------

def evaluate_policy_by_simulation(env, policy, n_episodes=1000, seed=42):
    """Danh gia mot policy bang cach chay mo phong nhieu episode.

    Ho tro ca deterministic policy (mang 1 chieu chi so action) va
    stochastic policy (mang 2 chieu phan phoi xac suat).
    """
    policy = np.asarray(policy)
    rewards = np.zeros(n_episodes)
    lengths = np.zeros(n_episodes, dtype=int)
    successes = 0
    rng = np.random.default_rng(seed)

    for ep in range(n_episodes):
        observation, info = env.reset(seed=int(seed + ep))
        terminated = False
        truncated = False
        total_reward = 0.0
        length = 0

        while not (terminated or truncated):
            if policy.ndim == 1:
                action = int(policy[observation])
            else:
                action = int(rng.choice(len(policy[observation]), p=policy[observation]))

            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            length += 1

        rewards[ep] = total_reward
        lengths[ep] = length
        if total_reward > 0:
            successes += 1

    return {
        "success_rate": successes / n_episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "min_length": int(np.min(lengths)),
        "max_length": int(np.max(lengths)),
    }


def timed_run(func, *args, **kwargs):
    """Chay func va do thoi gian thuc thi bang perf_counter.

    Tra ve (result, elapsed_seconds).
    """
    start = perf_counter()
    result = func(*args, **kwargs)
    elapsed = perf_counter() - start
    return result, elapsed
