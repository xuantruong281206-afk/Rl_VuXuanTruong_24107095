"""
Bai 20 - So sanh deterministic (is_slippery=False) va
stochastic (is_slippery=True) FrozenLake.
"""

import gymnasium as gym

ACTION_NAMES = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}


def show_transitions(is_slippery, state, action):
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=is_slippery)
    env.reset(seed=42)

    transitions = env.unwrapped.P[state][action]
    print(f"is_slippery={is_slippery}, state={state}, action={ACTION_NAMES[action]}")
    print(f"  So transition co the: {len(transitions)}")
    for probability, next_state, reward, terminated in transitions:
        print(
            f"    probability={probability:.3f}, next_state={next_state}, "
            f"reward={reward}, terminated={terminated}"
        )

    env.close()


def main():
    state = 0
    action = 2  # RIGHT

    show_transitions(False, state, action)
    show_transitions(True, state, action)

    # Ket luan:
    # - Khi is_slippery=False, moi (state, action) chi dan den DUY NHAT mot
    #   next_state voi xac suat 1.0: moi trong nay la deterministic.
    # - Khi is_slippery=True, mot action co the dan den 3 huong khac nhau
    #   (huong du dinh va 2 huong vuong goc) moi huong xac suat ~1/3: moi
    #   truong nay la stochastic, mo phong be mat bang truot.
    # - Vi vay khi lam Dynamic Programming voi is_slippery=True, khong the
    #   gia dinh mot action chi dan den mot next_state duy nhat; phai duyet
    #   va cong theo xac suat cua tat ca transition co the.


if __name__ == "__main__":
    main()
