"""
Bai 19 - Kiem tra tong xac suat transition cua moi (state, action) bang 1.
"""

import numpy as np
import gymnasium as gym


def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    env.reset(seed=42)

    P = env.unwrapped.P
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    all_valid = True
    for s in range(n_states):
        for a in range(n_actions):
            probabilities = [item[0] for item in P[s][a]]
            total = sum(probabilities)
            if not np.isclose(total, 1.0):
                print(f"Loi: state={s}, action={a}, tong xac suat={total}")
                all_valid = False

    print("Tat ca (state, action) deu co tong xac suat = 1:", all_valid)

    env.close()


if __name__ == "__main__":
    main()
