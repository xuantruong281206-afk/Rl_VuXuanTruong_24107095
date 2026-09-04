"""
Bai 22 - Tinh vector Q(s, .) cho toan bo action cua mot state.
"""

import numpy as np
import gymnasium as gym

from mdp_utils import action_values


def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    env.reset(seed=42)

    V = np.zeros(env.observation_space.n)

    for state in [0, 5, 14]:
        q_values = action_values(env, V, state, gamma=0.99)
        print(f"state={state}: Q = {q_values}")

    env.close()


if __name__ == "__main__":
    main()
