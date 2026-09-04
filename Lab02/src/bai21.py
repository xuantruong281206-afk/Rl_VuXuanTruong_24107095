"""
Bai 21 - Mot Bellman backup: tinh Q(s, a) tu V.
"""

import numpy as np
import gymnasium as gym

from mdp_utils import q_from_v


def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    env.reset(seed=42)

    V = np.zeros(env.observation_space.n)
    gamma = 0.99

    for action in range(env.action_space.n):
        q = q_from_v(env, V, state=0, action=action, gamma=gamma)
        print(f"Q(state=0, action={action}) = {q:.4f}")

    env.close()


if __name__ == "__main__":
    main()
