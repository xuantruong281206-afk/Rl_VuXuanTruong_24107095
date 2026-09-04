"""
Bai 17 - In toan bo transition model cua state = 0.
"""

import gymnasium as gym

from mdp_utils import describe_state


def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    env.reset(seed=42)

    describe_state(env, state=0)

    env.close()


if __name__ == "__main__":
    main()
