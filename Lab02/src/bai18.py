"""
Bai 18 - Kiem thu ham describe_state() voi nhieu state khac nhau.
"""

import gymnasium as gym

from mdp_utils import describe_state


def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    env.reset(seed=42)

    for state in [0, 1, 14]:
        describe_state(env, state)
        print()

    env.close()


if __name__ == "__main__":
    main()
