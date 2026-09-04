"""
Bai 16 - Thong tin co ban cua FrozenLake-v1 (stochastic).
"""

import gymnasium as gym


def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)

    observation, info = env.reset(seed=42)

    print("Number of states:", env.observation_space.n)
    print("Number of actions:", env.action_space.n)
    print("Initial observation:", observation)

    env.close()


if __name__ == "__main__":
    main()
