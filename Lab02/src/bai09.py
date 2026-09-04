"""
Bai 09 - Return tai moi timestep, tinh tu cuoi episode ve dau.
"""

from mdp_utils import discounted_returns


def main():
    rewards = [0, 0, 0, 1]
    gamma = 0.9

    returns = discounted_returns(rewards, gamma)

    for t, G_t in enumerate(returns):
        print(f"G_{t} = {G_t:.4f}")


if __name__ == "__main__":
    main()
