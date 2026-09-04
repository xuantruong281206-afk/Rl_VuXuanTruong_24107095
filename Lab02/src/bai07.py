"""
Bai 07 - Undiscounted return (gamma = 1.0).
"""

from mdp_utils import compute_return


def main():
    rewards = [1, 1, 1, 1, 1]
    G = compute_return(rewards, gamma=1.0)
    print(f"rewards = {rewards}")
    print(f"Return (gamma=1.0): {G}")


if __name__ == "__main__":
    main()
