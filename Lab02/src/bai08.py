"""
Bai 08 - Discounted return voi nhieu gia tri gamma.
"""

from mdp_utils import compute_return


def main():
    rewards = [1, 1, 1, 1, 1]
    gammas = [0.0, 0.5, 0.9, 0.99, 1.0]

    print(f"{'Gamma':>8s} | {'Return':>8s}")
    print("-" * 21)
    for gamma in gammas:
        G = compute_return(rewards, gamma)
        print(f"{gamma:8.2f} | {G:8.4f}")


if __name__ == "__main__":
    main()
