"""
Bai 15 - Stochastic policy: uniform random policy cho MDP nho.
"""

import numpy as np

from bai12 import N_ACTIONS, N_STATES


def main():
    policy = np.ones((N_STATES, N_ACTIONS)) / N_ACTIONS

    print("Stochastic policy (uniform):")
    print(policy)

    row_sums = policy.sum(axis=1)
    print("Tong xac suat moi state:", row_sums)
    print("Tat ca deu bang 1:", np.allclose(row_sums, 1.0))


if __name__ == "__main__":
    main()
