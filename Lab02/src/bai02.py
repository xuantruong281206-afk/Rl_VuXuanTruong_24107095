"""
Bai 02 - Kiem tra tinh hop le cua transition matrix.
"""

import numpy as np

from bai01 import P
from mdp_utils import validate_transition_matrix


def main():
    print("P hop le:", validate_transition_matrix(P))

    invalid_P = np.array([
        [0.5, 0.6, 0.0],
        [0.3, 0.4, 0.3],
        [0.2, 0.3, 0.5],
    ])
    print("invalid_P hop le:", validate_transition_matrix(invalid_P))


if __name__ == "__main__":
    main()
