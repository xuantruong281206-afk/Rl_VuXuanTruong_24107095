"""
Bai 04 - Distribution sau nhieu buoc.
"""

import numpy as np

from bai01 import P, STATE_NAMES
from mdp_utils import state_distribution


def main():
    p0 = np.array([1.0, 0.0, 0.0])

    for t in [1, 2, 5, 10, 50]:
        p_t = state_distribution(p0, P, t)
        formatted = ", ".join(
            f"{name}={prob:.4f}" for name, prob in zip(STATE_NAMES, p_t)
        )
        print(f"t={t:3d}: {formatted}")


if __name__ == "__main__":
    main()
