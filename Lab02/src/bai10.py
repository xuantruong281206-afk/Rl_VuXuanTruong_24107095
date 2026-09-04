"""
Bai 10 - Anh huong cua gamma len G_0, ve bieu do va luu vao figures/.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mdp_utils import compute_return

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def main():
    rewards = [0, 0, 0, 0, 10]
    gammas = np.linspace(0, 1, 101)

    g0_values = [compute_return(rewards, gamma) for gamma in gammas]

    plt.figure(figsize=(7, 5))
    plt.plot(gammas, g0_values, color="tab:blue")
    plt.title("Anh huong cua discount factor gamma len G_0")
    plt.xlabel("gamma")
    plt.ylabel("G_0 (discounted return)")
    plt.grid(True)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, "gamma_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Da luu bieu do vao: {out_path}")


if __name__ == "__main__":
    main()
