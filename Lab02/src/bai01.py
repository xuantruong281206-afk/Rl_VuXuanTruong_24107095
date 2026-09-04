"""
Bai 01 - Tao transition matrix cho Markov chain thoi tiet.

3 trang thai: Sunny, Cloudy, Rainy.
"""

import numpy as np

STATE_NAMES = ["Sunny", "Cloudy", "Rainy"]

# Moi hang la phan phoi xac suat chuyen tu trang thai hien tai sang cac
# trang thai ke tiep. Tong moi hang phai bang 1.
P = np.array([
    [0.7, 0.2, 0.1],   # tu Sunny
    [0.3, 0.4, 0.3],   # tu Cloudy
    [0.2, 0.3, 0.5],   # tu Rainy
])


def main():
    print("Transition matrix P (hang = trang thai hien tai):")
    print(f"{'':10s}" + "".join(f"{name:>10s}" for name in STATE_NAMES))
    for i, name in enumerate(STATE_NAMES):
        print(f"{name:10s}" + "".join(f"{p:10.2f}" for p in P[i]))

    print("\nTong moi hang:", P.sum(axis=1))


if __name__ == "__main__":
    main()
