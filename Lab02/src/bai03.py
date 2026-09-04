"""
Bai 03 - Tinh xac suat trang thai ke tiep sau mot buoc.
"""

import numpy as np

from bai01 import P, STATE_NAMES


def main():
    p0 = np.array([1.0, 0.0, 0.0])  # chac chan bat dau tu Sunny

    p1 = p0 @ P  # KHONG hard-code ket qua, tinh bang phep nhan ma tran

    print("Phan phoi ban dau p0:", p0)
    print("Phan phoi sau 1 buoc p1:")
    for name, prob in zip(STATE_NAMES, p1):
        print(f"  {name}: {prob:.3f}")


if __name__ == "__main__":
    main()
