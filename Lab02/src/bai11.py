"""
Bai 11 - So sanh reward som (sequence_A) va reward tre (sequence_B),
tim khoang gamma ma B co return lon hon A.
"""

import numpy as np

from mdp_utils import compute_return


def main():
    sequence_A = [5, 0, 0, 0, 0]
    sequence_B = [0, 0, 0, 0, 10]

    gammas = np.linspace(0, 1, 1001)
    crossover_gammas = []

    for gamma in gammas:
        G_A = compute_return(sequence_A, gamma)
        G_B = compute_return(sequence_B, gamma)
        if G_B > G_A:
            crossover_gammas.append(gamma)

    if crossover_gammas:
        print(
            f"B co return lon hon A khi gamma >= "
            f"{min(crossover_gammas):.3f} (xap xi, buoc quet 0.001)"
        )
    else:
        print("Khong tim thay gamma nao trong [0, 1] ma B > A.")

    # Nhan xet: A cho reward ngay lap tuc (5) nen khong bi chiet khau, trong
    # khi B cho reward lon hon (10) nhung phai doi 4 buoc, nen bi nhan
    # gamma^4. Khi gamma nho, agent "thien vi" phan thuong gan (A thang).
    # Khi gamma du lon (gan 1), reward tuong lai gan nhu khong bi chiet
    # khau nen B (10 > 5) se thang.


if __name__ == "__main__":
    main()
