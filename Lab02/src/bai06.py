"""
Bai 06 - So sanh phan phoi ly thuyet va phan phoi mo phong.
"""

import numpy as np

from bai01 import P, STATE_NAMES
from mdp_utils import sample_next_state, state_distribution


def main():
    rng = np.random.default_rng(42)
    n_transitions = 100_000
    n_states = P.shape[0]

    counts = np.zeros(n_states)
    current_state = 0
    counts[current_state] += 1

    for _ in range(n_transitions):
        current_state = sample_next_state(current_state, P, rng)
        counts[current_state] += 1

    empirical = counts / counts.sum()

    p0 = np.array([1.0, 0.0, 0.0])
    theoretical = state_distribution(p0, P, 200)  # 200 buoc coi nhu hoi tu

    print("Phan phoi thuc nghiem:", np.round(empirical, 4))
    print("Phan phoi ly thuyet (sau nhieu buoc):", np.round(theoretical, 4))
    print("Sai lech tuyet doi lon nhat:", np.max(np.abs(empirical - theoretical)))

    # Nhan xet:
    # - Sau du nhieu transition, phan phoi thuc nghiem (empirical) hoi tu rat
    #   gan voi phan phoi dung tinh bang p0 @ P^n (theoretical).
    # - Day chinh la stationary distribution cua Markov chain: phan phoi
    #   khong doi khi tiep tuc nhan them P (pi @ P = pi).
    # - So luong mau cang lon thi sai lech giua thuc nghiem va ly thuyet
    #   cang nho, phu hop voi luat so lon (law of large numbers).


if __name__ == "__main__":
    main()
