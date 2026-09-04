"""
Bai 13 - Kiem tra model MDP tu xay dung o Bai 12.
"""

from bai12 import N_ACTIONS, N_STATES, P
from mdp_utils import validate_mdp


def main():
    is_valid = validate_mdp(P, N_STATES, N_ACTIONS)
    print("MDP hop le:", is_valid)

    # Thu MDP loi de kiem tra ham phat hien sai
    bad_P = {
        0: {0: [(0.5, 0, 0.0, False)], 1: [(1.0, 1, 0.0, False)]},
        1: {0: [(1.0, 0, 0.0, False)], 1: [(1.0, 1, 0.0, False)]},
    }
    print("bad_P hop le:", validate_mdp(bad_P, 2, 2))


if __name__ == "__main__":
    main()
