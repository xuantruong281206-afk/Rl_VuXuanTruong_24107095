"""
Bai 12 - Xay dung mot MDP nho gom 2 state va 2 action.

State 0: "Nghi ngoi"
State 1: "Lam viec"
Action 0: "Thu dong"
Action 1: "Chu dong"

P[state][action] = list cac tuple (probability, next_state, reward, terminated)
"""

N_STATES = 2
N_ACTIONS = 2

P = {
    0: {
        0: [(1.0, 0, 0.0, False)],                       # o lai state 0, khong reward
        1: [(0.7, 1, 1.0, False), (0.3, 0, 0.0, False)],  # co gang, hay chuyen sang lam viec
    },
    1: {
        0: [(1.0, 0, 2.0, False)],                        # hoan thanh viec, tro ve nghi, reward lon
        1: [(0.5, 1, -1.0, False), (0.5, 0, 2.0, False)],  # co gang lam tiep, ton kem, hoac hoan thanh
    },
}


def main():
    print(f"So state: {N_STATES}, so action: {N_ACTIONS}")
    for s in P:
        for a in P[s]:
            print(f"P[{s}][{a}] = {P[s][a]}")


if __name__ == "__main__":
    main()
