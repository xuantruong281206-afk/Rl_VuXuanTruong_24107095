# Lab02 - Markov Decision Process va Dynamic Programming

## Thong tin sinh vien

- Ho ten: Vu Xuan Truong
- MSSV: 24107095
- Lop: K18.AI&ROBOT
- GitHub username: https://github.com/xuantruong281206-afk

## Muc tieu

Lab02 chuyen tu viec tuong tac voi moi truong (Lab01) sang mo hinh hoa va
giai bai toan hoc tang cuong bang **Markov Decision Process (MDP)**:

- Bieu dien Markov chain va MDP bang NumPy / Python.
- Tinh return, state-value function `V(s)`, state-action value function
  `Q(s,a)`.
- Cai dat Bellman backup, Policy Evaluation, Policy Improvement, Policy
  Iteration va Value Iteration **tu dau**, khong dung thu vien RL co san.
- Giai `FrozenLake-v1` bang Dynamic Programming va danh gia optimal policy
  bang mo phong.

## Cau truc thu muc

```text
Lab02/
├── README.md
├── requirements.txt
├── src/
│   ├── bai01.py ... bai36.py   # 36 bai tap
│   ├── mdp_utils.py            # cac ham DP dung chung
│   └── main.py                 # chuong trinh tong hop (= bai36.py)
├── notebooks/
│   └── Lab02_MSSV_HoTen.ipynb
├── figures/                    # bieu do sinh ra khi chay code
└── data/
    └── README.md
```

## Cai dat

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
cd Lab02
pip install -r requirements.txt
```

## Cach chay

Chay tung bai rieng le, vi du:

```bash
python src/bai01.py
python src/bai24.py
python src/bai29.py
python src/bai32.py
```

Chay chuong trinh tong hop (Value Iteration + Policy Iteration + danh gia +
bieu do hoi tu, tren ca hai truong hop `is_slippery=False` va `True`):

```bash
python src/main.py
```

Mo notebook:

```bash
jupyter notebook notebooks/Lab02_MSSV_HoTen.ipynb
```

## Thuat toan da cai dat (trong `src/mdp_utils.py`)

### Policy Evaluation

`policy_evaluation_sweep()` thuc hien mot sweep Bellman expectation backup
cho toan bo state; `policy_evaluation()` lap lai sweep nay den khi
`delta = max|V_new - V_old| < theta`, tra ve `(V, n_iterations)`.

### Policy Iteration

`policy_iteration()` xen ke Policy Evaluation va Policy Improvement
(`greedy_policy_from_value()`) cho den khi policy khong doi
(`policy_stable`), tra ve `(policy, V, n_policy_iterations)`.

### Value Iteration

`value_iteration()` thuc hien Bellman optimality backup
(`new_V[s] = max_a Q(s,a)`) lap lai den khi hoi tu, tra ve
`(V, n_iterations, deltas)`.

## Ket qua FrozenLake

Chay `python src/main.py` de xem:

- Value table va policy dang luoi 4x4 cho ca hai thuat toan.
- Ket qua danh gia bang mo phong (`success_rate`, `mean_reward`,
  `mean_length`) tren >= 1000 episode.
- Bieu do hoi tu luu tai `figures/value_iteration_convergence.png` va
  `figures/policy_iteration_convergence.png`.

`<Dien ket qua so cu the sau khi ban tu chay chuong trinh tren may minh>`

## So sanh Value Iteration va Policy Iteration

| Thuat toan | So vong lap | Thoi gian | Success rate | Mean reward |
|---|---:|---:|---:|---:|
| Value Iteration | `<dien>` | `<dien>` | `<dien>` | `<dien>` |
| Policy Iteration | `<dien>` | `<dien>` | `<dien>` | `<dien>` |

Nhan xet chi tiet xem trong `src/bai35.py` (phan comment cuoi ham `main()`)
va trong notebook.

## Nhan xet

`<Sinh vien tu viet nhan xet sau khi chay va quan sat ket qua thuc te>`

## Tai lieu tham khao

- Gymnasium documentation: https://gymnasium.farama.org/
- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd edition),
  chuong 3 (MDP) va chuong 4 (Dynamic Programming).
