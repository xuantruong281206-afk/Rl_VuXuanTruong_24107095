# Lab03 - Monte Carlo Methods

## Thông tin sinh viên

- Họ tên: Vu Xuan Truong
- MSSV: 24107095
- Lớp: K18.AI&ROBOT

## Môi trường đã chọn

- Môi trường chính: `Blackjack-v1`
- Môi trường phụ: `FrozenLake-v1`
- Lý do lựa chọn: Blackjack là task episodic ngắn, state rời rạc, có reward ở cuối ván nên phù hợp trực tiếp với
  Monte Carlo prediction/control. FrozenLake được thêm vào vì state có thể lặp lại trong một episode, giúp thấy
  sự khác nhau giữa First-Visit và Every-Visit (trong Blackjack hai phương pháp trùng nhau).

## Mục tiêu

Học value function và policy trực tiếp từ các episode (model-free): sinh episode, tính return, First/Every-Visit MC
prediction, ước lượng Q(s,a), epsilon-greedy và on-policy MC control; đánh giá policy học được.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Cách chạy

```bash
cd src
python main.py          # chạy mini-project (Bài 36) + tạo toàn bộ hình trong ../figures
python bai01.py         # chạy từng bài (bai01.py ... bai36.py)
```

Hoặc mở `notebooks/Lab03_MSSV_HoTen.ipynb` trên Google Colab / Jupyter và chạy từ đầu đến cuối.
Lần đầu train 300.000 episode mất vài phút; kết quả được cache trong `data/`.

Cấu trúc code: `src/mc_utils.py` (thuật toán), `src/experiments.py` (thí nghiệm + vẽ hình), `src/main.py`
(mini-project), `src/baiXX.py` (từng bài).

## Episode và Return

Episode là list `(state, action, reward)` do `generate_episode` sinh ra. Return tính ngược từ cuối episode:
`G_t = R_(t+1) + γ·G_(t+1)` (`compute_returns`). Blackjack chỉ có reward ở bước cuối nên `G_0 = γ^(T-1)·R_T`.

## First-Visit MC

`first_visit_mc_prediction`: với mỗi episode, chỉ cập nhật return tại lần đầu state xuất hiện; `V(s)` = trung bình return.

## Every-Visit MC

`every_visit_mc_prediction`: cập nhật ở mọi lần state xuất hiện. Trên Blackjack kết quả trùng First-Visit
(mean |ΔV| = 0) vì state không lặp lại trong một episode; trên FrozenLake có chênh lệch nhỏ.

## Action-value Q(s,a)

`mc_action_value_prediction` ước lượng `Q[state][action]` cho một policy cố định; `greedy_action` chọn `argmax`.

## Epsilon-greedy

`epsilon_greedy_action`: xác suất ε chọn ngẫu nhiên, còn lại greedy ⇒ P(greedy) = 1 − ε + ε/|A|.
Hình `figures/epsilon_comparison.png` so sánh các ε ∈ {0.01, 0.05, 0.10, 0.20, 0.50}.

## On-policy MC Control

`on_policy_mc_control`: sinh episode bằng epsilon-greedy theo Q hiện tại → tính return → first-visit cập nhật
Q(s,a) bằng incremental mean → policy được cải thiện ở episode kế tiếp. Train 300.000 episode, ε = 0.1, γ = 1.

## Kết quả

Đánh giá trên 50.000 episode mới (seed 123). *Các số dưới đây lấy từ một lần chạy mẫu; hãy cập nhật theo kết quả
máy của bạn sau khi chạy `main.py`.*

| Policy | Win | Loss | Draw | Mean reward |
|---|---:|---:|---:|---:|
| Random | 0.28 | 0.68 | 0.04 | ≈ -0.40 |
| Fixed (stick ≥ 20) | 0.29 | 0.65 | 0.06 | ≈ -0.35 |
| Fixed (stick ≥ 17) | 0.41 | 0.48 | 0.11 | ≈ -0.08 |
| MC learned | 0.42 | 0.48 | 0.10 | ≈ -0.06 |

## Learning Curve

`figures/mc_convergence.png`: moving average (window 1000) của reward lúc train tăng từ ≈ -0.22 lên gần -0.04
(vẫn còn explore với ε = 0.1 nên thấp hơn reward lúc đánh giá greedy).

## So sánh policy

`figures/policy_performance.png`. MC learned vượt xa random và fixed `stick ≥ 20`, và nhỉnh hơn heuristic
`stick ≥ 17`. Mean reward tối ưu vẫn âm vì Blackjack có lợi thế nhà cái.

## Nhận xét

- Policy học được gần với basic strategy ở vùng không có usable ace; vùng có usable ace còn nhiễu do state hiếm.
- Chênh lệch giữa các ε nhỏ so với sai số đánh giá (±0.013); ε quá nhỏ (0.01) có xu hướng kém hơn.
- Hạn chế: phải chờ hết episode, phương sai lớn, ε cố định nên không đảm bảo tối ưu ⇒ động lực cho Lab04 (TD).

## Tài liệu tham khảo

- Sutton & Barto, *Reinforcement Learning: An Introduction*, chương 5 (Monte Carlo Methods).
- Tài liệu học phần Học tăng cường; Gymnasium documentation (Blackjack, FrozenLake).
