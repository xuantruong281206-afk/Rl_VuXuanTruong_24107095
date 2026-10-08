# data/

Thư mục chứa dữ liệu sinh ra khi chạy code (không có dataset ngoài – Gymnasium tự sinh episode).

- `mc_control_<n_episodes>_<gamma>_<epsilon>_<seed>.pkl`: cache Q-table/policy/episode_rewards của
  `load_or_train_control()` để các bài 33–36 không phải train lại. Xóa file này nếu muốn train lại từ đầu.
