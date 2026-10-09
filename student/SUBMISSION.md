# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Lê Trung Kiên
- MSSV: 2A202602748
- Email: kien.letrung@vinuni.edu.vn
- Link repo (fork): https://github.com/keybeand/K4-L2L3-DAY23-LeTrungKien-2A202602748-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`):

## Tóm tắt kết quả

- `fusion_mode`: `compare`, `frames`: `[0, 198]`, `segment`: `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`, `seed`: `0`
- `detection.precision`: `0.9701`, `detection.recall`: `0.7004`, `detection.tp/fp/fn`: `519 / 16 / 222`
- `tracking.lidar`: `rmse` = `0.1503 m`, `matches` = `502`, `sum_sq_err` = `11.3437`, `ghost_track_frames` = `0`, `missed_gt_frames` = `239`, `mean_confirmed_tracks` = `2.5226`
- `tracking.fused`: `rmse` = `0.1359 m`, `matches` = `502`, `sum_sq_err` = `9.2668`, `ghost_track_frames` = `0`, `missed_gt_frames` = `239`, `mean_confirmed_tracks` = `2.5226`
- **Giải thích khác biệt hai mode:** Chế độ Fused (LiDAR + Camera) bổ sung các phép đo 2D pixel từ Camera FRONT giúp tinh chỉnh lại ước lượng trạng thái EKF, làm giảm sai số vị trí RMSE từ `0.1503 m` xuống `0.1359 m` (tăng độ chính xác định vị) trong khi số cặp ghép (`502 matches`) và số lượng ghost track (`0 ghosts`) không thay đổi.

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. **Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)**:
   - LiDAR đo trực tiếp vị trí 3D $(x, y, z)$ trong khung cảm biến, vector đo $z$ dạng 3x1, ma trận đo $H$ tuyến tính 3x6 và ma trận nhiễu $R$ kích thước 3x3.
   - Camera đo tọa độ pixel 2D $(u, v)$ trên mặt phẳng ảnh, vector đo $z$ dạng 2x1. Hàm đo $h(x)$ phi tuyến theo mô hình camera lỗ kim (pinhole), ma trận đo $H$ là ma trận Jacobian 2x6 và ma trận nhiễu $R$ kích thước 2x2.

2. **Vì sao cần gating Mahalanobis trước khi gán**:
   - Khoảng cách Mahalanobis tính tới ma trận hiệp biến $S$ (hiệp biến của đo lường và trạng thái). Cổng $\chi^2$ loại bỏ sớm các đo lường bất thường nằm xa ngoài phân bố thống kê tin cậy trước khi gán greedy, tránh lãng phí chi phí tính toán chiếu điểm không cần thiết và ngăn chặn việc gán nhầm đối tượng.

3. **Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`**:
   - Pipeline là **track-then-fuse**: Chỉ có duy nhất một danh sách track chính. Mỗi frame, EKF `predict` một lần duy nhất, sau đó thực hiện EKF `update` lượt LiDAR (AssocL), rồi EKF `update` lượt Camera (AssocC) trên cùng các track đó. Trên log `fusion-run-lab`, mỗi frame được xử lý theo trình tự dự báo một lần rồi cập nhật nối tiếp các cảm biến sẵn có.

4. **Nếu camera lệch calibration, triệu chứng gì trên innovation/residual**:
   - Khi camera bị lệch calibration (extrinsic hoặc intrinsic), phép chiếu $h(x)$ sẽ tính ra tọa độ pixel sai so với vị trí thực tế của vật thể. Điều này khiến vector residual/innovation $\gamma = z - h(x)$ tăng lớn bất thường, có thể làm cặp đo bị Cổng $\chi^2$ loại bỏ (không gán được) hoặc làm cập nhật EKF kéo lệch vị trí ước lượng của track.

5. **Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng? Giải thích vì sao lidar quyết định score/init/delete còn camera chỉ EKF update**:
   - `associate_and_update` cần `sensor` tường minh kể cả khi `meas_list` rỗng để truyền tham số cảm biến chuẩn xác cho `manager.manage_tracks(...)` biết cảm biến nào vừa kết thúc lượt. LiDAR đo trực tiếp không gian 3D tin cậy nên được chọn làm căn cứ duy nhất quyết định tồn tại/khởi tạo/xóa track (tính score); Camera có FOV hẹp và chỉ đo 2D góc nhìn nên chỉ đóng vai trò tinh chỉnh trạng thái hình học EKF khi vật thể nằm trong tầm nhìn, không được cộng/trừ score hay xóa/tạo track.

6. **Nêu điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xóa track**:
   - **Xác nhận (Confirm):** Một lượt LiDAR hit cộng $+1/window$ score; nếu `score > confirmed_threshold` thì chuyển sang trạng thái `"confirmed"`.
   - **Giữ Confirmed sau Miss:** Khi track đã ở trạng thái `"confirmed"`, nếu gặp LiDAR miss trong FOV thì score bị trừ $-1/window$ nhưng trạng thái vẫn giữ nguyên `"confirmed"` (không bị hạ về tentative).
   - **Xóa Track:** Xóa khi rơi vào một trong các trường hợp (OR):
     - Hiệp biến vị trí bị bùng nổ: $P_{xx} > max\_P$ hoặc $P_{yy} > max\_P$.
     - Track confirmed có `score < delete_threshold`.
     - Track chưa confirmed có `score <= 0`.

## Bonus (không bắt buộc)

Không

## Khai báo sử dụng AI (bắt buộc)

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Antigravity AI (Gemini 3.6 Flash)
- Dùng cho phần nào: Đọc hiểu tài liệu đề bài, lập kế hoạch thực hiện theo từng checkpoint, hướng dẫn cài đặt môi trường và hỗ trợ viết code Part E-H trong `student/workspace/`.
- Cách bạn đã kiểm tra lại: Tự chạy kiểm tra suite `pytest student/tests` pass 100% (128/128 passed), chạy tích hợp Waymo thành công qua `fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0` và đối chiếu kết quả trong `metrics.json`.


## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: không bắt buộc sửa (hoặc ghi chú nếu bạn đã sửa)
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [x] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [x] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))

