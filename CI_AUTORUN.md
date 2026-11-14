# Tự chạy full fine-tuning trên GitHub Actions

Workflow: `.github/workflows/train-absa.yml`. Việc thêm workflow vào main sẽ kích hoạt lượt chạy đầu tiên. Có thể chạy lại bằng Run workflow trong tab Actions; các kết quả cũ vẫn có provenance run ID riêng. Workflow không dùng GPU hoặc larger runner; dùng standard ubuntu-24.04. [GitHub xác nhận standard runner miễn phí trên repo public](https://docs.github.com/en/actions/concepts/billing-and-usage).

**Cập nhật vòng 2:** vòng 1 đã hoàn tất 8 lượt, stacking đạt 0.860469. Matrix hiện chạy 5 lượt mới ở PhoBERT_lan19..22 và ViSoBERT_lan2, thêm label attention, ViSoBERT và refit outer80. Collector tải lại 8 nguồn cũ từ run 43590132635 và cấp thư mục stacking mới. Chi tiết trong CI_PHASE1_ANALYSIS.md; mô tả 8 cấu hình dưới đây là vòng 1.

## Những việc tự chạy

1. Chạy 8 cấu hình full encoder fine-tuning, tối đa 2 job song song. PhoBERT_lan3..10 là các cấu hình chuẩn bị; kết quả thật được ghi ở PhoBERT_lan11..18 tương ứng. Batch hiệu dụng 16 review, tối đa 8 epoch, early stop theo tune.
2. CPU được phép huấn luyện thật. Dùng BF16 nếu máy có avx512_bf16/amx_bf16; nếu không dùng FP32. Giữ learning rate/loss theo cấu hình ablation đã định. Mỗi job có timeout 355 phút; trainer dừng ở ranh giới epoch sau ngân sách 18000 giây và ghi chính xác số epoch đã hoàn thành.
3. Lưu checkpoint FP32 vào cache của run, artifact nhỏ gồm chỉ số và xác suất theo ID, không có review text hoặc dữ liệu gốc. Các checkpoint cache có thể bị GitHub thu hồi theo chính sách cache; không phải kho model vĩnh viễn.
4. Khi các job kết thúc, tự tổng hợp kết quả thành CI_TRAINING_REPORT.md, ci_training_comparison.csv và các metrics/config/history trong repo. Nếu có ít nhất 2 nguồn hoàn thành, fit Stacking_lan2 trên tune OOF, sau đó chấm cùng validation từ train.
5. Chọn encoder đóng gói bằng F1 tune sau hiệu chỉnh tại checkpoint đã chọn, không chọn theo outer validation. Collector đọc đúng cột tune_F1_tuned_threshold trong history để không so nhầm điểm ngưỡng cố định của lan17 với điểm đã hiệu chỉnh của các lượt khác. Đóng gói checkpoint dưới dạng float16 để giảm dung lượng, load lại vào model float32 và chấm lại validation; lưu chỉ số sau làm tròn trọng số cùng artifact `absa-selected-encoder`. Artifact giữ 7 ngày. Không giả định điểm checkpoint nén bằng điểm trước nén.

## Dữ liệu và cách đọc kết quả

Chỉ train.csv được dùng. Audit guard chặn mở CSV có tên blind. Tách nhóm 2026/2027 giữ nguyên 2812 fit, 938 tune và 938 validation. Không tạo submission.csv. Xác suất nguồn là đầu ra mô hình, không phải nhãn được sinh bằng LLM. Checkpoint encoder đã được chọn trên tune, do đó meta CV chưa phải nested encoder CV hoàn toàn. Validation đã được xem trong lịch sử phát triển.

Đây là 8 cấu hình joint encoder và stacking; không tuyên bố đã tái hiện đúng cả 5 kiến trúc của bài mẫu 0.88. Không có bảo đảm đạt 0.96. Chỉ các job có metrics.json và provenance đúng train hash mới được tổng hợp. Job thất bại không được gán điểm. Số epoch thực tế, stop reason và thời gian có trong metrics; không lấy smoke test làm F1.

## Theo dõi

Mở tab Actions → Train ABSA from train only → từng job Full fine-tune PhoBERT_lanN. Trainer ghi tiến độ mỗi 25 batch và F1 tune sau mỗi epoch. Sau cùng đọc CI_TRAINING_REPORT.md trong main và các artifacts của run. Artifact có model không chứa train.csv/test_blind.csv; pipeline không upload checkpoint optimizer lớn vào artifact.

Các lần trước đã chạy trên workspace CPU và đẩy kết quả lên repo; chúng không phải GitHub Actions. Workflow này mới là cơ chế GitHub tự thực thi và ghi kết quả.

Lượt đầu tiên: https://github.com/rosenhi1999-commits/KPDLWEB/actions/runs/43590132635 . Nếu muốn chạy trực tiếp trên máy CPU riêng, có thể dùng `python run_reference_trials.py --runs PhoBERT_lan3 --allow-cpu`; không còn cần GPU để cho phép huấn luyện.
