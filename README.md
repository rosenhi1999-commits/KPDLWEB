# KPDLWEB — ABSA tiếng Việt

**Tự chạy trên GitHub:** [CI_AUTORUN.md](CI_AUTORUN.md) mô tả workflow full fine-tuning 8 cấu hình và tự tổng hợp/stacking. Kết quả mới chỉ được tính khi job huấn luyện hoàn tất; không dùng smoke test làm F1.

Lượt tự động đầu tiên đã được kích hoạt: [GitHub Actions run 43590132635](https://github.com/rosenhi1999-commits/KPDLWEB/actions/runs/43590132635). Xem trạng thái trực tiếp ở liên kết; kết quả hoàn tất sẽ có CI_TRAINING_REPORT.md.

**Phân tích bài mẫu 0.88:** xem [REFERENCE_088_ANALYSIS.md](REFERENCE_088_ANALYSIS.md). Đã chuẩn bị 8 cấu hình full fine-tuning và công cụ hiệu chỉnh ngưỡng trên tune; đây là các lượt chưa huấn luyện, không cộng vào 46 kết quả đã đo dưới đây.

Đã hoàn thành **46 lượt đo**, mỗi lượt trong thư mục PHUONGPHAP_lanN. Đợt mới chỉ dùng train.csv, không đọc hay dự đoán blind. Xem [báo cáo mới](TRAIN_ONLY_REPORT.md), [hướng dẫn tái hiện](TRAIN_ONLY_HUONG_DAN.md), [toàn bộ kết quả](all_experiments.csv) và [notebook đã chạy](submission.ipynb).

ABSA-F1 tốt nhất split 42: **0.782002** (Ensemble_lan1); split nhóm 2026: **0.799716** (Stacking_lan1), tăng từ 0.757043 trên cùng validation. Đây là validation đã được xem trong quá trình phát triển, không phải điểm công khai/riêng tư. **Chưa đạt 0.96.**

Đã học classifier/attention head trên đặc trưng PhoBERT đóng băng, thử PyVi và RDR, multi-task sparse, stacking và thêm 640 mẫu tổng hợp. Chưa fine-tune toàn bộ encoder PhoBERT/ViSoBERT. Hai thư mục PhoBERT_lan1 và PhoBERT_lan2 ban đầu là smoke, không phải F1 mới; các lượt học head thực tế là PhoBERT_Attention_lanN. Điểm 0.885296 trong tài liệu người dùng chưa được tái hiện do thiếu checkpoint và mã phụ trợ của ensemble đó.

Chưa có test_blind.csv, chưa tạo submission.csv để nộp. Các số 0 trong submission_demo.csv chỉ minh họa định dạng. Không commit dữ liệu, model hoặc báo cáo riêng có văn bản train lên repo công khai. .gitignore không tự xóa dữ liệu đã được track từ trước.
