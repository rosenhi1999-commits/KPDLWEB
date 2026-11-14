# KPDLWEB — ABSA tiếng Việt

Đã hoàn thành 24 lượt đánh giá mô hình cổ điển, mỗi lượt trong thư mục PHUONGPHAP_lanN. Xem [hướng dẫn](HUONG_DAN.md), [phương pháp và quy tắc](RESEARCH_AND_RULES.md), [kết quả split 42](experiment_comparison.csv), [kết quả split tham khảo 2026](experiment_comparison_reference.csv) và [notebook đã chạy](submission.ipynb).

ABSA-F1 tốt nhất split 42: **0.782002** (Ensemble_lan1); split tham khảo 2026: **0.757043** (NBSVM_lan3). Đây là validation từ train, không phải điểm công khai/riêng tư. Chưa đạt 0.96.

PhoBERT joint và label-attention có cấu hình và mã riêng; trạng thái kiểm tra nằm trong smoke.json nếu đã chạy. Kiểm tra nhỏ không phải huấn luyện đầy đủ, không có F1 deep mới. ViSoBERT mới có cấu hình, chưa chạy. Điểm 0.885296 người dùng cung cấp chưa được tái hiện vì thiếu các checkpoint và mã nguồn phụ trợ của ensemble đó.

Chưa có test_blind.csv, chưa tạo submission.csv để nộp. Các số 0 trong submission_demo.csv chỉ minh họa định dạng. Không commit dữ liệu, model hoặc báo cáo riêng có văn bản train lên repo công khai. .gitignore không tự xóa dữ liệu đã được track từ trước.
