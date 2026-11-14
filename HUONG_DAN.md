# Thử nghiệm ABSA

Python đã dùng: 3.12.14. Chạy từ thư mục gốc repo:

```bash
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python TFIDF_lan1/train.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python TFIDF_lan2/train.py
```

Đặt train.csv ở gốc trên máy riêng. Không tải dữ liệu lớp lên repo công khai. Mỗi lần thử mới phải dùng thư mục mới theo mẫu PHUONGPHAP_lanN. Không commit model, split chứa id hoặc dữ liệu gốc lên repo công khai.

Chuẩn hóa NFC, chữ thường, gộp khoảng trắng; giữ dấu, phủ định, dấu câu và placeholder. Kết hợp TF-IDF từ 1–2 gram với ký tự 3–5 gram để xử lý viết tắt và không dấu. Chỉ fit vectorizer trên phần train. Chia 3750/938 mẫu bằng seed 42, cùng tập validation cho cả hai hướng. Chưa dùng phân tầng đa nhãn. Chưa kiểm tra trùng gần; không có trùng văn bản sau chuẩn hóa NFC/chữ thường/khoảng trắng.

| Thử nghiệm | Mô hình | ABSA-F1 validation |
|---|---|---|
| TFIDF_lan1 | Logistic Regression C=4 | 0.674122 |
| TFIDF_lan2 | Linear SVM C=1, balanced | 0.761099 |

Đây là validation từ train, không phải điểm công khai/riêng tư. Điểm chính trung bình F1 của 14 nhãn khen/chê. Ngưỡng Logistic Regression mặc định; SVM dùng dấu của decision function. Nhãn nhắc được sửa để bao hàm khen/chê.

Scripts tạo metrics.json, split.csv và validation_model.joblib trong từng thư mục. Model chỉ fit 80% train, chưa phải model nộp bài. Chưa tạo submission.csv. Repo lúc kiểm tra chưa có test_blind.csv hoặc DeBai_ABSA.docx.

Mục tiêu 0.96 chưa đạt và không được bảo đảm. Môi trường hiện chưa có torch/transformers; PhoBERT chưa được chạy. PhoBERT cần pipeline tách từ tiếng Việt phù hợp tokenizer, giữ dấu và phủ định, phân loại đa nhãn bằng sigmoid/BCE; chỉ fine-tune từ dữ liệu train. Không gửi văn bản test vào LLM. Không tối ưu liên tục theo cùng validation mà coi điểm đó là ước lượng độc lập.

Chưa phải bộ nộp hoàn chỉnh: còn thiếu notebook có output, bootstrap CI, kappa/MCC/balanced accuracy, phân tích lỗi, cấu hình phần cứng đầy đủ, model fit toàn bộ train và kiểm tra submission theo score_absa.py. Không có dữ liệu bổ sung sinh bằng LLM trong các lần thử này.
