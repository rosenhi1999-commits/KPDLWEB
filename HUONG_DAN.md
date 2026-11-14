# Chạy lại thí nghiệm và chuẩn bị nộp

Python đã dùng: **3.12.14**, Linux x86_64, 3 CPU logic, RAM hệ thống khoảng 9.5 GiB, không GPU. Các phép tính sparse dùng OPENBLAS_NUM_THREADS=1 và OMP_NUM_THREADS=1. Torch kiểm tra trên CPU là 2.6.0+cpu.

## 1. Môi trường và dữ liệu cục bộ

Đặt train.csv và submission_demo.csv ở gốc repo. Dữ liệu train không được đóng trong zip nộp hoặc tải lên repo công khai. Không dùng blind để học, chọn mô hình, chọn ngưỡng hay phân tích lỗi.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 2. Hai suite độc lập

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_experiments.py --suite legacy
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_experiments.py --suite reference
```

Kết quả và model cục bộ mặc định vào private_runs/PHUONGPHAP_lanN. Các thư mục công khai cùng tên ở gốc chỉ chứa cấu hình/chỉ số của lượt đã chạy, không chứa checkpoint. Đừng coi các chỉ số JSON tải từ repo là bằng chứng code vừa được huấn luyện lại: chạy các lệnh trên để tạo lại model từ train.

Một lần chạy mới cần thư mục kết quả mới qua `--results-root private_runs_moi`, vì script từ chối ghi đè một lượt đã có. Trên máy đã dùng, phần fit/refit mỗi cấu hình sparse mất khoảng vài đến hơn 10 giây; tổng suite dài hơn do chuẩn hóa, tách từ, bootstrap và ghi model. Không có benchmark GPU. Thời gian trong metrics là từng phần đo thật; validation_inference_seconds trong runner chỉ tính model decision_function, chưa gồm chuẩn hóa/vectorizer. Notebook đo thêm suy luận end-to-end.

Suite legacy dùng outer split random seed 42 (3750/938), inner tune phân tầng đa nhãn. Suite reference dùng đúng GroupShuffleSplit SHA256 của clean(text).lower(), outer seed 2026 và inner seed 2027 như mã người dùng cung cấp (2812 fit, 938 tune, 938 validation). Model sparse có random_state 42. Các seed và split policy nằm trong báo cáo.

Mọi vectorizer và tỷ lệ NB chỉ fit trên phần học. Chọn ngưỡng trên tune; sau đó refit outer80 và chấm validation. Mọi cấu hình và top-3 ensemble được chốt bằng tune trước khi xem validation của suite. Validation đã được xem trong lịch sử phát triển; đây là kết quả thăm dò, không phải ước lượng độc lập hoàn toàn.

## 3. Notebook có output

Chạy submission.ipynb từ thư mục gốc sau hai suite. Notebook đọc model cục bộ trong private_runs, thực hiện lại suy luận validation và bootstrap 1000 lượt, kiểm tra điểm F1, viết báo cáo 5 lỗi riêng, rồi refit một model từ toàn bộ train.

Nếu dùng thư mục kết quả khác, đặt ABSA_RESULTS_DIR. Notebook công khai không hiển thị nội dung train; báo cáo private_error_analysis.md có ví dụ và nhận xét được tạo riêng khi chạy. Không đăng file đó lên repo công khai.

Notebook hiện đã chạy nhưng **chưa là bộ nộp hoàn chỉnh**, vì bước blind chưa chạy và chưa có submission.csv được chấm.

## 4. Refit và xuất file nộp bằng mô hình

TFIDF_lan14 là lựa chọn minh họa theo điểm tune cao nhất suite reference. Nó không phải model có validation cao nhất. Việc refit all-train không tạo ra điểm held-out mới cho model đó.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python refit_final.py --experiment private_runs/TFIDF_lan14
python predict_local.py --model private_runs/TFIDF_lan14/final_model.joblib --local-test /duong/dan/test_blind.csv --template submission_demo.csv --output submission.csv
python submission_io.py submission.csv submission_demo.csv
python score_absa.py --kiem-tra submission.csv /duong/dan/test_blind.csv
```

Các lệnh blind chỉ để người dùng chạy mô hình cục bộ khi có dữ liệu; không đưa văn bản blind vào LLM hoặc dịch vụ ngoài. Hoặc đặt ABSA_LOCAL_TEST trước khi chạy cell cuối notebook. Mặc định notebook không đọc blind.

CSV UTF-8 có đúng 732 dòng, 22 cột theo thứ tự file mẫu. Nhãn được ghi dưới dạng số nguyên 0/1. Không ghi xác suất, boolean hoặc 0.0. ID không trùng và khớp mẫu; lệnh score_absa đối chiếu thêm với blind thật. Khen/chê dương kéo nhắc lên 1. File mẫu toàn 0 không phải file kết quả nộp.

## 5. PhoBERT / ViSoBERT

```bash
python -m pip install -r requirements-neural.txt
python neural_train.py --config PhoBERT_lan1/config.json --smoke
python neural_train.py --config PhoBERT_lan1/config.json
python neural_train.py --config PhoBERT_lan2/config.json
python neural_train.py --config ViSoBERT_lan1/config.json
```

Các lệnh full training yêu cầu GPU trong runner này. Không dùng checkpoint của lượt smoke để nộp: smoke chỉ chạy vài mẫu và không lưu model kết quả. PhoBERT_lan1 là joint21; PhoBERT_lan2 là label attention; ViSoBERT_lan1 dùng văn bản không tách từ. Revision đã được cố định. Tài nguyên pretrained chỉ tải về máy; không gửi dữ liệu huấn luyện ra API sinh nhãn.

PhoBERT dùng PyVi để bám cách tiền xử lý của mã tham khảo; đây là lựa chọn khác khuyến nghị RDRSegmenter của VinAI, cần thử riêng để đo hiệu quả. Cửa sổ dài 256, stride 64, loss ở mức review sau max pooling cửa sổ, không gán nhãn toàn review cho từng đoạn. Checkpoint và ngưỡng chọn bằng tune; validation chỉ chấm sau khi chọn. Script deep này chỉ huấn luyện/đánh giá validation, chưa có bước production refit hoặc xuất blind cho neural.

Mã stacking tham khảo còn cần neural_evaluation.py, classical_source.py và các checkpoint/config/scores tương ứng. Không có các tệp đó thì chưa thể tái hiện F1 0.885296. Các encoder khác phạm vi fit/refit phải kiểm tra provenance; không trộn model đã fit tune vào nguồn tune dùng để học meta.

## 6. Bộ nộp theo đề

Sau khi có file được chấm, zip theo tên nhóm gồm submission.ipynb đã chạy với output, các .py cần thiết, requirements.txt, hướng dẫn, model đã huấn luyện/cấu hình môi trường, và đúng submission.csv có điểm công khai cao nhất. Không kèm train.csv hoặc test_blind.csv. Đối chiếu chạy lại file CSV bằng hash hoặc so sánh nội dung. Nếu dùng model neural cần kèm torch/transformers/CUDA, GPU/VRAM, RAM/CPU/OS và thời gian huấn luyện. Hiện chưa có dữ liệu thêm; các phép chuẩn hóa giữ nguyên số dòng train nên không tạo du_lieu_them.csv.
