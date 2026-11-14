# Tái hiện đợt train-only

Đợt hiện tại chỉ dùng train.csv, không cần hoặc mở blind. Xem TRAIN_ONLY_REPORT.md để biết phạm vi, chỉ số và giới hạn. Python đã dùng 3.12.14, torch 2.6.0+cpu, Linux, CPU quota 2, RAM cgroup 8 GiB. RDR dùng Java runtime OpenJDK 21.0.13. Phiên bản thư viện nằm trong requirements.txt và requirements-neural.txt.

## Môi trường CPU

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-neural.txt
```

Cần Java cho RDR. Các tài nguyên pretrained tải về máy; không gửi văn bản vào API sinh nhãn. Revision PhoBERT đã cố định, VnCoreNLP được kiểm tra SHA256 bằng setup_rdr.py.

## Tạo thư mục tái hiện mới

Đặt train.csv cục bộ ở gốc repo, sau đó:

```bash
python prepare_train_only_workspace.py
cd private_train_only_replay
```

Helper chỉ chép mã, requirements, train và tùy chọn file mẫu định dạng; không chép blind hay các thư mục kết quả công khai. Việc này tránh ghi đè các lượt đã công bố. Nếu thư mục đã tồn tại, dùng --destination với tên mới. Không chạy trực tiếp runner vào các thư mục kết quả đã có.

## Chạy các lượt theo thứ tự

Đặt OPENBLAS_NUM_THREADS=1 và OMP_NUM_THREADS=2 cho các lệnh CPU; sparse có thể dùng OMP_NUM_THREADS=1.

```bash
python TFIDF_lan1/train.py
python TFIDF_lan2/train.py
python run_experiments.py --suite legacy --results-root .
python run_experiments.py --suite reference --results-root .
python run_scoped.py
python extract_phobert_cpu.py --token-cache
python run_phobert_tfidf.py
python train_frozen_attention.py
python run_multitask_sparse.py
python train_only_stacking.py
python setup_rdr.py
python extract_phobert_cpu.py --segmenter rdr
python run_phobert_tfidf.py --feature-set rdr
python generate_train_augmentation.py
python run_augmentation.py
python build_train_only_notebook.py --execute
```

Các runner mới từ chối ghi đè thư mục lượt đã có. Tất cả nguồn stacking phải chạy xong trước stacking; token cache PyVi phải có trước attention. Không thay thứ tự/cache giữa PyVi và RDR. Scaler, PCA/SVD và vectorizer chỉ fit phần học; meta OOF chỉ dùng tune. Train-only audit guard chặn mở CSV có tên chứa blind.

Tùy máy, dự trù hàng chục phút cho toàn bộ chuỗi CPU. Thời gian đo của từng phần nằm trong metrics.json/history.csv và các metadata extraction; chưa có benchmark tổng chuỗi trên máy mới. Hai attention head chạy 24 epoch tối đa và refit theo epoch được chọn. File token cache khoảng 290 MiB ở dữ liệu hiện tại; thêm encoder cache và model checkpoint cần dung lượng đĩa vài GiB. Không coi thời gian tái tính meta trong notebook là thời gian chạy encoder.

## Đầu ra và cách kiểm tra

Mỗi PHUONGPHAP_lanN có config.json, metrics.json, model validation và dự đoán validation cục bộ. Đối chiếu ABSA-F1 của notebook với metrics; notebook assert chênh lệch dưới 1e-12 khi tái chạy mô hình scope và tái tính stacking từ nguồn đã lưu. Không có cell suy luận blind. Cell cuối ghi rõ không tạo submission.csv.

Các thư mục công khai chỉ có mã, cấu hình/chỉ số và notebook không chứa review. Model, token/vector cache, validation_prediction và phân tích lỗi chứa dữ liệu lớp phải giữ riêng. .gitignore không xóa dữ liệu đã track trước đó.

Du_lieu_them.csv có 640 dòng được tạo từ template mới, không từ blind; augmentation_provenance.json có công cụ, user prompt, đặc tả sinh, seed, hash. Notebook hiển thị provenance và bảng so sánh có/không augmentation. Nếu dùng model augmented khi nộp sau này phải kèm CSV đúng tên du_lieu_them.csv, provenance và generator. Model stacking tốt nhất hiện tại không dùng augmentation.

## Đối chiếu yêu cầu nộp bài

Đợt này chưa là bộ nộp hoàn chỉnh. Bộ nộp cuối vẫn cần notebook đã chạy, mã/thư viện/hướng dẫn, model production và đúng file CSV do model dự đoán: UTF-8, 732 dòng, id + 21 cột nhãn theo thứ tự mẫu, giá trị nguyên 0/1 và ràng buộc nhắc bao hàm khen/chê. Không dùng accuracy thay ABSA-F1, không gán nhãn bằng LLM cho blind, không kèm dữ liệu gốc trong zip. Hiện không chạy bước CSV này vì yêu cầu chỉ dùng train.
