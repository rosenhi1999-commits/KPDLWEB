# KPW2.3 — ba mô hình và xuất hồ sơ nộp bài

## Trạng thái đo thực tế

`colab_runner.py` chuẩn bị lượt học mới gồm cả PhoBERT, TF-IDF và ViSoBERT.
Chưa chạy full fine-tuning GPU cho phiên bản mới này; không gán điểm thử cache
cho mô hình chưa huấn luyện. Không bảo đảm ABSA-F1 0.96.

Một phép thử thực tế dùng **cache dự đoán encoder cũ** PhoBERT_lan25,
ViSoBERT_lan2 và TF-IDF hai dạng văn bản vừa học lại đạt ABSA-F1 validation
**0.8926458809**, trọng số **0.7 / 0.1 / 0.2**, CI95%
[0.8791632414, 0.9049270416]. Nó chưa vượt PhoBERT_lan25 (0.8936406022).
Đây là validation đã được xem khi phát triển, không phải điểm công khai.
Checkpoint encoder chuẩn hóa mới cần được huấn luyện và đo riêng.

## Ba mô hình bổ sung cho nhau

- PhoBERT: NFC, chuẩn hóa khoảng trắng, mở rộng từ viết tắt rồi tách từ PyVi.
- TF-IDF: word n-gram sau chuẩn hóa/tách từ + char n-gram câu gốc + char n-gram
  sau chuẩn hóa. Từ vựng và classifier chỉ học trên phần fit.
- ViSoBERT: NFC/khoảng trắng, giữ teencode và từ viết tắt, dùng tokenizer riêng.
- Giữ dấu, phủ định, emoji, dấu câu và số. Không tự khôi phục dấu toàn câu.
  Không thay toàn cục các từ nhiều nghĩa `vs`, `mk`, `p`, `kg`; `k` chỉ thay
  trong một số ngữ cảnh phủ định và không thay khi đứng sau số tiền/khối lượng.
- Mỗi nguồn có trọng số ít nhất 0.1 theo yêu cầu dùng đủ cả ba. Trọng số/ngưỡng
  chọn qua 5-fold threshold CV trên tune, không chọn bằng validation. Đây không
  phải nested encoder CV vì checkpoint encoder cũng đã được chọn bằng tune.
- Giữ GroupShuffleSplit seed 2026/2027 dựa trên văn bản gốc; hạt giống học 2026.

Mỗi C của TF-IDF, mỗi encoder và mỗi phép kết hợp có thư mục `PHUONGPHAP_lanN`.
Lượt hoàn tất cùng fingerprint dữ liệu/code/config được tái sử dụng.
Lượt neural bị ngắt phải chạy lại từ đầu; chưa có epoch-resume.

## Dùng checkpoint đã tải để có submission ngay

Trên Colab: restart session để bỏ audit hook train-only cũ, chọn GPU nếu có.
Cài thư viện văn bản/notebook mà không thay PyTorch CUDA có sẵn:

```python
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
    'transformers==4.51.3', 'tokenizers==0.21.4',
    'huggingface-hub==0.36.2', 'safetensors>=0.4.3',
    'sentencepiece==0.2.1', 'protobuf', 'pyvi==0.1.1',
    'iterative-stratification==0.1.9', 'nbformat', 'nbclient', 'ipykernel'])
```

Đặt ba script `export_colab.py`, `predict_bundle.py`, `colab_runner.py` cùng thư mục.
Tải lên ZIP `absa-selected-encoder.zip`, `train.csv`, `test_blind.csv`.
Không cần huấn luyện lại để dùng checkpoint có sẵn:

```bash
python export_colab.py --archive /content/absa-selected-encoder.zip \
  --train /content/train.csv --test /content/test_blind.csv \
  --team KPW2.3 --out /content/absa_deliverables
```

Test chỉ dùng inference, không fit, không chọn epoch/ngưỡng/trọng số, không tạo nhãn.
Giữ đúng tiền xử lý của checkpoint cũ; không áp bộ chuẩn hóa mới vào checkpoint đó.

## Chạy ba mô hình mới và tự xuất thư mục

```bash
python colab_runner.py --train /content/train.csv --out /content/absa_three_models \
  --export-test /content/test_blind.csv --export-out /content/absa_deliverables \
  --team KPW2.3
```

Mặc định batch_size=4, accumulation=4, tối đa 12 epoch, patience=3.
Nếu thiếu VRAM: thêm `--batch-size 2 --accumulation 8`.
Không có GPU thì phần huấn luyện từ chối chạy; replay checkpoint vẫn hỗ trợ CPU.
Đường dẫn test chỉ truyền cho bước xuất sau khi mọi lựa chọn mô hình đã cố định.

## Thư mục xuất sau mỗi lượt

```text
absa_deliverables/
  KPW2.3_lan1/
    submission.csv
    submission.ipynb               # đã thực thi, có output validation/inference
    requirements.txt               # phiên bản cụ thể + phiên bản Python
    HUONG_DAN.md
    models/                        # trọng số và cấu hình thật
    code/                          # inference, training, scorer và template
    training_provenance.json        # máy/thời gian huấn luyện gốc
    inference_hardware.json         # CPU/RAM/GPU/VRAM/OS/Python/CUDA hiện tại
    validation_metrics.json
    inference_report.json           # thời gian, SHA256 CSV/test/model
    execution_report.json
    PUBLIC_SCORE.json
    MANIFEST_SHA256.json
    KPW2.3.zip                     # dự thảo cho tới khi ghi điểm công khai
```

Không xuất dữ liệu train/test gốc. Không dùng thêm dữ liệu LLM/công cụ tạo nhãn,
nên không có `du_lieu_them.csv`. Notebook được nbclient chạy thực sự trong kernel
mới; không chèn output giả và không đóng ZIP nếu notebook thất bại.
Kiểm tra chính thức 732 dòng dữ liệu, 22 cột, đúng ID/nhãn nguyên 0/1:

```bash
python code/score_absa.py --kiem-tra submission.csv /content/test_blind.csv
```

Archive checkpoint GitHub cũ không ghi CPU model/RAM: các trường này để null và
ghi rõ thiếu. Không lấy máy inference thay làm máy huấn luyện. Lượt ba mô hình
mới ghi `training_hardware.json` cho từng nguồn; thời gian đo nằm trong báo cáo.
Phiên bản phụ thuộc thực tế được ghi khi xuất, không hứa tái hiện bit-for-bit qua
GPU/CPU khác nhau. Giữ nguyên CSV gốc được trang chấm cho điểm để nộp cuối kỳ.

## Chọn đúng lượt có điểm công khai cao nhất

Nộp CSV trên web bằng gmail đại diện và OTP của nhóm. Công cụ không đăng nhập
hay tự nộp bài. Sau khi có điểm, nhập **điểm thật** thay `0.90` ví dụ dưới đây:

```bash
python export_colab.py --out /content/absa_deliverables --team KPW2.3 \
  --run-folder /content/absa_deliverables/KPW2.3_lan1 --record-score 0.90
```

`CHON_NOP_GIANG_VIEN/KPW2.3.zip` và `submission.csv` tự cập nhật theo điểm công
khai cao nhất đã nhập cho cùng SHA256 test. Registry giữ đường dẫn lượt và hash
CSV/model. Điểm validation không tự được ghi thành điểm công khai.
Khi chưa có điểm công khai, không khẳng định lượt nào cao nhất.
Sau khi trang chấm đóng, nộp ZIP trong `CHON_NOP_GIANG_VIEN` cho giảng viên.

## Kiểm tra đã thực hiện

Đã nạp strict checkpoint compact thật, kiểm tra tiền xử lý/PyVi/chia cửa sổ/max
pooling khớp mã huấn luyện (sai khác xác suất 0 trên fixture train). Đã kiểm tra
định dạng 732 dòng qua scorer chính thức bằng fixture tạm, thứ tự ID, nhãn 0/1,
chọn public score cao nhất/hash và thực thi notebook bằng kernel thật.
Fixture chỉ phục vụ kiểm thử, bị xóa, không phải submission hay điểm thực nghiệm.
Chưa xuất submission trên test thật của người dùng trong môi trường này.
