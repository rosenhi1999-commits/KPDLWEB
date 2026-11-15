# KPW2.3 — chỉ dùng train.csv

Mọi lượt học, chọn trọng số/ngưỡng và đánh giá đều dùng các phần tách từ train.csv.
Mỗi lượt tạo thư mục riêng: PHUONGPHAP_lanN.

- PhoBERT: chuẩn hóa viết tắt rồi tách từ PyVi.
- TF-IDF: từ đã chuẩn hóa, ký tự câu gốc và ký tự câu chuẩn hóa.
- ViSoBERT: giữ teencode để bổ sung tín hiệu cho hai mô hình còn lại.
- Giữ dấu, phủ định, emoji, số và dấu câu. Từ nhiều nghĩa chỉ sửa theo ngữ cảnh.
- Dùng đủ ba nguồn, mỗi nguồn tối thiểu 10%. Chọn trọng số/ngưỡng qua 5-fold
  trên tune; chỉ sau đó đo validation. Seed 2026, chia nhóm theo văn bản gốc.
- Checkpoint encoder cũng chọn trên tune; CV ngưỡng không phải nested encoder CV.
- Validation đã được xem qua các lượt phát triển, không phải phép đo độc lập mới.

## Xuất gói nội bộ từ checkpoint đã có

Đặt export_colab.py, colab_runner.py, predict_bundle.py, train_only_policy.py cùng chỗ.
Cài transformers==4.51.3, tokenizers==0.21.4, huggingface-hub==0.36.2,
sentencepiece==0.2.1, pyvi==0.1.1, protobuf, safetensors, iterative-stratification,
nbformat, nbclient, ipykernel. Giữ PyTorch CUDA có sẵn trên Colab.

Lệnh chạy:

    python export_colab.py --archive /content/absa-selected-encoder.zip \
      --train /content/train.csv --team KPW2.3 --out /content/absa_deliverables

Với ensemble đã học ở notebook trước:

    python export_colab.py --ensemble-dir /content/absa_runs/Ensemble_lan1 \
      --legacy-code-dir /content/absa_code --train /content/train.csv \
      --team KPW2.3 --out /content/absa_deliverables

Phải giữ đúng hàm chuẩn hóa đi kèm checkpoint; không áp quy tắc mới vào model cũ.

## Huấn luyện ba mô hình mới và tự xuất

    python colab_runner.py --train /content/train.csv --out /content/absa_three_models \
      --team KPW2.3 --export-out /content/absa_deliverables

Tạo tối đa 12 epoch, patience 3, batch 4, accumulation 4. Nếu thiếu VRAM dùng
--batch-size 2 --accumulation 8. Lượt hoàn tất cùng dữ liệu/code/config được dùng lại.
Lượt neural bị ngắt phải chạy lại từ đầu. Thêm --no-export nếu chỉ cần huấn luyện.

## Thư mục sau mỗi lượt

    KPW2.3_lanN/
      submission.ipynb             (đã thực thi, có output validation)
      validation_predictions.csv  (nhãn 0/1, ID của validation)
      validation_metrics.json
      requirements.txt            (thư viện và phiên bản Python)
      HUONG_DAN.md
      models/
      code/
      training_provenance.json
      replay_hardware.json
      execution_report.json
      MANIFEST_SHA256.json
      KPW2.3.zip

Notebook được thực thi trong kernel mới, không chèn output giả. Chỉ đóng ZIP
khi notebook chạy thành công. Không đóng gói train gốc. Dự đoán validation
không được đổi tên/ID để làm bài nộp chính thức. Đây là gói đánh giá nội bộ,
chưa phải hồ sơ nộp hoàn chỉnh.

CAO_NHAT_VALIDATION/KPW2.3.zip giữ gói có ABSA-F1 validation quan sát cao nhất
cho cùng dữ liệu. LUA_CHON.json ghi rõ cơ sở lựa chọn và hash file dự đoán.
Không gọi chỉ số validation là điểm chấm chính thức.

Lượt mới ghi CPU/RAM/GPU/VRAM/OS/Python/CUDA, thư viện và thời gian. Checkpoint
cũ có trường máy huấn luyện chưa ghi nhận thì để null; tách khỏi máy đánh giá.
Không thêm dữ liệu tạo sinh. Chưa có kết quả full GPU cho phiên bản mới;
không bảo đảm 0.96. Phép thử cache encoder cũ + TF-IDF mới đạt 0.892646,
không phải kết quả huấn luyện encoder chuẩn hóa mới.

## Chọn lại trọng số/ngưỡng mà không huấn luyện lại

    python refine_ensemble.py --train /content/train.csv \
      --sources THU_MUC_PHOBERT THU_MUC_TFIDF THU_MUC_VISOBERT \
      --out /content/absa_runs --legacy-code-dir /content/absa_code

Thử 11 bộ trọng số với 2 mức ổn định ngưỡng, giữ đủ ba mô hình. Ngưỡng được
học trên 4 phần và chấm trên phần tune còn lại, lặp đủ 5 phần. Sau khi chốt
lựa chọn, mới đọc xác suất validation. Kết quả in cả ba mô hình đơn và ensemble.
Sau đó tự xuất notebook có output và gói nội bộ KPW2.3.zip.
Nếu chỉ có cache xác suất, dùng --no-export; cache không thay cho trọng số thật.
Đây là phép thử tối ưu bước kết hợp, không bảo đảm tăng điểm validation.

Xuất gói của lượt hiện tại trước khi thử tinh chỉnh. Sau đó xuất lượt mới;
CAO_NHAT_VALIDATION sẽ giữ gói có điểm nội bộ cao hơn trong các gói đã xuất.
