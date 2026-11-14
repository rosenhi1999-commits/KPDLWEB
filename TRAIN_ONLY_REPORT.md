# Báo cáo chỉ dùng train.csv — chưa đạt ABSA-F1 0.96

Tổng cộng **46 lượt đã đo**, gồm 18 lượt legacy split 42 và 28 lượt trên cùng split nhóm 2026/2027 của mã tham khảo. Đợt mới thêm 22 lượt. Không đọc hay dự đoán blind. Các runner mới bật audit guard chặn mở CSV có tên blind; các lượt trước khi thêm guard cũng chỉ nạp train.csv bằng đường dẫn cố định.

## Kết quả chính trên cùng validation 938 mẫu, split 2026

| Hướng | Lượt tốt nhất quan sát | ABSA-F1 |
|---|---|---:|
| Sparse trước đợt này | NBSVM_lan3 | 0.757043 |
| TF-IDF theo mệnh đề | ScopedTFIDF_lan4 | 0.787288 |
| PhoBERT vector đóng băng + TF-IDF | PhoBERT_TFIDF_lan2 | 0.769551 |
| Attention token PhoBERT + đặc trưng TF-IDF | PhoBERT_Attention_lan2 | 0.777668 |
| Chia sẻ hệ số cảm xúc giữa khía cạnh | MultiTaskTFIDF_lan3 | 0.761218 |
| RDRSegmenter + PhoBERT đóng băng + TF-IDF | PhoBERT_RDR_lan3 | 0.766883 |
| Thêm 640 mẫu tổng hợp | AugmentScoped_lan1 | 0.792965 |
| Meta stacking 6 nguồn | Stacking_lan1 | **0.799716** |

Đây là bảng điểm tốt nhất quan sát theo hướng, không phải chọn mô hình bằng validation rồi coi đó là test độc lập. Mỗi runner chọn ngưỡng/epoch/hyperparameter bằng tune trước khi chấm outer validation; việc phát triển nhiều hướng đã sử dụng phản hồi validation nên toàn bộ báo cáo mang tính thăm dò.

Stacking_lan1: CI95% **[0.782321,0.813990]**; F1 khen 0.879071; F1 chê 0.720361; precision 0.829417; recall 0.776929; accuracy 0.941288. Nhãn yếu nhất vi_tri_che: F1 0.574257. Accuracy không phải điểm mục tiêu. Chưa có lượt nào đạt ABSA-F1 0.9 hoặc 0.96.

So với NBSVM_lan3 trên cùng 938 mẫu, tăng F1 0.042673; CI bootstrap ghép cặp 1000 lượt của mức tăng [0.028764,0.057501]. CI này không xóa được thiên lệch do đã xem validation nhiều lần. Xem train_only_improvement.json, train_only_ablations.json và all_experiments.csv.

## Tiền xử lý và mô hình đã thử thật

- Chuẩn hóa Unicode NFC, khoảng trắng; giữ dấu, phủ định và placeholder. TF-IDF có biến thể chữ thường/viết tắt/tách từ; encoder giữ nguyên hoa/thường và dùng cửa sổ 256, stride 64.
- Scope theo định nghĩa 7 khía cạnh chọn các mệnh đề liên quan. Từ điển chỉ chọn đầu vào; nhãn do classifier học từ train, không có quy tắc gán nhãn blind.
- PhoBERT-base-v2 tải revision cố định, lượng tử hóa int8 Linear để trích xuất CPU. Vector CLS, masked mean và mean 4 lớp cuối; max giữa các cửa sổ. PCA/SVD/scaler của classifier chỉ fit ở phần học.
- Hai attention head được huấn luyện thật từ token PhoBERT đã lưu. Epoch và ngưỡng chọn bằng tune, sau đó refit head trên outer80. Encoder pretrained **đóng băng**, không phải fine-tune toàn bộ PhoBERT.
- Multi-task sparse chia sẻ hệ số từ vựng khen/chê, thêm residual riêng từng khía cạnh. Meta stacking dùng 6 nguồn, OOF 5-fold trên tune, chọn C và own-aspect/all-aspect bằng meta OOF. Neural head đã chọn epoch trên tune nên đây không phải nested encoder CV hoàn toàn. Outer validation không được fit vào nguồn hoặc meta.

## Hai ablation quan trọng

640 mẫu tổng hợp có nhãn từ template do ChatGPT/Codex soạn, Python seed 2026 kết hợp, không lấy review gốc hay văn bản blind. Có mẫu lời chê nhẹ, phủ định và vừa khen vừa chê. Các mẫu này chỉ thêm vào phần học; tune và validation giữ nguyên.

- NB-SVM theo mệnh đề: 0.785876 → 0.792965, tăng 0.007089; CI95% mức tăng [-0.003369,0.018483], chưa đủ bằng chứng cải thiện chắc chắn.
- Logistic theo mệnh đề: 0.787288 → 0.779123, giảm 0.008166.
- PyVi → RDR ở cùng cấu hình PhoBERT+TF-IDF logistic: 0.769551 → 0.763264; CI mức chênh lệch chứa 0. Không thể kết luận RDR sẽ kém khi fine-tune encoder; đây chỉ là ablation encoder đóng băng.

Do đó, tiền xử lý hoặc thêm dữ liệu không tự bảo đảm tăng điểm. Stacking tốt nhất hiện tại **không dùng dữ liệu tổng hợp**. Nếu chọn model AugmentScoped để nộp, bắt buộc kèm du_lieu_them.csv và augmentation_provenance.json, đồng thời giữ thông tin công cụ/prompt trong notebook. CSV được tái tạo bằng generate_train_augmentation.py, SHA256 10d53ce8c262710a98a7b5b337b88fcf509c48ddbfb3c1cc358d510af9b7c3b4.

## Phần chưa thực hiện

Chưa fine-tune toàn bộ PhoBERT/ViSoBERT, chưa có checkpoint của ensemble 0.885296 người dùng tham khảo và không tái hiện điểm đó. Máy hiện tại không GPU, CPU quota 2 lõi dù hệ thống báo 3 CPU logic, giới hạn RAM cgroup 8 GiB. Các phép chạy neural mới dùng 2 thread. Muốn đánh giá hướng full fine-tuning cần thêm thời gian/tài nguyên; không có bằng chứng cho rằng hướng đó chắc chắn đạt 0.96.

Notebook submission.ipynb đã chạy, hiển thị đủ chỉ số từ scorer chính thức và tái tính stacking từ đầu ra nguồn đã lưu. Cell này không huấn luyện lại encoder. Năm ví dụ sai cùng ghi chú được ghi riêng từ train; không đăng văn bản lớp lên repo công khai. Đây là nghiên cứu validation, chưa là bộ nộp hoàn chỉnh, chưa tạo submission.csv và chưa có model production fit toàn bộ train cho stacking.

## Nguồn phương pháp

[PhoBERT chính thức](https://github.com/VinAIResearch/PhoBERT) khuyến nghị RDRSegmenter; đợt này đã đo cả PyVi và RDR với encoder đóng băng. [PyTorch về dynamic quantization](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/) là cơ sở chạy encoder trên CPU. [Wang & Manning 2012](https://aclanthology.org/P12-2018/) là cơ sở hướng NB-SVM. Cách chọn ngưỡng riêng phần học theo [tài liệu scikit-learn](https://scikit-learn.org/1.8/modules/classification_threshold.html). Không tải dữ liệu ABSA bên ngoài hoặc tìm lại review gốc để khớp nhãn.
