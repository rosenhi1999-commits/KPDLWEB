# Phương pháp, nguồn và giới hạn đánh giá

**Cập nhật đợt mới:** đã đo 46 lượt; tốt nhất trên split 2026 là Stacking_lan1, ABSA-F1 0.799716. Đã thử RDR và dữ liệu tổng hợp. Xem [TRAIN_ONLY_REPORT.md](TRAIN_ONLY_REPORT.md); các con số 24 lượt và trạng thái chưa đo RDR bên dưới là lịch sử của đợt trước.

## Định dạng và điểm chính

Đã đọc submission_demo.csv (732 dòng, 22 cột) và score_absa.py của repo. File nộp có id và 21 nhãn 0/1. Score xếp hạng là macro F1 của 14 nhãn khen/chê; accuracy không phải mục tiêu. Báo cáo tham khảo của người dùng là F1 0.885296, CI95% [0.871441,0.898356]; accuracy 0.963 không có nghĩa đạt ABSA-F1 0.96. Báo cáo đó tái tính từ saved encoder outputs, không huấn luyện lại encoder.

## Các hướng đã đo

24 lượt gồm hai baseline ban đầu, 15 cấu hình sparse bổ sung, 5 cấu hình trên split tham khảo, và 2 ensemble. Có Logistic Regression, SVM, ComplementNB, NB-SVM, SGD modified-Huber, Ridge; TF-IDF từ/ký tự; thay đổi regularization; cân bằng lớp; chọn ngưỡng từng nhãn; chuẩn hóa viết tắt; tách từ PyVi; bỏ phiếu top-3 chọn bằng tune. Mỗi lượt có thư mục riêng và kết quả đo thực tế. Hai bảng comparison chia riêng theo split; không xếp chung như cùng validation.

Tốt nhất legacy là Ensemble_lan1 F1 0.782002, CI95% [0.766254,0.796525]. Tốt nhất reference là NBSVM_lan3 F1 0.757043, CI95% [0.740667,0.770977]. Chưa đạt 0.96. Chuẩn hóa kỹ hơn không bảo đảm tăng F1; có thể làm mất tín hiệu. Không thể thử hết mọi phương pháp/hyperparameter trong một không gian không giới hạn.

Các lượt có F1, P/R, accuracy, kappa, MCC, balanced accuracy và bootstrap 500 lần; notebook tái tính 1000 lần cho model reference được chọn bằng tune. Nhãn khó nhất của Ensemble_lan1 là tien_nghi_che, F1 0.579310; của NBSVM_lan3 là vi_tri_che, F1 0.551724. Năm ví dụ lỗi có nhận xét nằm trong báo cáo riêng tạo khi chạy notebook.

## Cơ sở tra cứu trên web

- [Wang & Manning, ACL 2012 — NB-SVM và bigrams](https://aclanthology.org/P12-2018/): cơ sở thử các đặc trưng từ đôi và tỷ lệ log-count NB kết hợp SVM. Đây là cách triển khai mở rộng cho 21 nhãn, không phải tái hiện nguyên benchmark của bài báo.
- [VinAI PhoBERT chính thức](https://github.com/VinAIResearch/PhoBERT): đề xuất RDRSegmenter/VnCoreNLP cho tiền xử lý phù hợp pretraining. Mã tham khảo dùng PyVi; cần coi RDR là một ablation mới, không khẳng định đã đo.
- [ViSoBERT, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.315/) và [model card](https://huggingface.co/uitnlp/visobert): encoder chuyên văn bản mạng xã hội tiếng Việt; là hướng thử cho viết tắt và ngôn ngữ không chuẩn, không bảo đảm điểm khách sạn.
- [Sun et al., NAACL 2019 — BERT với câu phụ trợ](https://aclanthology.org/N19-1035/): hướng aspect-pair/NLI. Chưa triển khai/đo lại hướng này trong đợt hiện tại.
- [scikit-learn — chọn ngưỡng](https://scikit-learn.org/1.8/modules/classification_threshold.html): chọn ngưỡng trên dữ liệu riêng với fit để hạn chế overfit. Đợt này dùng tune riêng, không tối ưu ngưỡng bằng outer validation.

## Đọc mã tham khảo của người dùng

Đã đồng bộ split SHA256 + GroupShuffleSplit 2026/2027 trong reference_splits. Thiết kế cửa sổ + review-level loss của PhoBERT được giữ khi xây neural_train.py. Mã mới dùng lệnh train-only và tách hoàn toàn việc xuất CSV cục bộ. Hai head joint/label-attention đã được chuẩn bị; chỉ trạng thái smoke.json mới xác nhận lượt kiểm tra nhỏ, không phải F1 học sâu.

Stacking dùng các điểm xác suất của encoder, meta logistic và OOF trên tune; validation được giữ ngoài fit meta. Tuy vậy encoder đã chọn checkpoint theo tune nên meta OOF không phải một hệ thống nested cross-fitting encoder hoàn toàn; cần ghi rõ khi diễn giải. Khi encoder refit outer80/all-train, phân phối score có thể đổi; meta/ngưỡng cũ không tự bảo đảm giữ nguyên hiệu quả. Báo cáo đã thừa nhận validation feedback từ các lượt phát triển, nên không thể trình bày điểm đó như test độc lập.

Chưa có neural_evaluation.py, classical_source.py, các head aspect_pair/label_attention nguyên bản và checkpoint thực tế của 5 encoder; không thể tái hiện ensemble tham khảo chỉ từ hai script được dán. Không tạo điểm giả, không dùng nhãn blind hoặc dịch vụ LLM để dự đoán blind. Nguồn web chỉ dùng nghiên cứu phương pháp, không tìm lại đánh giá gốc.
