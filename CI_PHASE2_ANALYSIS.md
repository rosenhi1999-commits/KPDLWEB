# Vòng 3: tiến từ mô hình đơn 0.890595

Mô hình tốt nhất đã đo trên validation nhóm 2026 là PhoBERT_lan20, ABSA-F1 **0.8905953814**. F1 khen 0.933028, F1 chê 0.848163. Các nhãn yếu: vi_tri_che 0.775862, gia_tri_che 0.796460, tien_nghi_che 0.816901, tien_nghi_khen 0.844444. Chưa đạt 0.96; không coi đây là điểm hệ thống hay test độc lập.

**Kiểm tra bổ sung:** Calibration_lan2 chọn shrink25 bằng tune OOF, đạt 0.891940607 từ cache xác suất FP32 của lan20. Đã tải checkpoint đóng gói và chạy encoder lại thật trên CPU: ngưỡng cũ 0.891124484, ngưỡng mới **0.891823139**, tăng 0.000698655 trong cùng lượt replay. Artifact trước đó ghi 0.891103328 ở ngưỡng cũ; có chênh lệch số nhỏ giữa môi trường/đường tính, nên trường original_threshold_replay_matches được giữ false trong checkpoint_replay.json. Không tuyên bố tái hiện bit-for-bit hoặc chọn ngưỡng bằng validation. Các số từ cache và từ checkpoint float16 được báo riêng.

Vòng 3 đã khởi chạy: https://github.com/rosenhi1999-commits/KPDLWEB/actions/runs/43701826710 . Các kết quả lượt mới chỉ được tính khi huấn luyện xong.

## Bằng chứng từ vòng trước

Lan20: LR 3e-5, mention loss 0.5, head LR 5x, dropout head 0.2, 12 epoch tối đa. Checkpoint chọn bằng tune ở epoch 11, tune F1 0.891547. Epoch 12 không cải thiện; không tiếp tục tăng epoch một cách mặc định.

Stacking_lan3 đạt 0.872659, thấp hơn mô hình đơn. Các nguồn có chất lượng và phân phối khác nhau, trong đó shared-attention lan21 chỉ đạt 0.508194; việc dùng nhiều feature/C/ngưỡng riêng trên tập tune nhỏ có thể không khái quát tốt. Không khẳng định một nguyên nhân duy nhất khi chưa có ablation đầy đủ.

Đã thử blending trên đúng 13 cache nguồn cũ, chỉ chọn nguồn/trọng số/ngưỡng bằng tune. Chọn ứng viên có OOF cao nhất cho ra Blending_lan1 0.880335. Quy tắc one-standard-error ưu tiên ít nguồn khi khác biệt OOF nhỏ chọn anchor PhoBERT_lan20; Blending_lan2 giữ nguyên 0.890595. Những kết quả không tăng được ghi lại, không trình bày thành cải thiện.

## Năm lượt mới, mỗi lượt một thư mục

| Lượt | Một hướng thay đổi từ lan20 |
|---|---|
| PhoBERT_lan23 | Training seed 3407; giữ split seed 2026 |
| PhoBERT_lan24 | Encoder LR 4e-5 |
| PhoBERT_lan25 | Head dropout 0.1 |
| PhoBERT_lan26 | R-Drop alpha 0.5, hai forward cùng review, KL Bernoulli đối xứng cho 14 nhãn khen/chê |
| PhoBERT_lan27 | Attention với 7 head độc lập; mỗi head nhận vector pooled của khía cạnh và CLS |

R-Drop giữ nguyên kiến trúc suy luận; chỉ dùng thêm consistency loss khi học. Nó có thể tăng chi phí gần gấp đôi; time budget ở ranh giới epoch và stop reason vẫn được ghi thật. Khi alpha=0, loss và đường một forward giữ nguyên baseline. Không nhân bản nhãn theo window.

Head attention mới khắc phục giới hạn thiết kế tiềm tàng của bộ phân loại 3 nhãn dùng chung: khi attention các khía cạnh gần nhau, đầu ra của chúng có thể gần giống nhau. Head độc lập vẫn có thể học khác nhau từ CLS, ngay cả khi attention ban đầu chưa chuyên biệt. Đây là giả thuyết đã được kiểm tra tính chạy được, chưa có điểm fine-tuning mới để khẳng định cải thiện.

## Chọn ensemble và lưu checkpoint

Vòng 3 mang theo 13 nguồn trước từ hai workflow, thêm các lượt mới hoàn tất. Stacking phức tạp và blending ít nguồn được báo cáo riêng. Blend thử một anchor và vài trọng số toàn cục, dùng 5-fold tune OOF; one-standard-error ưu tiên mô hình đơn/ít nguồn khi bằng chứng cải thiện yếu. Việc bootstrap ở đây chỉ phục vụ lựa chọn trên tune, không biến validation đã xem nhiều lần thành test độc lập.

Đóng gói encoder tiếp tục chọn theo điểm tune tại checkpoint đã chọn, không dùng validation để chọn checkpoint. Nếu nguồn thắng thuộc workflow cũ, artifact và cache được tải bằng run ID gốc của nguồn; không tìm nhầm checkpoint ở run mới. Trọng số nén được chấm lại bằng chính chế độ inference ghi trong config.

Chỉ dùng train.csv. Audit guard, kiểm tra ID của tune/validation và hash dữ liệu tiếp tục được giữ. Các lượt mới đang chuẩn bị/chạy, chưa có F1 mới. Workflow tự công bố kết quả đo thật sau khi hoàn tất; không có bảo đảm đạt 0.96.
