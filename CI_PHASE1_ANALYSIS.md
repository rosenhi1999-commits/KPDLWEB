# Kết quả thật vòng 1 và lựa chọn vòng 2

GitHub Actions run 43590132635 đã kết thúc **success**: 8/8 lượt full fine-tuning, collector/stacking và đóng gói model đều thành công. Tất cả chỉ dùng train.csv.

| Lượt | Thay đổi | Validation ABSA-F1 |
|---|---|---:|
| PhoBERT_lan11 | Baseline LR 2e-5 | 0.808473 |
| PhoBERT_lan12 | Head LR 2x | 0.798199 |
| PhoBERT_lan13 | Encoder LR 1e-5 | 0.761572 |
| PhoBERT_lan14 | Encoder LR 3e-5 | **0.854359** |
| PhoBERT_lan15 | Không positive reweighting | 0.781356 |
| PhoBERT_lan16 | Loss nhắc 0.5 | **0.843049** |
| PhoBERT_lan17 | Chọn epoch ở ngưỡng cố định | 0.815299 |
| PhoBERT_lan18 | Layer LR decay/no bias decay | 0.784918 |
| Stacking_lan2 | 8 nguồn fine-tuned | **0.860469** |

Stacking_lan2 có CI95% [0.845449,0.874968], F1 khen 0.918703 và F1 chê 0.802235. Nhãn yếu: vi_tri_che 0.690909, gia_tri_che 0.708333, tien_nghi_che 0.750000, tien_nghi_khen 0.783383. Điểm tăng so với stacking đóng băng 0.799716, nhưng chưa đạt 0.96. Điểm hệ thống 0.88 người dùng báo không phải cùng phép đo với validation này.

Phiên bản checkpoint đóng gói của PhoBERT_lan14 đã được load và chấm lại: ABSA-F1 vẫn **0.8543592768671614**. Trọng số lưu float16, tính toán float32, artifact khoảng 250 MB. Artifact được GitHub giữ 7 ngày. Các checkpoint FP32 trong cache không phải kho lưu vĩnh viễn.

## Vì sao không chỉ tăng epoch

PhoBERT_lan14 đạt tune F1 0.863864 ở epoch 6; epoch 7/8 lần lượt 0.861616/0.863558. Vì vậy chưa có bằng chứng rằng chạy lâu hơn với cùng lịch học sẽ tăng điểm. PhoBERT_lan16 còn tăng nhẹ đến epoch 8. Hai thay đổi tốt nhất là LR 3e-5 và mention loss 0.5; kết hợp chúng là giả thuyết cần thử, không phải cộng trực tiếp hai mức tăng.

Đã chạy thêm hiệu chỉnh ngưỡng thật cho PhoBERT_lan14 bằng tune OOF: shrink25 được chọn trên tune, nhưng validation chỉ đạt 0.854223, chênh -0.000136 so với lưới gốc. Calibration_lan1 ghi rõ kết quả. Không chọn lại phương pháp ngưỡng bằng validation để báo một mức tăng giả.

## Vòng 2 được chốt trước khi chấm

| Thư mục mới | Hướng thử |
|---|---|
| PhoBERT_lan19 | LR 3e-5, lịch tối đa 12 epoch |
| PhoBERT_lan20 | LR 3e-5 + mention loss 0.5, tối đa 12 epoch |
| PhoBERT_lan21 | PhoBERT label attention + cấu hình lan20 |
| ViSoBERT_lan2 | Encoder mạng xã hội, raw text, LR 2e-5, mention loss 0.5 |
| PhoBERT_lan22 | Refit recipe thắng trên 3750 mẫu, cố định 6 epoch và ngưỡng từ selection |

Trong lan22, model refit không được chấm tune vì tune đã nằm trong phần học. Cache tune dùng cho meta được sao chép từ model selection cũ, nơi các mẫu đó ngoài phần fit; cache validation được tính mới bằng model refit. Provenance và khả năng đổi phân bố xác suất sau refit được ghi rõ. Đây là lựa chọn có thể kiểm tra trên outer validation, không bảo đảm giữ nguyên calibration.

Workflow mang tiếp 8 nguồn cũ qua artifact cùng run ID, kết hợp với các nguồn mới đã hoàn thành để học stacking kế tiếp. Không ghi đè Stacking_lan2; collector cấp số thư mục mới. Không dùng blind. Validation được giữ ngoài train/refit nhưng đã được xem qua các lượt phát triển, nên toàn bộ kết quả vẫn là nghiên cứu validation.
