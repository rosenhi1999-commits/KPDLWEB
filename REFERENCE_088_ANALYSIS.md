# Phân tích bài mẫu 0.88 và thiết kế tối ưu

Bạn xác nhận bài mẫu được hệ thống chấm khoảng **0.88**. Tệp dán trước đó ghi **validation ABSA-F1 0.885296**, CI95% [0.871441,0.898356]. Đây là hai nguồn điểm khác nhau; không coi validation là điểm hệ thống. Chưa có checkpoint và đầy đủ các module của bài mẫu để tái hiện các số đó.

## 1. Cần giữ đúng phần mạnh của bài mẫu

Báo cáo mẫu kết hợp **5 encoder có giám sát**: 2 PhoBERT joint21, 1 PhoBERT aspect-pair, 1 ViSoBERT joint21 và 1 PhoBERT label-attention. Các nguồn có phạm vi fit/refit khác nhau. Meta logistic học từ xác suất nguồn, kiểm tra provenance của checkpoint và lựa chọn C/feature mode bằng OOF trên tune.

Đây là khác biệt lớn trong thiết kế so với các lượt 0.7997 dùng encoder pretrained đóng băng trước đây. Thêm TF-IDF hoặc đổi tách từ trên encoder đóng băng không tái hiện được ensemble đã fine-tune. Việc tối ưu tiếp phải bắt đầu từ baseline fine-tuning và các đầu ra ngoài tập học của đúng nguồn.

Giữ cố định split nhóm: 2812 fit, 938 tune, 938 validation; SHA256 clean(text).lower(), seed split 2026/2027. Đã tách training_seed khỏi split_seed trong runner mới để thử seed huấn luyện sau này mà không đổi validation. Giữ PyVi, chữ hoa/thường, NFC, 256 token/stride 64 và review-level loss khi tái hiện baseline. Không thay đồng thời tokenizer, loss và cấu trúc model rồi gán thay đổi điểm cho learning rate.

## 2. Điểm nghẽn nhìn thấy trong báo cáo mẫu

F1 khen khoảng **0.932**, F1 chê khoảng **0.839**. Các nhãn cần ưu tiên phân tích: tien_nghi_che 0.785, gia_tri_che 0.808, vi_tri_che 0.825, nhan_vien_che 0.827, tien_nghi_khen 0.817. Đây là số được dán từ báo cáo, không phải số vừa tái đo.

Trên split train đã kiểm tra, tune chỉ có **43 mẫu gia_tri_che**, **47 mẫu vi_tri_che**. Một vài dự đoán đổi trạng thái có thể làm ngưỡng F1 tối ưu biến động rõ. Vì vậy nên kiểm tra ngưỡng bằng CV và thử co ngưỡng của nhãn ít mẫu về ngưỡng chung cùng cực tính, thay vì chỉ tăng độ dày lưới ngưỡng.

Trong cache PyVi của dữ liệu hiện tại chỉ **29/4688 review cần nhiều cửa sổ**, tạo 37 cửa sổ thêm. Tối ưu stride có thể hữu ích cho một số lỗi nhưng không phải ưu tiên đầu tiên cho toàn bộ bài.

Mục tiêu 0.96 cần cải thiện rộng: nếu chỉ hai nhãn tiện nghi trở nên hoàn hảo, macro F1 từ mức 0.8853 mới lên khoảng **0.9137** (tính từ các F1 đã làm tròn trong báo cáo). Ngay cả khi F1 khen đạt 1, F1 chê vẫn phải đạt ít nhất 0.92 để macro đạt 0.96. Đây là phép tính mục tiêu, không phải dự báo điểm sẽ đạt.

## 3. Các thông số gốc đọc được và ablation đã chuẩn bị

Thông số dưới đây là mặc định trong mã được dán, **chưa chứng minh đó chính là cấu hình của toàn bộ ensemble 0.88**: encoder LR 2e-5, head LR gấp 5, AdamW weight decay 0.01, warmup 0.1, head dropout 0.2, 8 epoch/patience 3, positive weight sqrt(negative/positive) clip [1,3], effective batch 16 review.

| Thư mục mới | Thay đổi so với baseline joint encoder | Mục đích kiểm tra |
|---|---|---|
| PhoBERT_lan3 | Baseline theo mặc định mã gốc | Mốc fine-tuning một nguồn |
| PhoBERT_lan4 | Head LR 2x thay 5x | Hạn chế head học quá nhanh |
| PhoBERT_lan5 | Encoder LR 1e-5 | Kiểm tra overfit/độ ổn định |
| PhoBERT_lan6 | Encoder LR 3e-5 | Kiểm tra encoder chưa thích nghi đủ |
| PhoBERT_lan7 | Không positive reweighting | Đo đánh đổi precision/recall với train có lệch phân bố |
| PhoBERT_lan8 | Loss nhãn nhắc nhân 0.5 | Giảm áp lực từ 7 nhãn không tính điểm chính |
| PhoBERT_lan9 | Chọn epoch bằng F1 ở ngưỡng cố định 0.5, rồi mới hiệu chỉnh | Giảm việc chọn checkpoint dựa trên ngưỡng dao động |
| PhoBERT_lan10 | Layer LR decay 0.9, không decay bias/LayerNorm | Một ablation chính sách optimizer |

Tất cả đều **prepared_not_trained**, chưa có F1 mới. PhoBERT_lan3 chỉ là thành phần joint encoder, không phải bản tái hiện đầy đủ ensemble 5 nguồn. Chạy baseline trước, sau đó ưu tiên lan4/lan9/lan7; không cần chạy cả lưới trước khi xem đường học trên tune. Chỉ chấm outer validation sau khi đã chốt lựa chọn cho mỗi hướng.

## 4. Sửa quy trình hiệu chỉnh và stacking

- Mỗi nguồn phải xuất tune_scores.npz và validation_scores.npz với xác suất và ID. Mã mới lưu cả hai, checkpoint model.pt và run_info.json. ID của tune và validation khác nhau dù cùng 938 dòng.
- calibrate_reference.py bắt buộc tập ID đúng tune, từ chối validation. So sánh fixed 0.5, lưới mã gốc, ngưỡng PR chính xác, và shrink25/shrink75 qua 5-fold trên tune. CV này dùng chọn bộ hiệu chỉnh; không phải điểm hệ thống hoặc outer validation. Encoder đã chọn epoch trên tune thì phải ghi rõ meta/calibration chưa phải nested encoder CV hoàn toàn.
- Mở rộng C của meta từ [0.01,0.1,1] thành [0.001,0.003,0.01,0.03,0.1,0.3,1], vẫn chọn bằng OOF. Ưu tiên nguồn khác kiến trúc và kiểm tra ablation bỏ từng nguồn trên tune OOF; không thêm encoder tương tự chỉ vì số lượng nhiều.
- Nếu đổi checkpoint encoder, phải fit lại meta trên score ngoài tập học tương ứng. Không ghép meta cũ với checkpoint mới tùy ý; model_identity trong mã gốc là kiểm tra cần giữ.
- Sau refit outer80/all-train, score có thể đổi phân bố. Không dùng score in-sample của tune đã được encoder học để hiệu chỉnh rồi trình bày là out-of-fit. Hướng chắc hơn để thử sau là cross-fitting encoder trong outer80; cần thêm nhiều lượt fine-tune. Validation 938 mẫu vẫn phải ở ngoài toàn bộ quá trình.

**Lưu ý mã gốc:** train mặc định có production refit và có thể chạy blind. Để tạo nguồn train-only cho stacking phải đặt --no-refit và --skip-test. stacked_absa.fit cũng từ chối nguồn production_refit=True. Runner mới neural_train.py không có nhánh đọc blind và đã bật audit guard.

## 5. Cách chạy bộ tối ưu mới

Chạy tại gốc repo, nơi có PhoBERT_lan3..10/config.json và train.csv cục bộ. Dùng môi trường PyTorch/Transformers trong requirements-neural.txt, có GPU cho full fine-tuning.

```bash
python verify_reference_optimization.py
python run_reference_trials.py --dry-run
python run_reference_trials.py --runs PhoBERT_lan3
python run_reference_trials.py --runs PhoBERT_lan4 PhoBERT_lan9 PhoBERT_lan7
python calibrate_reference.py --train train.csv --scores PhoBERT_lan3/tune_scores.npz --out Calibration_lan1 --stage encoder --validation-scores PhoBERT_lan3/validation_scores.npz
```

Các lệnh full fine-tuning sẽ không chạy nếu chưa có GPU. Khi có score của meta, dùng --stage meta để đối chiếu lưới 17 điểm đúng mã meta gốc. --scores bắt buộc là tune; validation_scores chỉ dùng qua --validation-scores sau khi đã chốt hiệu chỉnh để đo delta so với lưới gốc. Công cụ không chọn ngưỡng bằng validation. Chưa có bản refit production hoặc file nộp mới trong đợt phân tích này.

## 6. Đã kiểm tra và chưa kiểm chứng

Đã kiểm tra loss mặc định và gradient tương đương BCE của mã gốc, positive weights mặc định tương đương, nhóm LR/no-decay đúng, sampler không bỏ/trùng review, và kiểm tra ID chặn dùng validation làm tune. Đã chạy một cập nhật trọng số thật từ PhoBERT pretrained trên CPU cho nhánh mention loss 0.5; đó là smoke test, không phải điểm F1.

Chưa có điểm cải thiện so với 0.88 được kiểm chứng. Máy hiện tại không GPU; chưa huấn luyện đầy đủ 8 cấu hình mới và chưa tái hiện ensemble bài mẫu. reference_search_plan.json đánh dấu rõ trạng thái này. Cần chạy các lượt fine-tuning và đánh giá đúng split trước khi kết luận thông số nào nâng kết quả cuối cùng.
