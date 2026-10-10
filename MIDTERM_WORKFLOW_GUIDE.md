# HƯỚNG DẪN QUY TRÌNH TỔNG QUÁT BÀI TẬP GIỮA KỲ – ADVANCED DEEP LEARNING

> Mục đích: một "khung sườn" tái sử dụng được cho **bất kỳ bài toán/dữ liệu nào** (âm thanh, ảnh, chuỗi…), rút ra từ dự án KWS điều khiển robot (`Data_1400`, M1→M5).
> Ràng buộc đề bài: **tự xây dựng mô hình (from scratch, không dùng pretrained)** và **vận dụng kiến thức đã học trong môn**.

---

## 0. TƯ DUY CỐT LÕI (đọc trước khi làm)

1. **Câu chuyện khoa học > con số đẹp.** Bài giữa kỳ được chấm ở chỗ bạn *giải thích được vì sao* mỗi thành phần có mặt, không phải ở việc đạt 100%.
2. **Mỗi bước nâng cấp phải trả lời một câu hỏi**: "Thêm X để giải quyết vấn đề gì của mô hình trước?" (xem mục 4).
3. **Đo lường như một kỹ sư**: ngoài accuracy phải có Params, FLOPs, độ trễ, kích thước file (trade-off).
4. **Dữ liệu sạch và không rò rỉ** quan trọng hơn mô hình phức tạp (xem mục 2).
5. **Giữa kỳ chưa cần real-time** – chỉ cần mô hình hóa, huấn luyện, đánh giá khoa học. Real-time để dành cuối kỳ.
6. **Trung thực với kết quả**: nếu các mô hình cùng đạt trần (ceiling), hãy phân tích bằng loss, margin, Grad-CAM, thay vì phóng đại.

---

## 1. SƠ ĐỒ QUY TRÌNH TỔNG THỂ (8 GIAI ĐOẠN)

```
[1 Định nghĩa bài toán] → [2 Dữ liệu & kiểm định] → [3 Tiền xử lý & chia tập]
        → [4 Thiết kế chuỗi mô hình M1→Mn (Ablation)] → [5 Huấn luyện: Loss + Optimizer + LR schedule]
        → [6 Đánh giá: metrics, confusion matrix, profiling, Pareto] → [7 XAI: Grad-CAM]
        → [8 Báo cáo + lộ trình cuối kỳ]
```

---

## 2. GIAI ĐOẠN 1–2: BÀI TOÁN & DỮ LIỆU

### 2.1. Định nghĩa bài toán (viết ra trước khi code)
- Đầu vào/đầu ra là gì? Số lớp? Metric chính? Ràng buộc triển khai (kích thước, độ trễ, thiết bị)?
- Vì sao bài toán cần học sâu? Ứng dụng cuối kỳ là gì?

### 2.2. Thu thập & chuẩn hóa dữ liệu
- Cân bằng lớp (ví dụ 200 mẫu/lớp). Thống nhất định dạng (vd 16 kHz mono; ảnh cùng kích thước).
- Đặt tên/thư mục theo lớp: `Data/<TenLop>/file`.

### 2.3. Kiểm định chất lượng (BẮT BUỘC – bài học Data Leakage)
- Băm **MD5 của file** và **hash nội dung đã giải mã** (PCM/pixel) → tìm bản sao chép.
- Quy tắc: **0 trùng lặp** trước khi chia train/val/test. Nếu trùng lọt vào cả train và test → accuracy ảo.
- Kiểm tra: nhãn sai, file hỏng, độ dài/kích thước bất thường, phân bố lớp.
- Ghi lại thống kê (số mẫu/lớp, phân bố độ dài…) để đưa vào báo cáo.

```python
import hashlib, pathlib, collections
seen = collections.defaultdict(list)
for p in pathlib.Path("Data").rglob("*.*"):
    seen[hashlib.md5(p.read_bytes()).hexdigest()].append(p)
dups = {h: v for h, v in seen.items() if len(v) > 1}
print("Số nhóm trùng:", len(dups))
```

---

## 3. GIAI ĐOẠN 3: TIỀN XỬ LÝ & CHIA TẬP

1. **Chuẩn hóa kích thước đầu vào** (âm thanh: center-pad/crop về độ dài cố định; ảnh: resize).
2. **Trích xuất đặc trưng** phù hợp miền dữ liệu (âm thanh → Log-Mel 128×63; ảnh → tensor chuẩn hóa).
3. **Tiền xử lý offline**: lưu thành tensor `.pt` một lần để huấn luyện nhanh.
4. **Chia Stratified 80/10/10** (train/val/test), cố định `seed`, mỗi lớp chia đều.
5. **Tăng cường dữ liệu chỉ ở tập train** (âm thanh: SpecAugment; ảnh: flip/crop/color…). Tắt hoàn toàn khi val/test.
6. Tính mean/std **chỉ từ tập train**.

---

## 4. GIAI ĐOẠN 4: THIẾT KẾ CHUỖI MÔ HÌNH (ABLATION) – TRÁI TIM BÀI

Nguyên tắc: **xuất phát từ baseline đơn giản, mỗi bước chỉ thay đổi MỘT thành phần**, giữ nguyên siêu tham số để cô lập tác dụng.

| Mô hình | Thay đổi so với bước trước | Kiến thức môn học | Câu hỏi phải trả lời |
| :--- | :--- | :--- | :--- |
| **M1 Baseline** | CNN thuần: Conv + ReLU + Pool + Flatten + FC | Conv, Pooling, FC | Mốc so sánh; có nghẽn tham số ở FC không? |
| **M2 +Dropout** | Thêm Dropout | Regularization | Có giảm overfit/co-adaptation không? |
| **M3 +BatchNorm** | Thêm BatchNorm2d | Chuẩn hóa kích hoạt | Hội tụ có ổn định/nhanh hơn không? |
| **M4 ResNet + GAP** | Residual block, thay Flatten+FC bằng Global Average Pooling | Skip connection, GAP | Giảm bao nhiêu % tham số? Gradient có ổn hơn không? |
| **M5 +Attention (SE)** | Thêm Squeeze-and-Excitation | Channel attention | Tái hiệu chuẩn kênh có cải thiện loss/độ bền/giải thích không? |

Lựa chọn thay thế theo bài toán (chọn kiến thức đã học phù hợp):
- **Giảm chi phí tính toán**: Depthwise Separable Conv (MobileNet), Grouped Conv (ResNeXt).
- **Mở rộng tầm nhìn không tăng tham số**: Dilated Conv.
- **Đa tỷ lệ**: Inception (GoogleNet).
- **Mất cân bằng lớp**: Focal Loss. **Học đặc trưng/nhận diện**: Triplet Loss.

> Lưu ý: M5 có thể có Params/FLOPs/Latency **cao hơn** M4 (SE là module cộng thêm). Điều này bình thường – cần **biện luận cost–benefit**: chi phí tăng bao nhiêu % và nhận lại điều gì (loss thấp hơn, margin an toàn hơn, Grad-CAM tập trung hơn, khả năng chống nhiễu).

Quy tắc code:
- Tự viết `nn.Module`, **không dùng trọng số pretrained**.
- Mỗi mô hình có hàm đếm Params/FLOPs (thop hoặc tự tính).
- Cố định seed, cùng pipeline dữ liệu cho mọi mô hình.

---

## 5. GIAI ĐOẠN 5: HUẤN LUYỆN

### 5.1. Hàm mất mát
- Baseline: CrossEntropy. Nâng cấp: **Label Smoothing CE (α=0.1)** chống overconfidence. Mất cân bằng → Focal Loss.

### 5.2. Benchmark bộ tối ưu (trên mô hình đề xuất)
So sánh **SGD → SGD+Momentum → Adam → AdamW** với cùng số epoch:
- Vẽ đường cong loss/accuracy, nhận xét tốc độ hội tụ và điểm dừng.
- Chọn cuối cùng thường là **AdamW + Cosine Annealing LR** (weight decay tách rời, LR giảm mượt).

### 5.3. Thiết lập chuẩn
- Batch size, epoch, LR, weight decay ghi rõ trong bảng siêu tham số.
- Lưu **checkpoint tốt nhất theo val** (không chọn theo test).
- Lưu log (`ablation_summary.json`) để vẽ bảng/biểu đồ về sau.

---

## 6. GIAI ĐOẠN 6: ĐÁNH GIÁ KHOA HỌC

1. **Metric trên tập test**: accuracy, loss, (precision/recall/F1 theo lớp nếu cần).
2. **Confusion matrix** – phân tích lớp nào dễ nhầm, vì sao (âm gần giống, nhiễu…).
3. **Profiling phần cứng**: Params, FLOPs, độ trễ CPU (warm-up + trung bình ≥100 lần), kích thước file `.pth`.
4. **Pareto trade-off**: Accuracy vs Params và Accuracy vs FLOPs/Latency; mô hình tốt nằm trên đường biên.
5. **Nếu mọi mô hình cùng đạt trần accuracy**: so sánh thêm bằng test loss, độ tự tin, biên phân lớp, và (khi có) độ bền với nhiễu/biến đổi.

---

## 7. GIAI ĐOẠN 7: AI GIẢI THÍCH ĐƯỢC (XAI)

- **Grad-CAM** trên lớp conv cuối: kiểm tra mô hình "nhìn" vào đâu (âm thanh: vùng formant/âm vị; ảnh: vật thể).
- Chọn ≥3 mẫu thuộc các lớp khác nhau, gồm cả lớp "nhiễu/không xác định".
- Kết luận: mô hình học đặc trưng thật hay học vẹt/nhiễu nền.

---

## 8. GIAI ĐOẠN 8: BÁO CÁO

### 8.1. Cấu trúc đề xuất (6 chương)
1. **Giới thiệu**: bài toán, mục tiêu, ràng buộc, đóng góp, sơ đồ pipeline tổng thể.
2. **Dữ liệu & tiền xử lý**: thống kê, kiểm định không trùng lặp, đặc trưng, chia tập, tăng cường.
3. **Kiến trúc**: M1→Mn, sơ đồ khối, công thức (Conv, BN, Residual, SE…).
4. **Thực nghiệm**: Loss, Optimizer, Ablation (bảng + phân tích), Pareto, Confusion matrix, Grad-CAM.
5. **Thảo luận & hướng triển khai**: lượng tử hóa, kiến trúc real-time, hạn chế.
6. **Kết luận**: đóng góp chính + lộ trình cuối kỳ.

### 8.2. Quy tắc viết
- **Mọi số liệu phải khớp** với file kết quả (`ablation_summary.json`, biểu đồ). Rà soát lại mỗi lần chạy lại thực nghiệm.
- Mỗi hình/bảng đều có đánh số, chú thích và được **bình luận trong văn bản** (không chèn rồi bỏ đó).
- Không đưa thông tin nội bộ (ví dụ "đây là giữa kỳ nên data sạch") hay câu hỏi cá nhân vào báo cáo; lồng biện luận vào đúng mục phân tích.
- Không tạo đề mục con thừa; lồng ghép biện luận vào mục Ablation.
- Khi thay đổi dữ liệu/kết quả: **vẽ lại toàn bộ biểu đồ và cập nhật cả bản tiếng Anh lẫn tiếng Việt**.

---

## 9. CHECKLIST TRƯỚC KHI NỘP

- [ ] Dữ liệu: 0 trùng lặp, cân bằng lớp, chia stratified có seed cố định.
- [ ] Mô hình tự xây, không pretrained; có ít nhất một baseline + chuỗi nâng cấp từng bước.
- [ ] Vận dụng rõ kiến thức môn (Dropout, BN, Residual, GAP, Attention, AdamW, Cosine, Label Smoothing, Grad-CAM…).
- [ ] Bảng Ablation đủ: Params, FLOPs, độ trễ, accuracy, loss.
- [ ] Benchmark ≥3–4 optimizer.
- [ ] Confusion matrix, Pareto, Grad-CAM có phân tích.
- [ ] Mọi số liệu trong báo cáo khớp kết quả; hình ảnh đã cập nhật.
- [ ] Có lộ trình cuối kỳ (dữ liệu nhiễu, real-time, tối ưu nhúng).

---

## 10. CẤU TRÚC THƯ MỤC MẪU CHO BÀI MỚI

```
Project/
├── Data/<Lop>/...                 # dữ liệu sạch, 0 trùng lặp
├── src/
│   ├── check_duplicates.py        # kiểm định hash
│   ├── preprocess_offline.py      # đặc trưng + chia tập → dataset.pt
│   ├── dataset.py                 # Dataset + augmentation (chỉ train)
│   ├── models.py                  # M1…Mn viết tay
│   ├── utils.py                   # đếm Params/FLOPs, đo latency, seed
│   ├── train_ablation.py          # huấn luyện + optimizer benchmark
│   ├── evaluate.py                # confusion matrix, Pareto
│   └── gradcam.py                 # XAI
├── results/                       # checkpoint, json, biểu đồ
└── report/                        # báo cáo Word + hình
```

## 11. KHUNG CODE HUẤN LUYỆN TỐI GIẢN

```python
import torch, torch.nn as nn

def set_seed(s=42):
    import random, numpy as np
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def run(model, train_loader, val_loader, epochs=35, lr=1e-3, wd=1e-4):
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    best = 0
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
        sched.step()
        # đánh giá val -> lưu checkpoint tốt nhất theo val
```

---

## 12. LỖI THƯỜNG GẶP

| Lỗi | Hậu quả | Cách tránh |
| :--- | :--- | :--- |
| File trùng giữa train/test | Accuracy ảo | Hash trước khi chia |
| Chọn checkpoint theo test | Đánh giá thiên lệch | Chọn theo val |
| Augment cả val/test | Kết quả không ổn định | Chỉ augment train |
| Đổi nhiều thứ cùng lúc giữa các M | Không biết thành phần nào hiệu quả | Mỗi bước đổi 1 thứ |
| Số liệu báo cáo lệch kết quả | Mất uy tín | Rà soát lại sau mỗi lần chạy |
| Quên cập nhật hình khi đổi dữ liệu | Báo cáo mâu thuẫn | Vẽ lại toàn bộ hình |
