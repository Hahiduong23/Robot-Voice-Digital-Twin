# Advanced Deep Learning - Repository & Project Portfolio
> **Môn học:** Học Sâu Nâng Cao (Advanced Deep Learning)  
> **Giảng viên hướng dẫn:** TS. Lê Minh Huy | ICSLab, Trường Đại học Phenikaa  
> **Sinh viên thực hiện:** Nguyễn Hải Dương (Trưởng nhóm / Chủ nhiệm dự án)  
> **Giai đoạn Giữa kỳ (MT):** Huấn luyện From-scratch M1-M5, Báo cáo & Ablation Study tại [MT/](MT/)  
> **Giai đoạn Cuối kỳ (FN):** Triển khai Thời gian thực SOTA BC-ResNet, Digital Twin & Robot BLE tại [FN/](FN/)  
> **Web Dashboard Real-time:** Trực tiếp trên GitHub Pages tại [https://hahiduong23.github.io/Robot-Voice-Digital-Twin/](https://hahiduong23.github.io/Robot-Voice-Digital-Twin/)  
> **Tài liệu bối cảnh AI Master:** Chi tiết tại [PROJECT_CONTEXT_MASTER.md](PROJECT_CONTEXT_MASTER.md)

---

## 🌟 1. ĐỒ ÁN TRỌNG TÂM: NHẬN DIỆN GIỌNG NÓI ĐIỀU KHIỂN ROBOT DI ĐỘNG (KWS)
**Đề tài:** *Deep Learning-Based Keyword Spotting for Mobile Robot Voice Control on Resource-Constrained Embedded Edge Platforms*  
**Báo cáo học thuật:** [Report/DLPRP.docx](Report/DLPRP.docx) (Tiếng Anh) & [Report/DLPRP_VI.docx](Report/DLPRP_VI.docx) (Tiếng Việt) | **Slide thuyết trình:** [MT/END_TO_END_PIPELINE.pptx](MT/END_TO_END_PIPELINE.pptx)

### 1.1. Mục tiêu bài toán
- Xây dựng hệ thống nhận diện từ khóa giọng nói tiếng Việt ngắn (Keyword Spotting - KWS) độ trễ thấp để điều khiển robot di động trên các thiết bị tính toán biên giới hạn tài nguyên (Raspberry Pi 4, Jetson Nano, ESP32, Trình duyệt Web di động).
- **Phân loại 7 lớp khẩu lệnh tiếng Việt:** `KhoiDong`, `KhongXacDinh` (tạp âm môi trường), `Lui`, `RePhai`, `ReTrai`, `TamDung`, `Tien`.
- **Ràng buộc phần cứng biên:** Dung lượng mô hình < 10 MB (tối ưu < 2 MB weights), FLOPs < 50 MFLOPs, độ trễ CPU < 20 ms.

### 1.2. Bộ dữ liệu chuẩn hóa `Data_MT/`
- Gồm **1,400 file âm thanh .wav** (16 kHz, 16-bit Mono PCM).
- Cân bằng hoàn hảo: Đúng **200 mẫu/lớp** trên cả 7 lớp.
- Kiểm định chất lượng: **100% Unique (0 duplicate)** qua kiểm tra mã hash MD5 và PCM raw audio bytes.
- Quy trình tiền xử lý: Cố định 2.0s (Center-pad/crop đối xứng), trích xuất Log-Mel Spectrogram (128 dải Mel x 63 khung thời gian), phân chia Stratified 80:10:10 (Train: 1,120; Val: 140; Test: 140) đóng gói tại `MT/dataset_processed.pt` (45.17 MB).

### 1.3. Bảng so sánh tiến hóa kiến trúc (M1 -> M5 -> SOTA BC-ResNet)

| Mô hình | Giai đoạn | Kiến trúc đặc trưng | Tham số (Params) | FLOPs | Độ trễ CPU (ms) | Test Accuracy | Đánh giá kỹ thuật |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **M1** | Giữa kỳ | BaselineCNN (3 Conv + Flatten + 2 Dense) | 1,928,711 | 5.51 M | 1.95 ms | 100.00% | Bị nghẽn tham số ở tầng FC (>95%) |
| **M2** | Giữa kỳ | DropoutCNN (M1 + Dropout 0.4) | 1,928,711 | 5.51 M | 1.89 ms | 100.00% | Hạn chế co-adaptation |
| **M3** | Giữa kỳ | BatchNormCNN (M1 + BatchNorm2d) | 1,929,159 | 5.51 M | 2.13 ms | 100.00% | Ổn định phân phối kích hoạt nội tầng |
| **M4** | Giữa kỳ | VoiceResNet (ResNet Blocks + GAP) | 307,687 | 14.94 M | 4.82 ms | 100.00% | **Cắt giảm 84.05% tham số** |
| **M5** | Giữa kỳ | **SE_VoiceResNet (Đề xuất Giữa kỳ)** | **310,503** | **14.95 M** | **5.48 ms** | **100.00%** | **Tái hiệu chuẩn phổ kênh (Attention)** |
| **BC-ResNet** | **Cuối kỳ** | **BC-ResNet (Qualcomm SOTA KWS)** | **157,415** | **6.82 M** | **4.71 ms** | **100.00%** | 🟢 **Nhẹ hơn 49.3%, tối ưu WebAssembly** |

---

## 📁 2. CẤU TRÚC KHO LƯU TRỮ (REPOSITORY STRUCTURE)

```text
Deep_Learning/
│
├── MT/                                     # [GIAI ĐOẠN GIỮA KỲ]: Nghiên cứu & Huấn luyện From-Scratch
│   ├── preprocess_offline.py               # Tiền xử lý Log-Mel Spectrogram & tuần tự hóa .pt
│   ├── train_ablation.py                   # Huấn luyện M1-M5, Đo FLOPs/Latency, Xuất Grad-CAM
│   ├── models.py                           # Định nghĩa kiến trúc M1 -> M5 (from scratch)
│   ├── utils.py                            # FLOPs, Latency profiling, Grad-CAM engine
│   ├── dataset_processed.pt                # Tensor nén sẵn (Train: 1120, Val: 140, Test: 140)
│   ├── results/                            # Checkpoint tốt nhất & toàn bộ biểu đồ 300 DPI
│   │   ├── M5_SE_VoiceResNet_AdamW.pth     # Trọng số mô hình M5 tốt nhất (~1.27 MB)
│   │   ├── confusion_matrix_M5.png         # Ma trận nhầm lẫn 140/140 mẫu (20/20 mỗi lớp)
│   │   ├── optimizer_comparison.png        # Đồ thị hội tụ Loss/Accuracy 4 bộ tối ưu
│   │   └── pareto_tradeoff.png             # Đường biên Pareto (Latency vs Params vs Acc)
│   └── END_TO_END_PIPELINE.pptx            # Slide thuyết trình End-to-End Pipeline
│
├── FN/                                     # [GIAI ĐOẠN CUỐI KỲ]: Triển khai Thời gian thực SOTA & Robot
│   ├── README.md                           # Cẩm nang toàn diện giai đoạn Cuối Kỳ
│   ├── models.py                           # Kiến trúc SOTA BC-ResNet (Qualcomm KWS) & M5
│   ├── train_bc_resnet.py                  # Huấn luyện SOTA BC-ResNet (AdamW, Cosine, SpecAugment)
│   ├── export_onnx.py                      # Xuất mô hình ONNX tự thân cho WebAssembly
│   ├── realtime_eval.py                    # Trình giả lập & kiểm thử suy luận Realtime
│   ├── firmware/
│   │   └── esp32_ble_robot.ino             # Mã nguồn ESP32 Bluetooth Low Energy điều khiển Xe Robot
│   └── results/
│       ├── BC_ResNet_AdamW.pth             # Trọng số mô hình SOTA BC-ResNet (157k params, ~654 KB)
│       ├── bc_resnet.onnx                  # File mô hình ONNX WebAssembly (~617 KB)
│       └── curves_BC_ResNet_AdamW.png      # Đồ thị học tập Loss & Accuracy của BC-ResNet
│
├── docs/                                   # [WEB DIGITAL TWIN]: Triển khai trực tiếp trên GitHub Pages
│   ├── index.html                          # Giao diện Cyberpunk/Neon Dashboard giám sát thời gian thực
│   ├── app.js                              # Bộ điều phối chính, ONNX Runtime Web, Push-to-Talk state
│   ├── audio_engine.js                     # Audio Engine: Sinc Resample, Silence Guard, Norm 0.65, Playback
│   ├── ble_controller.js                   # Web Bluetooth API kết nối trực tiếp Xe Robot vật lý
│   ├── digital_twin.js                     # Mô phỏng Sa bàn Robot 2D Digital Twin
│   ├── audio_constants.json                # Bộ lọc Mel, cửa sổ Hann, danh sách nhãn lệnh
│   └── models/bc_resnet.onnx               # Mô hình ONNX phục vụ WebAssembly
│
├── Data_MT/                                # BỘ DỮ LIỆU CHUẨN DUY NHẤT (1,400 file WAV 16kHz mono)
│   ├── KhoiDong/ (200)                     # Khẩu lệnh Khởi động
│   ├── KhongXacDinh/ (200)                 # Tạp âm môi trường / Ngoài luồng
│   ├── Lui/ (200)                          # Khẩu lệnh Lùi
│   ├── RePhai/ (200)                       # Khẩu lệnh Rẽ Phải
│   ├── ReTrai/ (200)                       # Khẩu lệnh Rẽ Trái
│   ├── TamDung/ (200)                      # Khẩu lệnh Tạm Dừng / Dừng khẩn cấp
│   └── Tien/ (200)                         # Khẩu lệnh Tiến
│
├── Report/                                 # BÁO CÁO KHOA HỌC & TÀI LIỆU NỘP BÀI
│   ├── DLPRP.docx                          # Báo cáo kỹ thuật tiếng Anh hoàn chỉnh
│   ├── DLPRP_VI.docx                       # Báo cáo kỹ thuật tiếng Việt hoàn chỉnh
│   └── DL_FN.docx / DL_FN.pdf              # Báo cáo tổng kết đồ án
│
├── HW/                                     # CÁC BÀI TẬP THỰC HÀNH TRƯỚC ĐÓ
│   ├── Bai_tap_1/                          # AlexNet vs VGG10 vs ResNet10 (CIFAR-100)
│   ├── Bai_tap_2/                          # GoogleNet-10 vs ResNeXt-10 vs SENet-10 (CIFAR-100)
│   └── Bai_tap_3_AI_Fitness_Trainer/       # Ứng dụng Pose Estimation đếm số lần tập Gym
│
├── PROJECT_CONTEXT_MASTER.md               # HỒ SƠ NGỮ CẢNH MASTER TOÀN DIỆN CHO AI & DEVELOPER
├── MIDTERM_WORKFLOW_GUIDE.md               # CẨM NANG QUY TRÌNH 8 BƯỚC BÀI TẬP GIỮA KỲ
├── REALTIME_NOISE_ROBUSTNESS_PLAN.md       # TÀI LIỆU ĐỘT PHÁ TRIỂN KHAI REALTIME VÀ ÂM HỌC
└── README.md                               # [FILE NÀY] Tài liệu tổng quan toàn bộ kho lưu trữ
```

---

## 🚀 3. HƯỚNG DẪN THỰC THI NHANH (QUICK START)

### 3.1. Giai đoạn Giữa kỳ (MT): Huấn luyện From-scratch M1 -> M5
```bash
# 1. Tiền xử lý dữ liệu offline:
python3 MT/preprocess_offline.py

# 2. Huấn luyện thực nghiệm so sánh M1-M5 và 4 bộ tối ưu:
python3 MT/train_ablation.py
```

### 3.2. Giai đoạn Cuối kỳ (FN): Mô hình SOTA BC-ResNet & Kiểm thử Real-time
```bash
# 1. Huấn luyện mô hình SOTA BC-ResNet:
python3 FN/train_bc_resnet.py

# 2. Xuất mô hình sang ONNX Standalone Web:
python3 FN/export_onnx.py

# 3. Kiểm thử Pipeline Realtime mô phỏng âm học:
python3 FN/realtime_eval.py
```

### 3.3. Trải nghiệm Giao diện Web Digital Twin Real-time
* Truy cập trực tiếp link GitHub Pages: **[https://hahiduong23.github.io/Robot-Voice-Digital-Twin/](https://hahiduong23.github.io/Robot-Voice-Digital-Twin/)**
* Hoặc chạy dev server cục bộ:
```bash
cd docs && python3 -m http.server 8000
# Mở trình duyệt tại http://localhost:8000
```

---

## 📚 4. BÁO CÁO & TÀI LIỆU KHOA HỌC
- **Báo cáo nộp chính thức (Song ngữ):**
  - Tiếng Anh: [Report/DLPRP.docx](Report/DLPRP.docx)
  - Tiếng Việt: [Report/DLPRP_VI.docx](Report/DLPRP_VI.docx)
- **Slide thuyết trình End-to-End Pipeline:**
  - [MT/END_TO_END_PIPELINE.pptx](MT/END_TO_END_PIPELINE.pptx)
- **Hồ sơ ngữ cảnh kỹ thuật Master cho AI & Kỹ sư:**
  - [PROJECT_CONTEXT_MASTER.md](PROJECT_CONTEXT_MASTER.md)
- **Kế hoạch & Phân tích âm học Realtime:**
  - [REALTIME_NOISE_ROBUSTNESS_PLAN.md](REALTIME_NOISE_ROBUSTNESS_PLAN.md)
- **Cẩm nang giai đoạn Cuối kỳ:**
  - [FN/README.md](FN/README.md)
