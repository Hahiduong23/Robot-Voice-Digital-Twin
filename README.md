# Advanced Deep Learning - Repository & Project Portfolio
> **Môn học:** Học Sâu Nâng Cao (Advanced Deep Learning)  
> **Giảng viên hướng dẫn:** TS. Lê Minh Huy | ICSLab, Trường Đại học Phenikaa  
> **Sinh viên thực hiện:** Nguyễn Hải Dương (Trưởng nhóm / Chủ nhiệm dự án)  
> **Tài liệu bối cảnh AI Master:** Chi tiết tại [PROJECT_CONTEXT_MASTER.md](PROJECT_CONTEXT_MASTER.md)  
> **Cẩm nang quy trình bài tập giữa kỳ:** Chi tiết tại [MIDTERM_WORKFLOW_GUIDE.md](MIDTERM_WORKFLOW_GUIDE.md)

---

## 🌟 1. ĐỒ ÁN TRỌNG TÂM: NHẬN DIỆN GIỌNG NÓI ĐIỀU KHIỂN ROBOT DI ĐỘNG (KWS)
**Đề tài:** *Deep Learning-Based Keyword Spotting for Mobile Robot Voice Control on Resource-Constrained Embedded Edge Platforms*  
**Báo cáo học thuật:** [Bao_cao/DLPRP.docx](Bao_cao/DLPRP.docx) (Tiếng Anh) & [Bao_cao/DLPRP_VI.docx](Bao_cao/DLPRP_VI.docx) (Tiếng Việt) | **Slide thuyết trình:** [ADL/END_TO_END_PIPELINE.pptx](ADL/END_TO_END_PIPELINE.pptx)

### 1.1. Mục tiêu bài toán
- Xây dựng hệ thống nhận diện từ khóa giọng nói tiếng Việt ngắn (Keyword Spotting - KWS) độ trễ thấp để điều khiển robot di động trên các thiết bị tính toán biên giới hạn tài nguyên (Raspberry Pi 4, Jetson Nano).
- **Phân loại 7 lớp khẩu lệnh tiếng Việt:** `KhoiDong`, `KhongXacDinh` (tạp âm môi trường), `Lui`, `RePhai`, `ReTrai`, `TamDung`, `Tien`.
- **Ràng buộc phần cứng biên:** Dung lượng mô hình < 10 MB (tối ưu < 2 MB weights), FLOPs < 50 MFLOPs, độ trễ CPU < 20 ms.

### 1.2. Bộ dữ liệu chuẩn hóa `Data_1400/`
- Gồm **1,400 file âm thanh .wav** (16 kHz, 16-bit Mono PCM).
- Cân bằng hoàn hảo: Đúng **200 mẫu/lớp** trên cả 7 lớp.
- Kiểm định chất lượng: **100% Unique (0 duplicate)** qua kiểm tra mã hash MD5 và PCM raw audio bytes.
- Quy trình tiền xử lý: Cố định 2.0s (Center-pad/crop đối xứng), trích xuất Log-Mel Spectrogram (128 dải Mel x 63 khung thời gian), phân chia Stratified 80:10:10 (Train: 1,120; Val: 140; Test: 140) đóng gói tại `ADL/dataset_processed.pt` (43.08 MB).

### 1.3. Bảng kết quả nghiên cứu bóc tách (Ablation Study M1 -> M5)

| Mô hình | Kiến trúc đặc trưng | Tham số (Params) | FLOPs | Độ trễ CPU (ms) | Test Accuracy | Test Loss | Đánh giá |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **M1** | BaselineCNN (3 Conv + Flatten + 2 Dense) | 1,928,711 | 5.51 M | 1.95 ms | 100.00% | 0.4776 | Bị nghẽn tham số ở tầng FC (>95%) |
| **M2** | DropoutCNN (M1 + Dropout 0.4) | 1,928,711 | 5.51 M | 1.89 ms | 100.00% | 0.4822 | Hạn chế co-adaptation |
| **M3** | BatchNormCNN (M1 + BatchNorm2d) | 1,929,159 | 5.51 M | 2.13 ms | 100.00% | 0.4975 | Ổn định phân phối kích hoạt nội tầng |
| **M4** | VoiceResNet (ResNet Blocks + GAP) | 307,687 | 14.94 M | 4.82 ms | 100.00% | 0.5010 | **Cắt giảm 84.05% tham số** |
| **M5** | **SE_VoiceResNet (Đề xuất tối ưu)** | **310,503** | **14.95 M** | **5.48 ms** | **100.00%** | **0.4826** | **Tái hiệu chuẩn phổ kênh (Attention)** |

> **Điểm đột phá của mô hình đề xuất M5 (SE_VoiceResNet):**
> - **Cơ chế lọc dải tần thích nghi (Acoustic Spectral Recalibration):** Khối Squeeze-and-Excitation tự động học trọng số động $s_c \in [0, 1]$ khuếch đại dải tần Formant giọng người (300 Hz - 3,400 Hz) và dập tắt các kênh tần số thấp bị nhiễu động cơ và quạt gió.
> - **Test Loss thấp nhất toàn diện (0.4826):** Khoảng cách biên phân lớp an toàn (Safety Margin) cao hơn M4 (0.5010), giải thích sắc nét qua bản đồ nhiệt Grad-CAM.
> - **Thời gian thực tuyệt đối:** Độ trễ CPU 5.48 ms chỉ tốn ~5.5% ngân sách 20 ms của vi xử lý nhúng.

### 1.4. Đua tốc độ các bộ tối ưu hóa (Optimizer Benchmark trên M5)
- **Vanilla SGD:** 69.29% (Bị kẹt ở cực tiểu địa phương/yên ngựa).
- **SGD + Momentum:** 100.00% (Loss: 0.5042, vượt qua điểm yên ngựa).
- **Adam:** 100.00% (Loss: 0.4908, hội tụ nhanh sau 3-5 epochs).
- **AdamW + Cosine Annealing (Tối ưu nhất):** **100.00%** (Loss: **0.4826**, hội tụ phẳng và biên an toàn cao nhất).

---

## 📁 2. CẤU TRÚC KHO LƯU TRỮ (REPOSITORY STRUCTURE)

```text
Deep_Learning/
│
├── ADL/                                    # [DỰ ÁN CHÍNH]: Hệ thống Nhận diện Giọng nói KWS
│   ├── preprocess_offline.py               # Tiền xử lý Log-Mel Spectrogram & tuần tự hóa .pt
│   ├── train_ablation.py                   # Huấn luyện M1-M5, Đo FLOPs/Latency, Xuất Grad-CAM
│   ├── models.py                           # Định nghĩa kiến trúc M1 -> M5 (from scratch)
│   ├── utils.py                            # FLOPs, Latency profiling, Grad-CAM engine
│   ├── dataset_processed.pt                # Tensor nén sẵn (Train: 1120, Val: 140, Test: 140)
│   ├── results/                            # Checkpoint tốt nhất & toàn bộ biểu đồ 300 DPI
│   │   ├── M5_SE_VoiceResNet_AdamW.pth     # Trọng số mô hình M5 tốt nhất (~1.27 MB)
│   │   ├── confusion_matrix_M5.png         # Ma trận nhầm lẫn 140/140 mẫu (20/20 mỗi lớp)
│   │   ├── optimizer_comparison.png        # Đồ thị hội tụ Loss/Accuracy 4 bộ tối ưu
│   │   ├── pareto_tradeoff.png             # Đường biên Pareto (Latency vs Params vs Acc)
│   │   ├── end_to_end_pipeline.png         # Sơ đồ quy trình tổng quan 6 giai đoạn
│   │   └── gradcam_*.png                   # Bản đồ nhiệt giải thích quyết định mạng
│   └── END_TO_END_PIPELINE.pptx            # Slide thuyết trình End-to-End Pipeline
│
├── Data_1400/                              # BỘ DỮ LIỆU CHUẨN DUY NHẤT (1,400 file WAV 16kHz mono)
│   ├── KhoiDong/ (200)                     # Khẩu lệnh Khởi động
│   ├── KhongXacDinh/ (200)                 # Tạp âm môi trường / Ngoài luồng
│   ├── Lui/ (200)                          # Khẩu lệnh Lùi
│   ├── RePhai/ (200)                       # Khẩu lệnh Rẽ Phải
│   ├── ReTrai/ (200)                       # Khẩu lệnh Rẽ Trái
│   ├── TamDung/ (200)                      # Khẩu lệnh Tạm Dừng / Dừng khẩn cấp
│   └── Tien/ (200)                         # Khẩu lệnh Tiến
│
├── Bao_cao/                                # THƯ MỤC BÁO CÁO & TÀI LIỆU NỘP BÀI
│   ├── DLPRP.docx                          # Báo cáo kỹ thuật tiếng Anh hoàn chỉnh
│   ├── DLPRP_VI.docx                       # Báo cáo kỹ thuật tiếng Việt hoàn chỉnh
│   ├── DL_FN.docx / DL_FN.pdf              # Báo cáo tổng kết đồ án
│   ├── Report template.docx                # Template định dạng chuẩn
│   └── build_slide_content_doc.py          # Script sinh kịch bản thuyết trình song ngữ
│
├── Phenikaa_Segmentation/                  # [DỰ ÁN COMPUTER VISION]: Phân vùng thực thể
│   ├── Phenikaa-Campus-Segmentation/       # Dataset 540 ảnh 5 lớp (Roboflow YOLO format)
│   ├── Phenikaa-Campus-Segmentation.zip    # File nén dataset tải lên Colab
│   └── train_phenikaa_campus_yolo11_seg.ipynb # Notebook huấn luyện YOLO11-Seg (GPU T4)
│
├── Bai_tap_cu/                             # CÁC BÀI TẬP THỰC HÀNH TRƯỚC ĐÓ
│   ├── Bai_tap_1/                          # AlexNet vs VGG10 vs ResNet10 (CIFAR-100)
│   ├── Bai_tap_2/                          # GoogleNet-10 vs ResNeXt-10 vs SENet-10 (CIFAR-100)
│   ├── Bai_tap_3_AI_Fitness_Trainer/       # Ứng dụng Pose Estimation đếm số lần tập Gym (kèm weights)
│   └── data/                               # Dữ liệu CIFAR-100
│
├── Source codes/                           # Mã nguồn tham khảo mở rộng (DLModel & Realtime)
├── Tai_lieu_tham_khao/                     # Slides bài giảng & tài liệu môn học
├── PROJECT_CONTEXT_MASTER.md               # HỒ SƠ NGỮ CẢNH MASTER TOÀN DIỆN CHO AI & DEVELOPER
├── MIDTERM_WORKFLOW_GUIDE.md               # CẨM NANG QUY TRÌNH 8 BƯỚC BÀI TẬP GIỮA KỲ TỔNG QUÁT
└── README.md                               # [FILE NÀY] Tài liệu tổng quan toàn bộ kho lưu trữ
```

---

## 🚀 3. HƯỚNG DẪN THỰC THI NHANH (QUICK START)

### 3.1. Đồ án Nhận diện Giọng nói Robot (KWS - ADL)

#### Cài đặt thư viện:
```bash
pip install torch torchaudio torchvision numpy matplotlib seaborn scikit-learn thop python-docx
```

#### Chạy tiền xử lý dữ liệu:
```bash
python3 ADL/preprocess_offline.py
```

#### Chạy huấn luyện và đánh giá thực nghiệm (M1-M5 & Optimizers):
```bash
python3 ADL/train_ablation.py
```

#### Nạp mô hình M5 để chạy suy luận (Inference):
```python
import torch
from ADL.models import get_model

device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
model = get_model('M5_SE_VoiceResNet', num_classes=7).to(device)
ckpt = torch.load('ADL/results/M5_SE_VoiceResNet_AdamW.pth', map_location=device, weights_only=False)
model.load_state_dict(ckpt['state_dict'])
model.eval()

# Input tensor phổ Mel: [1, 1, 128, 63] (cố định 2.0s)
classes = ['KhoiDong', 'KhongXacDinh', 'Lui', 'RePhai', 'ReTrai', 'TamDung', 'Tien']
with torch.no_grad():
    logits = model(x)
    pred_class = classes[torch.argmax(logits, dim=1).item()]
print(f"Dự đoán khẩu lệnh: {pred_class}")
```

---

### 3.2. Dự án Phân vùng thực thể (Phenikaa Campus Segmentation)
- Mở notebook [Phenikaa_Segmentation/train_phenikaa_campus_yolo11_seg.ipynb](Phenikaa_Segmentation/train_phenikaa_campus_yolo11_seg.ipynb) trên Google Colab với GPU T4.
- Huấn luyện mô hình YOLO11-Seg (5 lớp: `camera`, `emergency light`, `light`, `speaker`, `switch`) và đánh giá mAP trên 54 ảnh test.

---

### 3.3. Các bài tập thực hành trước đó (CIFAR-100 Labs)

#### Lab 2: GoogleNet-10 vs ResNeXt-10 vs SENet-10
```bash
cd Bai_tap_cu/Bai_tap_2
python3 models_lab2.py                     # Xem thông số Params, MACs, FLOPs
python3 train_lab2.py --epochs 10          # Huấn luyện và tự động xuất đồ thị so sánh
```

#### Lab 1: AlexNet vs VGG10 vs ResNet10
```bash
cd Bai_tap_cu/Bai_tap_1
python3 models_lab1.py                     # Xem thông số Params, MACs, FLOPs
python3 train_lab1.py --epochs 10          # Huấn luyện và tự động xuất đồ thị so sánh
```

---

## 📚 4. BÁO CÁO & TÀI LIỆU KHOA HỌC
- **Báo cáo nộp chính thức (Song ngữ):**
  - Tiếng Anh: [Bao_cao/DLPRP.docx](Bao_cao/DLPRP.docx)
  - Tiếng Việt: [Bao_cao/DLPRP_VI.docx](Bao_cao/DLPRP_VI.docx)
- **Slide thuyết trình End-to-End Pipeline:**
  - [ADL/END_TO_END_PIPELINE.pptx](ADL/END_TO_END_PIPELINE.pptx)
- **Tài liệu ngữ cảnh kỹ thuật Master cho AI & Kỹ sư:**
  - [PROJECT_CONTEXT_MASTER.md](PROJECT_CONTEXT_MASTER.md)
- **Cẩm nang quy trình 8 bước bài tập giữa kỳ:**
  - [MIDTERM_WORKFLOW_GUIDE.md](MIDTERM_WORKFLOW_GUIDE.md)
