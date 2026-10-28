# 🎙️ GIAI ĐOẠN CUỐI KỲ: TRIỂN KHAI THỜI GIAN THỰC (REAL-TIME EDGE AI & DIGITAL TWIN ROBOT)

Dự án phát triển hệ thống điều khiển Robot di động thời gian thực bằng giọng nói tiếng Việt dựa trên mô hình học sâu SOTA (**BC-ResNet**), ứng dụng công nghệ WebAssembly và kết nối không dây Bluetooth Low Energy (BLE) với vi điều khiển ESP32.

---

## 🌟 1. TỔNG QUAN NÂNG CẤP CUỐI KỲ

Trong giai đoạn Cuối kỳ, toàn bộ hệ thống được chuyển dịch từ môi trường nghiên cứu thực nghiệm offline sang **triển khai thời gian thực hoàn chỉnh trên thiết bị biên (Real-time Edge Deployment)**:

| Tiêu chí | Giai đoạn Giữa kỳ (MT) | Giai đoạn Cuối kỳ (FN) | Đột phá đạt được |
| :--- | :--- | :--- | :--- |
| **Mô hình AI** | M5: `SE_VoiceResNet` (From-scratch) | **`BC-ResNet` (Qualcomm AI Research SOTA)** | Nhẹ hơn 49.3%, trễ chỉ **4.84 ms**, tối ưu hoàn hảo cho WebAssembly |
| **Tập dữ liệu** | `Data_MT` (1,400 mẫu - 200/lớp) | **`Data_FN` (2,900 mẫu: 500 `RePhai` + 400/lớp khác)** | Hơn gấp đôi kích thước, đa dạng ngữ điệu và sắc thái âm thanh |
| **Cơ chế thu âm** | Cắt lát tĩnh từ file âm thanh | **Push-to-Talk 2.0s chuẩn hóa thời gian thực** | Loại bỏ 100% lỗi lệch pha thời gian và cắt dở câu lệnh |
| **Xử lý âm học** | Chia 32768 thông thường | **Sinc Resampler + Normalization 0.65 + Silence Guard** | Triệt tiêu lỗi im lặng ra "Rẽ Phải", độ chính xác realtime đạt **90–95%** |
| **Giao diện & Điều khiển**| Báo cáo tĩnh Word & Notebook | **Digital Twin 2D Dashboard + BLE ESP32 Hardware** | Tương tác hai chiều: Mô phỏng sa bàn + Điều khiển xe vật lý |
| **Nền tảng chạy** | PyTorch trên máy tính | **WebAssembly ONNX Runtime Web (Serverless)** | Chạy mượt mà trực tiếp trên mọi trình duyệt web di động |

---

## 🧠 2. MÔ HÌNH SOTA BC-RESNET (QUALCOMM KWS)

Mô hình **BC-ResNet (Broadcasted Residual Network)** là kiến trúc giữ Top-1 thế giới trên bảng xếp hạng Google Speech Commands dành cho bài toán Keyword Spotting (KWS):

### 2.1. Bảng so sánh thông số kỹ thuật M5 vs BC-ResNet:

| Chỉ số kỹ thuật | M5: `SE_VoiceResNet` (Giữa kỳ) | `BC-ResNet` (Cuối kỳ) | Mức độ cải thiện |
| :--- | :---: | :---: | :--- |
| **Số lượng tham số (Parameters)** | 310,503 | **157,415** | 🟢 **Giảm 49.3% (Nhẹ gần gấp đôi)** |
| **Khối lượng tính toán (FLOPs)** | 771.69 M | **83.41 M** | 🟢 **Giảm 89.2% khối lượng phép tính** |
| **Độ trễ suy luận (CPU Latency)**| 5.48 ms | **4.84 ms** | 🟢 **Nhanh hơn 11.7%** |
| **Kích thước file ONNX** | 1.22 MB | **0.60 MB (617 KB)** | 🟢 **Tải cực nhanh trên mạng 4G/5G** |
| **Tập dữ liệu huấn luyện** | 1,400 mẫu | **2,900 mẫu (`Data_FN`)** | 🟢 **Mở rộng hơn gấp đôi quy mô dữ liệu** |
| **Test Accuracy (Offline)** | 100.00% | **100.00%** | 🟢 Đánh giá khắt khe trên 290 mẫu test |
| **Cơ chế đặc trưng** | 2D SE-Channel Attention | **Broadcasted Temporal-Frequency Learning** | Bắt trọn vẹn đặc trưng Formant động của tiếng Việt |

### 2.2. Bảng kết quả phân loại chi tiết trên tập Kiểm thử độc lập (290 mẫu Test của `Data_FN`):

| Nhãn khẩu lệnh | Precision | Recall | F1-Score | Số mẫu kiểm thử (Support) |
| :--- | :---: | :---: | :---: | :---: |
| **KhoiDong** | **1.0000** | **1.0000** | **1.0000** | 40 mẫu |
| **KhongXacDinh** | **1.0000** | **1.0000** | **1.0000** | 40 mẫu |
| **Lui** | **1.0000** | **1.0000** | **1.0000** | 40 mẫu |
| **RePhai** | **1.0000** | **1.0000** | **1.0000** | 50 mẫu |
| **ReTrai** | **1.0000** | **1.0000** | **1.0000** | 40 mẫu |
| **TamDung** | **1.0000** | **1.0000** | **1.0000** | 40 mẫu |
| **Tien** | **1.0000** | **1.0000** | **1.0000** | 40 mẫu |
| **TRUNG BÌNH (Macro Avg)** | **1.0000** | **1.0000** | **1.0000** | **290 mẫu** |
| **ĐỘ CHÍNH XÁC TOÀN BỘ** | — | — | **100.00%** | **(290 / 290 mẫu chính xác)** |

---

## 🛡️ 3. CÁC NÂNG CẤP ÂM HỌC TRIỆT TIÊU LỆCH MIỀN (ACOUSTIC DOMAIN SHIFT)

1. **Bộ lọc bảo vệ khoảng lặng (Silence Guard)**:
   * Ngắt suy luận ngay lập tức nếu năng lượng $\text{Peak} < 0.025$ hoặc $\text{RMS} < 0.003$.
   * **Kết quả**: Triệt tiêu $100\%$ lỗi ma trận trọng số tự động dự đoán ra *"Rẽ Phải"* khi người dùng không nói gì.
2. **Chuẩn hóa biên độ thích nghi (Peak Normalization to 0.65)**:
   * Tự động khuếch đại tín hiệu thô từ Web Audio API về đúng mức đỉnh trung bình của tập huấn luyện `Data_FN`.
   * **Kết quả**: Dù người dùng nói to hay nói nhỏ, độ chính xác luôn đạt tối ưu.
3. **Bộ hạ tần số lấy mẫu Polyphase Sinc (`OfflineAudioContext`)**:
   * Sử dụng C++ engine phần cứng của WebKit/Blink hạ mẫu mượt mà từ $44.1\text{k}/48\text{k}$ về $16,000\text{ Hz}$ không méo pha và triệt tiêu răng cưa.
4. **Bảo tồn thời lượng tự nhiên (Preserve Natural Pause Envelope)**:
   * Giữ trọn vẹn phụ âm đầu (*T, Kh, R*) và căn giữa hoàn hảo vào khung $2.0\text{s}$ ($32,000$ mẫu).
5. **Tính năng kiểm chứng "🔊 Nghe lại âm AI"**:
   * Cho phép người dùng bấm nghe lại trực tiếp đoạn âm thanh $2.0\text{s}$ nạp vào AI.

---

## 📁 4. CẤU TRÚC THƯ MỤC `FN/`

```text
FN/
├── README.md                           # [FILE NÀY] Tài liệu cẩm nang giai đoạn Cuối Kỳ
├── preprocess_data_fn.py               # Tiền xử lý Data_FN (2,900 mẫu) sang dataset_processed_2800.pt
├── models.py                           # Định nghĩa kiến trúc SOTA BC-ResNet & M5 Comparator
├── train_bc_resnet.py                  # Kịch bản huấn luyện BC-ResNet (AdamW, Cosine Annealing, SpecAugment)
├── export_onnx.py                      # Kịch bản xuất mô hình ONNX tự thân cho WebAssembly
├── realtime_eval.py                    # Trình giả lập & kiểm thử suy luận Realtime
├── firmware/
│   └── esp32_ble_robot.ino             # Mã nguồn ESP32 Bluetooth Low Energy điều khiển Xe Robot
└── results/
    ├── BC_ResNet_AdamW.pth             # Trọng số mô hình SOTA BC-ResNet (~654 KB, 100.00% Test Acc)
    ├── bc_resnet.onnx                  # File mô hình ONNX Standalone Web (~617 KB)
    ├── curves_BC_ResNet_AdamW.png      # Đồ thị học tập Loss & Accuracy của BC-ResNet
    └── confusion_matrix_bc_resnet.png  # Ma trận nhầm lẫn 7 lớp trên 290 mẫu Test
```

> **Giao diện Web Digital Twin**: Mã nguồn Web Dashboard triển khai chính thức được đồng bộ tại thư mục `docs/` để kích hoạt máy chủ tĩnh serverless trên **GitHub Pages**.

---

## 🚀 5. HƯỚNG DẪN THỰC THI (QUICK START)

### 5.1. Tiền xử lý siêu tập dữ liệu Data_FN
```bash
python3 FN/preprocess_data_fn.py
```

### 5.2. Huấn luyện lại mô hình SOTA BC-ResNet
```bash
python3 FN/train_bc_resnet.py
```

### 5.3. Xuất mô hình sang ONNX Standalone và đồng bộ sang Web
```bash
python3 FN/export_onnx.py
```

### 5.4. Kiểm thử Pipeline mô phỏng thời gian thực
```bash
python3 FN/realtime_eval.py
```

### 5.5. Nạp Firmware cho Xe Robot vật lý (ESP32)
1. Mở tệp `FN/firmware/esp32_ble_robot.ino` trong **Arduino IDE**.
2. Chọn Board: **ESP32 Dev Module**.
3. Cài đặt thư viện: `BLEDevice`, `BLEServer`, `BLEUtils`.
4. Nạp code vào ESP32. Khi khởi động, xe sẽ phát Bluetooth với tên: `ESP32_ROBOT_VOICE`.

### 5.6. Trải nghiệm Trực tiếp trên Web Dashboard
* Truy cập trang web chính thức: **[https://hahiduong23.github.io/Robot-Voice-Digital-Twin/](https://hahiduong23.github.io/Robot-Voice-Digital-Twin/)**
* Bấm **Ấn giữ để nói** $\rightarrow$ Đọc khẩu lệnh $\rightarrow$ Thả nút $\rightarrow$ Quan sát Robot phản hồi trên sa bàn 2D và truyền lệnh Bluetooth tới xe thật!
