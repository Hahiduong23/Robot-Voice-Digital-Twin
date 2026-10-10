with open("ADL/create_word_report_vi.py", "r", encoding="utf-8") as f:
    vi_code = f.read()

# Dataset intro
vi_code = vi_code.replace(
    'Tập dữ liệu thực nghiệm bao gồm 700 file ghi âm chất lượng cao lưu trữ trong thư mục Data_DL/',
    'Tập dữ liệu thực nghiệm bao gồm 3.200 file ghi âm chất lượng cao lưu trữ trong thư mục Data_DLP/'
)

# Table 2 rows
old_d_rows_vi = """    dataset_rows = [
        ("0", "KhoiDong", "Khởi động / Sẵn sàng vận hành hệ thống", "100", "1.12 - 1.84"),
        ("1", "KhongXacDinh", "Tạp âm môi trường / Tiếng nói ngoài luồng", "100", "1.04 - 1.96"),
        ("2", "Lui", "Điều khiển Robot lùi lại phía sau", "100", "1.08 - 1.62"),
        ("3", "RePhai", "Điều khiển Robot quay phải 90 độ", "100", "1.15 - 1.78"),
        ("4", "ReTrai", "Điều khiển Robot quay trái 90 độ", "100", "1.18 - 1.82"),
        ("5", "TamDung", "Dừng khẩn cấp / Tạm dừng chuyển động", "100", "1.10 - 1.65"),
        ("6", "Tien", "Điều khiển Robot tiến thẳng về phía trước", "100", "1.05 - 1.58")
    ]"""

new_d_rows_vi = """    dataset_rows = [
        ("0", "KhoiDong", "Khởi động / Sẵn sàng vận hành hệ thống", "500", "0.96 - 1.81"),
        ("1", "KhongXacDinh", "Tạp âm môi trường / Tiếng nói ngoài luồng", "200", "0.87 - 3.01"),
        ("2", "Lui", "Điều khiển Robot lùi lại phía sau", "500", "0.96 - 1.56"),
        ("3", "RePhai", "Điều khiển Robot quay phải 90 độ", "500", "1.39 - 2.81"),
        ("4", "ReTrai", "Điều khiển Robot quay trái 90 độ", "500", "0.87 - 1.81"),
        ("5", "TamDung", "Dừng khẩn cấp / Tạm dừng chuyển động", "500", "0.87 - 2.15"),
        ("6", "Tien", "Điều khiển Robot tiến thẳng về phía trước", "500", "0.87 - 2.15")
    ]"""
assert old_d_rows_vi in vi_code, "old_d_rows_vi not found"
vi_code = vi_code.replace(old_d_rows_vi, new_d_rows_vi)

# Figure 3 & Benchmark sample description
old_bench_vi = """    if os.path.exists("ADL/results/waveform_rephai_original.png"):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.add_run().add_picture("ADL/results/waveform_rephai_original.png", width=Inches(6.0))
        add_caption(doc, "Hình 3. Mẫu âm thanh chuẩn ('Re Phai 100.wav'): (a) Dạng sóng âm thanh gốc và (b) Sau khi chuẩn hóa 2.0s bằng kỹ thuật Center-Padding.", is_table=False)

    add_body_p(doc, "Nhằm mang lại tính minh bạch khoa học và khả năng truy vết xuyên suốt toàn bộ báo cáo, chúng tôi thiết lập một nghiên cứu điển hình (case study) đồng bộ: theo dõi một file âm thanh chuẩn duy nhất—'Re Phai 100.wav' (lớp RePhai)—đi qua từng bước biến đổi trong toàn bộ quy trình kỹ thuật học sâu.")
    add_body_p(doc, "Như minh họa trên Hình 3(a), file ghi âm gốc có độ dài 1.58s (25.259 mẫu). Nếu áp dụng phương pháp cắt ngắn thô sơ (ví dụ: cắt ở 1.0s), âm tiết thứ hai 'Phải' sẽ bị chém đứt hoàn toàn. Thông qua thuật toán Center-Padding đối xứng (Công thức 1), Hình 3(b) thể hiện kết quả thực thi vật lý: đúng 3.370 mẫu giá trị 0 (0.211s) được chèn vào đầu và 3.371 mẫu giá trị 0 (0.211s) được chèn vào đuôi, tạo nên chuỗi tín hiệu chuẩn 2.0s (32.000 mẫu). Giải pháp này đảm bảo 100% năng lượng giọng nói được giữ nguyên vẹn tại trung tâm mà không hề bị biến dạng thời gian.")"""

new_bench_vi = """    if os.path.exists("ADL/results/waveform_rephai_original.png"):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.add_run().add_picture("ADL/results/waveform_rephai_original.png", width=Inches(6.0))
        add_caption(doc, "Hình 3. Mẫu âm thanh chuẩn ('Re Phải (1).wav'): (a) Dạng sóng âm thanh gốc và (b) Sau khi chuẩn hóa 2.0s bằng kỹ thuật Center-Padding.", is_table=False)

    add_body_p(doc, "Nhằm mang lại tính minh bạch khoa học và khả năng truy vết xuyên suốt toàn bộ báo cáo, chúng tôi thiết lập một nghiên cứu điển hình (case study) đồng bộ: theo dõi một file âm thanh chuẩn duy nhất—'Re Phải (1).wav' (lớp RePhai)—đi qua từng bước biến đổi trong toàn bộ quy trình kỹ thuật học sâu.")
    add_body_p(doc, "Như minh họa trên Hình 3(a), file ghi âm gốc có độ dài 1.64s (26.261 mẫu). Nếu áp dụng phương pháp cắt ngắn thô sơ (ví dụ: cắt ở 1.0s), âm tiết thứ hai 'Phải' sẽ bị chém đứt hoàn toàn. Thông qua thuật toán Center-Padding đối xứng (Công thức 1), Hình 3(b) thể hiện kết quả thực thi vật lý: đúng 2.869 mẫu giá trị 0 (0.179s) được chèn vào đầu và 2.870 mẫu giá trị 0 (0.179s) được chèn vào đuôi, tạo nên chuỗi tín hiệu chuẩn 2.0s (32.000 mẫu). Giải pháp này đảm bảo 100% năng lượng giọng nói được giữ nguyên vẹn tại trung tâm mà không hề bị biến dạng thời gian.")"""
assert old_bench_vi in vi_code, "old_bench_vi not found"
vi_code = vi_code.replace(old_bench_vi, new_bench_vi)

# Figure 4 caption and text
vi_code = vi_code.replace(
    'add_caption(doc, "Hình 4. Minh họa kỹ thuật SpecAugment trên mẫu chuẩn (\'Re Phai 100.wav\'): (a) Mel-Spectrogram gốc và (b) Sau khi áp dụng che tần số & thời gian.", is_table=False)',
    'add_caption(doc, "Hình 4. Minh họa kỹ thuật SpecAugment trên mẫu chuẩn (\'Re Phải (1).wav\'): (a) Mel-Spectrogram gốc và (b) Sau khi áp dụng che tần số & thời gian.", is_table=False)'
)
vi_code = vi_code.replace(
    "Tiếp nối hành trình đồng bộ của file 'Re Phai 100.wav', Hình 4",
    "Tiếp nối hành trình đồng bộ của file 'Re Phải (1).wav', Hình 4"
)

# Section 2.3: dataset_processed.pt size
vi_code = vi_code.replace('(dataset_processed.pt, dung lượng 22.5 MB)', '(dataset_processed.pt, dung lượng 98.5 MB)')

# Table 4 (Optimizer Benchmark)
old_opt_rows_vi = """    opt_rows = [
        ("SGD thuần", "0.01", "0.0", "60.00%", "Hội tụ cực chậm; mắc kẹt tại điểm yên ngựa"),
        ("SGD + Momentum", "0.01", "1e-4", "94.29%", "Hội tụ ổn định; còn dao động nhẹ ở cuối"),
        ("Adam", "0.001", "0.0", "100.00%", "Hội tụ rất nhanh ngay từ các epoch đầu"),
        ("AdamW + Cosine", "0.001", "1e-4", "100.00%", "Hội tụ tối ưu nhất; đường loss giảm mượt mà")
    ]"""

new_opt_rows_vi = """    opt_rows = [
        ("SGD thuần", "0.01", "0.0", "75.94%", "Hội tụ cực chậm; mắc kẹt tại điểm yên ngựa"),
        ("SGD + Momentum", "0.01", "1e-4", "96.25%", "Hội tụ ổn định; còn dao động nhẹ ở cuối"),
        ("Adam", "0.001", "0.0", "100.00%", "Hội tụ rất nhanh ngay từ các epoch đầu"),
        ("AdamW + Cosine", "0.001", "1e-4", "100.00%", "Hội tụ tối ưu nhất; đường loss giảm mượt mà")
    ]"""
assert old_opt_rows_vi in vi_code, "old_opt_rows_vi not found"
vi_code = vi_code.replace(old_opt_rows_vi, new_opt_rows_vi)

# Table 3 (Ablation Study)
old_abl_rows_vi = """    abl_rows = [
        ("M1_BaselineCNN", "1,928,711", "5.51", "0.28", "100.00%", "98.57%", "23.35"),
        ("M2_DropoutCNN", "1,928,711", "5.51", "0.36", "98.57%", "98.57%", "22.73"),
        ("M3_BatchNormCNN", "1,929,159", "5.51", "0.52", "98.57%", "98.57%", "29.23"),
        ("M4_VoiceResNet", "307,687", "14.94", "1.09", "98.57%", "98.57%", "82.05"),
        ("M5_SE_VoiceResNet", "310,503", "14.95", "1.33", "98.57%", "100.00%", "99.09")
    ]"""

new_abl_rows_vi = """    abl_rows = [
        ("M1_BaselineCNN", "1,928,711", "5.51", "2.07", "100.00%", "100.00%", "54.0"),
        ("M2_DropoutCNN", "1,928,711", "5.51", "1.82", "100.00%", "100.00%", "53.8"),
        ("M3_BatchNormCNN", "1,929,159", "5.51", "2.23", "100.00%", "100.00%", "74.0"),
        ("M4_VoiceResNet", "307,687", "14.94", "8.56", "100.00%", "100.00%", "230.1"),
        ("M5_SE_VoiceResNet", "310,503", "14.95", "7.70", "100.00%", "100.00%", "384.1")
    ]"""
assert old_abl_rows_vi in vi_code, "old_abl_rows_vi not found"
vi_code = vi_code.replace(old_abl_rows_vi, new_abl_rows_vi)

# Table 5 (Edge Profiling)
old_edge_rows_vi = """    edge_rows = [
        ("M1_BaselineCNN", "7.72 MB", "~18.4 MB", "0.28 ms", "3,571 mẫu/giây"),
        ("M2_DropoutCNN", "7.72 MB", "~18.4 MB", "0.36 ms", "2,777 mẫu/giây"),
        ("M3_BatchNormCNN", "7.73 MB", "~18.5 MB", "0.52 ms", "1,923 mẫu/giây"),
        ("M4_VoiceResNet", "1.26 MB", "~5.2 MB", "1.09 ms", "917 mẫu/giây"),
        ("M5_SE_VoiceResNet", "1.27 MB", "~5.3 MB", "1.33 ms", "751 mẫu/giây")
    ]"""

new_edge_rows_vi = """    edge_rows = [
        ("M1_BaselineCNN", "7.72 MB", "~18.4 MB", "2.07 ms", "483 mẫu/giây"),
        ("M2_DropoutCNN", "7.72 MB", "~18.4 MB", "1.82 ms", "549 mẫu/giây"),
        ("M3_BatchNormCNN", "7.73 MB", "~18.5 MB", "2.23 ms", "448 mẫu/giây"),
        ("M4_VoiceResNet", "1.26 MB", "~5.2 MB", "8.56 ms", "117 mẫu/giây"),
        ("M5_SE_VoiceResNet", "1.27 MB", "~5.3 MB", "7.70 ms", "130 mẫu/giây")
    ]"""
assert old_edge_rows_vi in vi_code, "old_edge_rows_vi not found"
vi_code = vi_code.replace(old_edge_rows_vi, new_edge_rows_vi)

# Confusion Matrix Section
vi_code = vi_code.replace(
    'tập kiểm thử độc lập (70 mẫu, 10 mẫu cho mỗi lớp).',
    'tập kiểm thử độc lập (320 mẫu: 50 mẫu cho mỗi lớp khẩu lệnh, 20 mẫu cho lớp KhongXacDinh).'
)
vi_code = vi_code.replace(
    'với 10/10 mẫu dự đoán chính xác cho từng lớp: KhoiDong (100%), KhongXacDinh (100%), Lui (100%), RePhai (100%), ReTrai (100%), TamDung (100%), và Tien (100%).',
    'với đúng 320/320 mẫu dự đoán chính xác tuyệt đối: KhoiDong (50/50, 100%), KhongXacDinh (20/20, 100%), Lui (50/50, 100%), RePhai (50/50, 100%), ReTrai (50/50, 100%), TamDung (50/50, 100%), và Tien (50/50, 100%).'
)

# Figure 14 caption and narrative
vi_code = vi_code.replace(
    'add_caption(doc, "Hình 14. Bản đồ nhiệt Grad-CAM giải thích quyết định của mô hình SE_VoiceResNet trên mẫu chuẩn đồng bộ \'Re Phai 100.wav\' (Khẩu lệnh Rẽ Phải).", is_table=False)',
    'add_caption(doc, "Hình 14. Bản đồ nhiệt Grad-CAM giải thích quyết định của mô hình SE_VoiceResNet trên mẫu chuẩn đồng bộ \'Re Phải (1).wav\' (Khẩu lệnh Rẽ Phải).", is_table=False)'
)
vi_code = vi_code.replace(
    "Đối với mẫu chuẩn đồng bộ ('Re Phai 100.wav', Hình 14):",
    "Đối với mẫu chuẩn đồng bộ ('Re Phải (1).wav', Hình 14):"
)
vi_code = vi_code.replace(
    "bold_prefix=\"Mẫu chuẩn đồng bộ ('Re Phai 100.wav'): \"",
    "bold_prefix=\"Mẫu chuẩn đồng bộ ('Re Phải (1).wav'): \""
)

with open("ADL/create_word_report_vi.py", "w", encoding="utf-8") as f:
    f.write(vi_code)
print("[SUCCESS] Updated ADL/create_word_report_vi.py for Data_DLP!")
