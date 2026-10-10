import re

# =============================================================================
# 1. REVERT create_word_report.py (ENGLISH)
# =============================================================================
with open("ADL/create_word_report.py", "r", encoding="utf-8") as f:
    en_code = f.read()

# Dataset intro
en_code = en_code.replace(
    'The experimental dataset comprises 3,200 high-fidelity audio recordings stored in the Data_DLP/ directory',
    'The experimental dataset comprises 700 high-fidelity audio recordings stored in the Data_DL/ directory'
)

# Table 2 rows
old_d_rows_en = """    dataset_rows = [
        ("0", "KhoiDong", "Start / Initialize Robot System", "500", "0.96 - 1.81"),
        ("1", "KhongXacDinh", "Unknown / Ambient Noise / Non-Command Audio", "200", "0.87 - 3.01"),
        ("2", "Lui", "Drive Backward", "500", "0.96 - 1.56"),
        ("3", "RePhai", "Turn Right 90 Degrees", "500", "1.39 - 2.81"),
        ("4", "ReTrai", "Turn Left 90 Degrees", "500", "0.87 - 1.81"),
        ("5", "TamDung", "Emergency Stop / Pause Motion", "500", "0.87 - 2.15"),
        ("6", "Tien", "Drive Forward", "500", "0.87 - 2.15")
    ]"""

new_d_rows_en = """    dataset_rows = [
        ("0", "KhoiDong", "Start / Initialize Robot System", "100", "1.12 - 1.84"),
        ("1", "KhongXacDinh", "Unknown / Ambient Noise / Non-Command Audio", "100", "1.04 - 1.96"),
        ("2", "Lui", "Drive Backward", "100", "1.08 - 1.62"),
        ("3", "RePhai", "Turn Right 90 Degrees", "100", "1.15 - 1.78"),
        ("4", "ReTrai", "Turn Left 90 Degrees", "100", "1.18 - 1.82"),
        ("5", "TamDung", "Emergency Stop / Pause Motion", "100", "1.10 - 1.65"),
        ("6", "Tien", "Drive Forward", "100", "1.05 - 1.58")
    ]"""
assert old_d_rows_en in en_code, "old_d_rows_en not found"
en_code = en_code.replace(old_d_rows_en, new_d_rows_en)

# Benchmark sample description
old_bench_en = """    if os.path.exists("ADL/results/waveform_rephai_original.png"):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.add_run().add_picture("ADL/results/waveform_rephai_original.png", width=Inches(6.0))
        add_caption(doc, "Figure 3. Benchmark Sample ('Re Phải (1).wav'): (a) Raw Audio Waveform vs. (b) Standardized 2.0s Center-Padding.", is_table=False)

    add_body_p(doc, "To ensure absolute methodological transparency and traceability throughout this report, we establish an end-to-end synchronized case study following a single benchmark audio file—'Re Phải (1).wav' (Class: RePhai)—across every transformation stage of our deep learning engineering pipeline.")
    add_body_p(doc, "As demonstrated in Figure 3(a), the raw recording spans 1.64s (26,261 samples). Under a naive truncation approach (e.g., slicing at 1.0s), the crucial second syllable 'Phải' would be completely severed. Through our symmetric Center-Padding formulation (Equation 1), Figure 3(b) displays the exact physical realization: exactly 2,869 zero samples (0.179s) are prepended to the left, and 2,870 zero samples (0.179s) are appended to the right, yielding a standardized 2.0s sequence (32,000 samples). This guarantees that 100% of the speech energy remains perfectly centered and aligned without temporal distortion.")"""

new_bench_en = """    if os.path.exists("ADL/results/waveform_rephai_original.png"):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.add_run().add_picture("ADL/results/waveform_rephai_original.png", width=Inches(6.0))
        add_caption(doc, "Figure 3. Benchmark Sample ('Re Phai 100.wav'): (a) Raw Audio Waveform vs. (b) Standardized 2.0s Center-Padding.", is_table=False)

    add_body_p(doc, "To ensure absolute methodological transparency and traceability throughout this report, we establish an end-to-end synchronized case study following a single benchmark audio file—'Re Phai 100.wav' (Class: RePhai)—across every transformation stage of our deep learning engineering pipeline.")
    add_body_p(doc, "As demonstrated in Figure 3(a), the raw recording spans 1.58s (25,259 samples). Under a naive truncation approach (e.g., slicing at 1.0s), the crucial second syllable 'Phải' would be completely severed. Through our symmetric Center-Padding formulation (Equation 1), Figure 3(b) displays the exact physical realization: exactly 3,370 zero samples (0.211s) are prepended to the left, and 3,371 zero samples (0.211s) are appended to the right, yielding a standardized 2.0s sequence (32,000 samples). This guarantees that 100% of the speech energy remains perfectly centered and aligned without temporal distortion.")"""
assert old_bench_en in en_code, "old_bench_en not found"
en_code = en_code.replace(old_bench_en, new_bench_en)

# Figure 4 caption and text
en_code = en_code.replace(
    'add_caption(doc, "Figure 4. SpecAugment Demonstration on Benchmark Sample (\'Re Phải (1).wav\'): (a) Original vs (b) Frequency & Time Masked Mel-Spectrogram.", is_table=False)',
    'add_caption(doc, "Figure 4. SpecAugment Demonstration on Benchmark Sample (\'Re Phai 100.wav\'): (a) Original vs (b) Frequency & Time Masked Mel-Spectrogram.", is_table=False)'
)
en_code = en_code.replace(
    "Continuing the synchronized trajectory of 'Re Phải (1).wav', Figure 4",
    "Continuing the synchronized trajectory of 'Re Phai 100.wav', Figure 4"
)

# Section 2.3: dataset_processed.pt size
en_code = en_code.replace('(dataset_processed.pt, 98.5 MB)', '(dataset_processed.pt, 21.5 MB)')

# Table 4 (Optimizer Benchmark)
old_opt_rows_en = """    opt_rows = [
        ("SGD (Pure)", "0.01", "0.0", "75.94%", "Extremely slow; trapped in saddle points"),
        ("SGD + Momentum", "0.01", "1e-4", "96.25%", "Steady convergence; minor late fluctuations"),
        ("Adam", "0.001", "0.0", "100.00%", "Very fast initial convergence; stable"),
        ("AdamW + Cosine", "0.001", "1e-4", "100.00%", "Optimal convergence; smoothest loss trajectory")
    ]"""

new_opt_rows_en = """    opt_rows = [
        ("SGD (Pure)", "0.01", "0.0", "57.14%", "Extremely slow; trapped in saddle points"),
        ("SGD + Momentum", "0.01", "1e-4", "92.86%", "Steady convergence; minor late fluctuations"),
        ("Adam", "0.001", "0.0", "100.00%", "Very fast initial convergence; stable"),
        ("AdamW + Cosine", "0.001", "1e-4", "100.00%", "Optimal convergence; smoothest loss trajectory")
    ]"""
assert old_opt_rows_en in en_code, "old_opt_rows_en not found"
en_code = en_code.replace(old_opt_rows_en, new_opt_rows_en)

# Table 3 (Ablation Study)
old_abl_rows_en = """    abl_rows = [
        ("M1_BaselineCNN", "1,928,711", "5.51", "2.07", "100.00%", "100.00%", "54.0"),
        ("M2_DropoutCNN", "1,928,711", "5.51", "1.82", "100.00%", "100.00%", "53.8"),
        ("M3_BatchNormCNN", "1,929,159", "5.51", "2.23", "100.00%", "100.00%", "74.0"),
        ("M4_VoiceResNet", "307,687", "14.94", "8.56", "100.00%", "100.00%", "230.1"),
        ("M5_SE_VoiceResNet", "310,503", "14.95", "7.70", "100.00%", "100.00%", "384.1")
    ]"""

new_abl_rows_en = """    abl_rows = [
        ("M1_BaselineCNN", "1,928,711", "5.51", "1.45", "100.00%", "98.57%", "23.7"),
        ("M2_DropoutCNN", "1,928,711", "5.51", "2.78", "100.00%", "98.57%", "22.6"),
        ("M3_BatchNormCNN", "1,929,159", "5.51", "1.73", "98.57%", "98.57%", "29.5"),
        ("M4_VoiceResNet", "307,687", "14.94", "5.89", "100.00%", "100.00%", "82.4"),
        ("M5_SE_VoiceResNet", "310,503", "14.95", "6.01", "98.57%", "100.00%", "98.1")
    ]"""
assert old_abl_rows_en in en_code, "old_abl_rows_en not found"
en_code = en_code.replace(old_abl_rows_en, new_abl_rows_en)

# Table 5 (Edge Profiling)
old_edge_rows_en = """    edge_rows = [
        ("M1_BaselineCNN", "7.72 MB", "~18.4 MB", "2.07 ms", "483 samples/s"),
        ("M2_DropoutCNN", "7.72 MB", "~18.4 MB", "1.82 ms", "549 samples/s"),
        ("M3_BatchNormCNN", "7.73 MB", "~18.5 MB", "2.23 ms", "448 samples/s"),
        ("M4_VoiceResNet", "1.26 MB", "~5.2 MB", "8.56 ms", "117 samples/s"),
        ("M5_SE_VoiceResNet", "1.27 MB", "~5.3 MB", "7.70 ms", "130 samples/s")
    ]"""

new_edge_rows_en = """    edge_rows = [
        ("M1_BaselineCNN", "7.72 MB", "~18.4 MB", "1.45 ms", "690 samples/s"),
        ("M2_DropoutCNN", "7.72 MB", "~18.4 MB", "2.78 ms", "360 samples/s"),
        ("M3_BatchNormCNN", "7.73 MB", "~18.5 MB", "1.73 ms", "578 samples/s"),
        ("M4_VoiceResNet", "1.26 MB", "~5.2 MB", "5.89 ms", "170 samples/s"),
        ("M5_SE_VoiceResNet", "1.27 MB", "~5.3 MB", "6.01 ms", "166 samples/s")
    ]"""
assert old_edge_rows_en in en_code, "old_edge_rows_en not found"
en_code = en_code.replace(old_edge_rows_en, new_edge_rows_en)

# Confusion Matrix Section
en_code = en_code.replace(
    'independent test set (320 samples: 50 samples per command class, 20 samples for KhongXacDinh).',
    'independent test set (70 samples, 10 samples per class).'
)
en_code = en_code.replace(
    'with exactly 320/320 correct classifications across every single class: KhoiDong (50/50, 100%), KhongXacDinh (20/20, 100%), Lui (50/50, 100%), RePhai (50/50, 100%), ReTrai (50/50, 100%), TamDung (50/50, 100%), and Tien (50/50, 100%).',
    'with exactly 10/10 correct classifications across every single class: KhoiDong (100%), KhongXacDinh (100%), Lui (100%), RePhai (100%), ReTrai (100%), TamDung (100%), and Tien (100%).'
)

# Figure 14 caption and narrative
en_code = en_code.replace(
    'add_caption(doc, "Figure 14. Grad-CAM Heatmap Visualization for Benchmark Sample \'Re Phải (1).wav\' (Class: RePhai).", is_table=False)',
    'add_caption(doc, "Figure 14. Grad-CAM Heatmap Visualization for Benchmark Sample \'Re Phai 100.wav\' (Class: RePhai).", is_table=False)'
)
en_code = en_code.replace(
    "For the synchronized benchmark sample ('Re Phải (1).wav', Figure 14),",
    "For the synchronized benchmark sample ('Re Phai 100.wav', Figure 14),"
)
en_code = en_code.replace(
    "bold_prefix=\"Synchronized Benchmark ('Re Phải (1).wav'): \"",
    "bold_prefix=\"Synchronized Benchmark ('Re Phai 100.wav'): \""
)

with open("ADL/create_word_report.py", "w", encoding="utf-8") as f:
    f.write(en_code)
print("[SUCCESS] Reverted ADL/create_word_report.py to Data_DL!")


# =============================================================================
# 2. REVERT create_word_report_vi.py (VIETNAMESE)
# =============================================================================
with open("ADL/create_word_report_vi.py", "r", encoding="utf-8") as f:
    vi_code = f.read()

# Dataset intro
vi_code = vi_code.replace(
    'Tập dữ liệu thực nghiệm bao gồm 3.200 file ghi âm chất lượng cao lưu trữ trong thư mục Data_DLP/',
    'Tập dữ liệu thực nghiệm bao gồm 700 file ghi âm chất lượng cao lưu trữ trong thư mục Data_DL/'
)

# Table 2 rows
old_d_rows_vi = """    dataset_rows = [
        ("0", "KhoiDong", "Khởi động / Sẵn sàng vận hành hệ thống", "500", "0.96 - 1.81"),
        ("1", "KhongXacDinh", "Tạp âm môi trường / Tiếng nói ngoài luồng", "200", "0.87 - 3.01"),
        ("2", "Lui", "Điều khiển Robot lùi lại phía sau", "500", "0.96 - 1.56"),
        ("3", "RePhai", "Điều khiển Robot quay phải 90 độ", "500", "1.39 - 2.81"),
        ("4", "ReTrai", "Điều khiển Robot quay trái 90 độ", "500", "0.87 - 1.81"),
        ("5", "TamDung", "Dừng khẩn cấp / Tạm dừng chuyển động", "500", "0.87 - 2.15"),
        ("6", "Tien", "Điều khiển Robot tiến thẳng về phía trước", "500", "0.87 - 2.15")
    ]"""

new_d_rows_vi = """    dataset_rows = [
        ("0", "KhoiDong", "Khởi động / Sẵn sàng vận hành hệ thống", "100", "1.12 - 1.84"),
        ("1", "KhongXacDinh", "Tạp âm môi trường / Tiếng nói ngoài luồng", "100", "1.04 - 1.96"),
        ("2", "Lui", "Điều khiển Robot lùi lại phía sau", "100", "1.08 - 1.62"),
        ("3", "RePhai", "Điều khiển Robot quay phải 90 độ", "100", "1.15 - 1.78"),
        ("4", "ReTrai", "Điều khiển Robot quay trái 90 độ", "100", "1.18 - 1.82"),
        ("5", "TamDung", "Dừng khẩn cấp / Tạm dừng chuyển động", "100", "1.10 - 1.65"),
        ("6", "Tien", "Điều khiển Robot tiến thẳng về phía trước", "100", "1.05 - 1.58")
    ]"""
assert old_d_rows_vi in vi_code, "old_d_rows_vi not found"
vi_code = vi_code.replace(old_d_rows_vi, new_d_rows_vi)

# Benchmark sample description
old_bench_vi = """    if os.path.exists("ADL/results/waveform_rephai_original.png"):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.add_run().add_picture("ADL/results/waveform_rephai_original.png", width=Inches(6.0))
        add_caption(doc, "Hình 3. Mẫu âm thanh chuẩn ('Re Phải (1).wav'): (a) Dạng sóng âm thanh gốc và (b) Sau khi chuẩn hóa 2.0s bằng kỹ thuật Center-Padding.", is_table=False)

    add_body_p(doc, "Nhằm mang lại tính minh bạch khoa học và khả năng truy vết xuyên suốt toàn bộ báo cáo, chúng tôi thiết lập một nghiên cứu điển hình (case study) đồng bộ: theo dõi một file âm thanh chuẩn duy nhất—'Re Phải (1).wav' (lớp RePhai)—đi qua từng bước biến đổi trong toàn bộ quy trình kỹ thuật học sâu.")
    add_body_p(doc, "Như minh họa trên Hình 3(a), file ghi âm gốc có độ dài 1.64s (26.261 mẫu). Nếu áp dụng phương pháp cắt ngắn thô sơ (ví dụ: cắt ở 1.0s), âm tiết thứ hai 'Phải' sẽ bị chém đứt hoàn toàn. Thông qua thuật toán Center-Padding đối xứng (Công thức 1), Hình 3(b) thể hiện kết quả thực thi vật lý: đúng 2.869 mẫu giá trị 0 (0.179s) được chèn vào đầu và 2.870 mẫu giá trị 0 (0.179s) được chèn vào đuôi, tạo nên chuỗi tín hiệu chuẩn 2.0s (32.000 mẫu). Giải pháp này đảm bảo 100% năng lượng giọng nói được giữ nguyên vẹn tại trung tâm mà không hề bị biến dạng thời gian.")"""

new_bench_vi = """    if os.path.exists("ADL/results/waveform_rephai_original.png"):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.add_run().add_picture("ADL/results/waveform_rephai_original.png", width=Inches(6.0))
        add_caption(doc, "Hình 3. Mẫu âm thanh chuẩn ('Re Phai 100.wav'): (a) Dạng sóng âm thanh gốc và (b) Sau khi chuẩn hóa 2.0s bằng kỹ thuật Center-Padding.", is_table=False)

    add_body_p(doc, "Nhằm mang lại tính minh bạch khoa học và khả năng truy vết xuyên suốt toàn bộ báo cáo, chúng tôi thiết lập một nghiên cứu điển hình (case study) đồng bộ: theo dõi một file âm thanh chuẩn duy nhất—'Re Phai 100.wav' (lớp RePhai)—đi qua từng bước biến đổi trong toàn bộ quy trình kỹ thuật học sâu.")
    add_body_p(doc, "Như minh họa trên Hình 3(a), file ghi âm gốc có độ dài 1.58s (25.259 mẫu). Nếu áp dụng phương pháp cắt ngắn thô sơ (ví dụ: cắt ở 1.0s), âm tiết thứ hai 'Phải' sẽ bị chém đứt hoàn toàn. Thông qua thuật toán Center-Padding đối xứng (Công thức 1), Hình 3(b) thể hiện kết quả thực thi vật lý: đúng 3.370 mẫu giá trị 0 (0.211s) được chèn vào đầu và 3.371 mẫu giá trị 0 (0.211s) được chèn vào đuôi, tạo nên chuỗi tín hiệu chuẩn 2.0s (32.000 mẫu). Giải pháp này đảm bảo 100% năng lượng giọng nói được giữ nguyên vẹn tại trung tâm mà không hề bị biến dạng thời gian.")"""
assert old_bench_vi in vi_code, "old_bench_vi not found"
vi_code = vi_code.replace(old_bench_vi, new_bench_vi)

# Figure 4 caption and text
vi_code = vi_code.replace(
    'add_caption(doc, "Hình 4. Minh họa kỹ thuật SpecAugment trên mẫu chuẩn (\'Re Phải (1).wav\'): (a) Mel-Spectrogram gốc và (b) Sau khi áp dụng che tần số & thời gian.", is_table=False)',
    'add_caption(doc, "Hình 4. Minh họa kỹ thuật SpecAugment trên mẫu chuẩn (\'Re Phai 100.wav\'): (a) Mel-Spectrogram gốc và (b) Sau khi áp dụng che tần số & thời gian.", is_table=False)'
)
vi_code = vi_code.replace(
    "Tiếp nối hành trình đồng bộ của file 'Re Phải (1).wav', Hình 4",
    "Tiếp nối hành trình đồng bộ của file 'Re Phai 100.wav', Hình 4"
)

# Section 2.3: dataset_processed.pt size
vi_code = vi_code.replace('(dataset_processed.pt, dung lượng 98.5 MB)', '(dataset_processed.pt, dung lượng 21.5 MB)')

# Table 4 (Optimizer Benchmark)
old_opt_rows_vi = """    opt_rows = [
        ("SGD thuần", "0.01", "0.0", "75.94%", "Hội tụ cực chậm; mắc kẹt tại điểm yên ngựa"),
        ("SGD + Momentum", "0.01", "1e-4", "96.25%", "Hội tụ ổn định; còn dao động nhẹ ở cuối"),
        ("Adam", "0.001", "0.0", "100.00%", "Hội tụ rất nhanh ngay từ các epoch đầu"),
        ("AdamW + Cosine", "0.001", "1e-4", "100.00%", "Hội tụ tối ưu nhất; đường loss giảm mượt mà")
    ]"""

new_opt_rows_vi = """    opt_rows = [
        ("SGD thuần", "0.01", "0.0", "57.14%", "Hội tụ cực chậm; mắc kẹt tại điểm yên ngựa"),
        ("SGD + Momentum", "0.01", "1e-4", "92.86%", "Hội tụ ổn định; còn dao động nhẹ ở cuối"),
        ("Adam", "0.001", "0.0", "100.00%", "Hội tụ rất nhanh ngay từ các epoch đầu"),
        ("AdamW + Cosine", "0.001", "1e-4", "100.00%", "Hội tụ tối ưu nhất; đường loss giảm mượt mà")
    ]"""
assert old_opt_rows_vi in vi_code, "old_opt_rows_vi not found"
vi_code = vi_code.replace(old_opt_rows_vi, new_opt_rows_vi)

# Table 3 (Ablation Study)
old_abl_rows_vi = """    abl_rows = [
        ("M1_BaselineCNN", "1,928,711", "5.51", "2.07", "100.00%", "100.00%", "54.0"),
        ("M2_DropoutCNN", "1,928,711", "5.51", "1.82", "100.00%", "100.00%", "53.8"),
        ("M3_BatchNormCNN", "1,929,159", "5.51", "2.23", "100.00%", "100.00%", "74.0"),
        ("M4_VoiceResNet", "307,687", "14.94", "8.56", "100.00%", "100.00%", "230.1"),
        ("M5_SE_VoiceResNet", "310,503", "14.95", "7.70", "100.00%", "100.00%", "384.1")
    ]"""

new_abl_rows_vi = """    abl_rows = [
        ("M1_BaselineCNN", "1,928,711", "5.51", "1.45", "100.00%", "98.57%", "23.7"),
        ("M2_DropoutCNN", "1,928,711", "5.51", "2.78", "100.00%", "98.57%", "22.6"),
        ("M3_BatchNormCNN", "1,929,159", "5.51", "1.73", "98.57%", "98.57%", "29.5"),
        ("M4_VoiceResNet", "307,687", "14.94", "5.89", "100.00%", "100.00%", "82.4"),
        ("M5_SE_VoiceResNet", "310,503", "14.95", "6.01", "98.57%", "100.00%", "98.1")
    ]"""
assert old_abl_rows_vi in vi_code, "old_abl_rows_vi not found"
vi_code = vi_code.replace(old_abl_rows_vi, new_abl_rows_vi)

# Table 5 (Edge Profiling)
old_edge_rows_vi = """    edge_rows = [
        ("M1_BaselineCNN", "7.72 MB", "~18.4 MB", "2.07 ms", "483 mẫu/giây"),
        ("M2_DropoutCNN", "7.72 MB", "~18.4 MB", "1.82 ms", "549 mẫu/giây"),
        ("M3_BatchNormCNN", "7.73 MB", "~18.5 MB", "2.23 ms", "448 mẫu/giây"),
        ("M4_VoiceResNet", "1.26 MB", "~5.2 MB", "8.56 ms", "117 mẫu/giây"),
        ("M5_SE_VoiceResNet", "1.27 MB", "~5.3 MB", "7.70 ms", "130 mẫu/giây")
    ]"""

new_edge_rows_vi = """    edge_rows = [
        ("M1_BaselineCNN", "7.72 MB", "~18.4 MB", "1.45 ms", "690 mẫu/giây"),
        ("M2_DropoutCNN", "7.72 MB", "~18.4 MB", "2.78 ms", "360 mẫu/giây"),
        ("M3_BatchNormCNN", "7.73 MB", "~18.5 MB", "1.73 ms", "578 mẫu/giây"),
        ("M4_VoiceResNet", "1.26 MB", "~5.2 MB", "5.89 ms", "170 mẫu/giây"),
        ("M5_SE_VoiceResNet", "1.27 MB", "~5.3 MB", "6.01 ms", "166 mẫu/giây")
    ]"""
assert old_edge_rows_vi in vi_code, "old_edge_rows_vi not found"
vi_code = vi_code.replace(old_edge_rows_vi, new_edge_rows_vi)

# Confusion Matrix Section
vi_code = vi_code.replace(
    'tập kiểm thử độc lập (320 mẫu: 50 mẫu cho mỗi lớp khẩu lệnh, 20 mẫu cho lớp KhongXacDinh).',
    'tập kiểm thử độc lập (70 mẫu, 10 mẫu cho mỗi lớp).'
)
vi_code = vi_code.replace(
    'với đúng 320/320 mẫu dự đoán chính xác tuyệt đối: KhoiDong (50/50, 100%), KhongXacDinh (20/20, 100%), Lui (50/50, 100%), RePhai (50/50, 100%), ReTrai (50/50, 100%), TamDung (50/50, 100%), và Tien (50/50, 100%).',
    'với 10/10 mẫu dự đoán chính xác cho từng lớp: KhoiDong (100%), KhongXacDinh (100%), Lui (100%), RePhai (100%), ReTrai (100%), TamDung (100%), và Tien (100%).'
)

# Figure 14 caption and narrative
vi_code = vi_code.replace(
    'add_caption(doc, "Hình 14. Bản đồ nhiệt Grad-CAM giải thích quyết định của mô hình SE_VoiceResNet trên mẫu chuẩn đồng bộ \'Re Phải (1).wav\' (Khẩu lệnh Rẽ Phải).", is_table=False)',
    'add_caption(doc, "Hình 14. Bản đồ nhiệt Grad-CAM giải thích quyết định của mô hình SE_VoiceResNet trên mẫu chuẩn đồng bộ \'Re Phai 100.wav\' (Khẩu lệnh Rẽ Phải).", is_table=False)'
)
vi_code = vi_code.replace(
    "Đối với mẫu chuẩn đồng bộ ('Re Phải (1).wav', Hình 14):",
    "Đối với mẫu chuẩn đồng bộ ('Re Phai 100.wav', Hình 14):"
)
vi_code = vi_code.replace(
    "bold_prefix=\"Mẫu chuẩn đồng bộ ('Re Phải (1).wav'): \"",
    "bold_prefix=\"Mẫu chuẩn đồng bộ ('Re Phai 100.wav'): \""
)

with open("ADL/create_word_report_vi.py", "w", encoding="utf-8") as f:
    f.write(vi_code)
print("[SUCCESS] Reverted ADL/create_word_report_vi.py to Data_DL!")
