# =========================================================================
# Quantitative & Visual Benchmark for RNNoise Front-End Denoiser with M5
# =========================================================================
import os
import glob
import time
import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import torch
import torchaudio.transforms as T
import sys

sys.path.append('MT')
from models import get_model
from denoise_rnnoise import RNNoiseDenoiser

TARGET_SR = 16000
TARGET_DURATION = 2.0
TARGET_SAMPLES = 32000
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128

def add_synthetic_environmental_noise(clean_wav, snr_db=10.0):
    """
    Adds realistic environmental robot noise:
    - Motor rumble / low-frequency harmonics (120Hz, 300Hz, 800Hz)
    - Wideband fan / background hiss
    """
    p_signal = np.mean(clean_wav ** 2)
    if p_signal <= 1e-10:
        return clean_wav.copy()

    # 1. Base Gaussian noise
    gaussian_noise = np.random.randn(len(clean_wav))

    # 2. Motor harmonic hums
    t = np.linspace(0, len(clean_wav)/TARGET_SR, len(clean_wav))
    motor_hum = (
        0.5 * np.sin(2 * np.pi * 120 * t) +
        0.3 * np.sin(2 * np.pi * 300 * t) +
        0.2 * np.sin(2 * np.pi * 850 * t)
    )

    combined_noise = gaussian_noise + motor_hum
    p_noise = np.mean(combined_noise ** 2)

    # Scale noise according to desired SNR (dB)
    target_p_noise = p_signal / (10.0 ** (snr_db / 10.0))
    scale = np.sqrt(target_p_noise / (p_noise + 1e-10))
    noisy_wav = clean_wav + combined_noise * scale

    return np.clip(noisy_wav, -1.0, 1.0).astype(np.float32)

def compute_mel_tensor(waveform, mel_transform, amp_to_db):
    """
    Converts 1D float32 waveform (length 32,000) to [1, 1, 128, 63] tensor.
    """
    cur_len = len(waveform)
    if cur_len < TARGET_SAMPLES:
        pad = TARGET_SAMPLES - cur_len
        waveform = np.pad(waveform, (pad // 2, pad - pad // 2), mode='constant')
    else:
        start = (cur_len - TARGET_SAMPLES) // 2
        waveform = waveform[start : start + TARGET_SAMPLES]

    wav_tensor = torch.from_numpy(waveform).float().unsqueeze(0)
    mel_db = amp_to_db(mel_transform(wav_tensor)) # [1, 128, 63]
    return mel_db.unsqueeze(0) # [1, 1, 128, 63]

def main():
    print('=========================================================================')
    print('          BENCHMARK ĐÁNH GIÁ ĐỊNH LƯỢNG BỘ LỌC NHIỄU RNNOISE VỚI SOTA BC-RESNET')
    print('=========================================================================')

    # 1. Khởi tạo Denoiser và Mô hình M5
    denoiser = RNNoiseDenoiser()
    mel_transform = T.MelSpectrogram(
        sample_rate=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0
    )
    amp_to_db = T.AmplitudeToDB(top_db=80.0)

    checkpoint_path = 'MT/results/BC_ResNet_AdamW.pth'
    if not os.path.exists(checkpoint_path):
        print(f'[ERROR] Không tìm thấy checkpoint {checkpoint_path}')
        return

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = get_model('BC_ResNet', num_classes=7)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    class_names = ['KhoiDong', 'KhongXacDinh', 'Lui', 'RePhai', 'ReTrai', 'TamDung', 'Tien']
    label_map = {name: idx for idx, name in enumerate(class_names)}

    # 2. Chọn các file đại diện để test
    test_classes = ['Tien', 'RePhai', 'Lui', 'TamDung', 'KhoiDong']
    results_table = []
    plot_data = None

    print(f'Đang tiến hành stress-test trên 5 lớp lệnh với mức nhiễu SNR = 10 dB và 5 dB...')

    for class_name in test_classes:
        files = sorted(glob.glob(f'Data_MT/{class_name}/*.wav'))
        if not files:
            continue
        test_file = files[0]
        sr, data = wavfile.read(test_file)
        if data.dtype == np.int16:
            clean_audio = data.astype(np.float32) / 32768.0
        else:
            clean_audio = data.astype(np.float32)

        true_label = class_name
        true_idx = label_map[true_label]

        for snr in [10.0, 5.0]:
            # Tạo âm thanh bẩn
            noisy_audio = add_synthetic_environmental_noise(clean_audio, snr_db=snr)

            # Lọc sạch bằng RNNoise và đo thời gian
            t_start = time.time()
            denoised_audio, speech_prob = denoiser.denoise_waveform(noisy_audio)
            t_denoise_ms = (time.time() - t_start) * 1000

            # Suy luận AI qua M5
            with torch.no_grad():
                mel_clean = compute_mel_tensor(clean_audio, mel_transform, amp_to_db)
                mel_noisy = compute_mel_tensor(noisy_audio, mel_transform, amp_to_db)
                mel_denoised = compute_mel_tensor(denoised_audio, mel_transform, amp_to_db)

                probs_clean = torch.softmax(model(mel_clean), dim=1)[0].numpy()
                probs_noisy = torch.softmax(model(mel_noisy), dim=1)[0].numpy()
                probs_denoised = torch.softmax(model(mel_denoised), dim=1)[0].numpy()

            pred_clean = class_names[np.argmax(probs_clean)]
            conf_clean = probs_clean[true_idx] * 100

            pred_noisy = class_names[np.argmax(probs_noisy)]
            conf_noisy = probs_noisy[true_idx] * 100

            pred_denoised = class_names[np.argmax(probs_denoised)]
            conf_denoised = probs_denoised[true_idx] * 100

            results_table.append({
                'class': class_name,
                'snr': snr,
                'conf_clean': conf_clean,
                'pred_noisy': pred_noisy,
                'conf_noisy': conf_noisy,
                'pred_denoised': pred_denoised,
                'conf_denoised': conf_denoised,
                'latency_ms': t_denoise_ms
            })

            # Lưu mẫu lớp 'Tien' ở SNR 5dB để vẽ biểu đồ trực quan hóa
            if class_name == 'Tien' and snr == 5.0:
                plot_data = {
                    'clean': clean_audio,
                    'noisy': noisy_audio,
                    'denoised': denoised_audio,
                    'mel_clean': mel_clean[0, 0].numpy(),
                    'mel_noisy': mel_noisy[0, 0].numpy(),
                    'mel_denoised': mel_denoised[0, 0].numpy(),
                    'probs_clean': probs_clean,
                    'probs_noisy': probs_noisy,
                    'probs_denoised': probs_denoised,
                    'classes': class_names,
                    'true_class': class_name
                }

    # 3. In bảng kết quả định lượng
    print('-' * 103)
    print(f'{"Khẩu lệnh":<12} | {"Mức SNR":<8} | {"Clean %":<9} | {"Nhiễu (Chưa lọc)":<18} | {"RNNoise (Đã lọc)":<18} | {"Độ trễ lọc":<10}')
    print('-------------------------------------------------------------------------------------------------------')
    for row in results_table:
        noisy_str = f"{row['pred_noisy']} ({row['conf_noisy']:.1f}%)"
        denoised_str = f"{row['pred_denoised']} ({row['conf_denoised']:.1f}%)"
        print(f"{row['class']:<12} | {row['snr']:<6.0f}dB | {row['conf_clean']:<7.1f}% | {noisy_str:<18} | {denoised_str:<18} | {row['latency_ms']:<6.1f} ms")
    print('-------------------------------------------------------------------------------------------------------')

    # 4. Vẽ biểu đồ so sánh chi tiết
    if plot_data is not None:
        fig, axes = plt.subplots(3, 3, figsize=(16, 10))
        plt.subplots_adjust(hspace=0.4, wspace=0.3)

        titles = ['(A) Âm thanh Gốc Sạch (Clean)', '(B) Nhiễu nặng SNR=5dB (Noisy)', '(C) Sau lọc RNNoise (Denoised)']
        audios = [plot_data['clean'], plot_data['noisy'], plot_data['denoised']]
        mels = [plot_data['mel_clean'], plot_data['mel_noisy'], plot_data['mel_denoised']]
        probs_list = [plot_data['probs_clean'], plot_data['probs_noisy'], plot_data['probs_denoised']]
        colors = ['#2ca02c', '#d62728', '#1f77b4']

        # Hàng 1: Dạng sóng Waveform
        for i in range(3):
            ax = axes[0, i]
            time_axis = np.linspace(0, len(audios[i])/TARGET_SR, len(audios[i]))
            ax.plot(time_axis, audios[i], color=colors[i], alpha=0.85, linewidth=0.8)
            ax.set_title(f'Waveform: {titles[i]}', fontsize=11, fontweight='bold')
            ax.set_ylim(-1.1, 1.1)
            ax.set_xlabel('Thời gian (giây)')
            ax.set_ylabel('Biên độ')
            ax.grid(True, linestyle='--', alpha=0.5)

        # Hàng 2: Phổ Log-Mel Spectrogram
        for i in range(3):
            ax = axes[1, i]
            im = ax.imshow(mels[i], origin='lower', aspect='auto', cmap='inferno')
            ax.set_title(f'Log-Mel Spectrogram: {titles[i]}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Khung thời gian (Frames)')
            ax.set_ylabel('Dải tần số Mel')

        # Hàng 3: Phân phối Softmax
        for i in range(3):
            ax = axes[2, i]
            bars = ax.bar(plot_data['classes'], probs_list[i] * 100, color='#34495e')
            true_idx = plot_data['classes'].index(plot_data['true_class'])
            bars[true_idx].set_color(colors[i])
            ax.set_title(f'Softmax Conf: {plot_data["true_class"]}={probs_list[i][true_idx]*100:.1f}%', fontsize=11, fontweight='bold')
            ax.set_ylim(0, 105)
            ax.set_ylabel('Độ tự tin (%)')
            ax.set_xticklabels(plot_data['classes'], rotation=40, ha='right', fontsize=9)
            ax.grid(axis='y', linestyle='--', alpha=0.5)

        out_img = 'MT/results/rnnoise_benchmark_evaluation.png'
        plt.savefig(out_img, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'[SUCCESS] Đã lưu biểu đồ đối soát trực quan tại: {out_img}')

if __name__ == '__main__':
    main()
