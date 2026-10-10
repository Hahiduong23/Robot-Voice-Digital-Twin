import os
import glob
import torch
import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import torchaudio.transforms as T
from models import get_model
from utils import GradCAM

TARGET_SR = 16000
TARGET_DURATION = 2.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)  # 32,000 samples
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128

def load_audio_file(file_path):
    sr, data = wavfile.read(file_path)
    if sr != TARGET_SR:
        raise ValueError(f"Expected {TARGET_SR} Hz, got {sr} in {file_path}")
    if data.dtype == np.int16:
        waveform = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        waveform = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        waveform = (data.astype(np.float32) - 128.0) / 128.0
    else:
        waveform = data.astype(np.float32)

    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)
    return waveform

def center_pad_crop(waveform, target_samples=TARGET_SAMPLES):
    cur_len = len(waveform)
    if cur_len < target_samples:
        pad_total = target_samples - cur_len
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        padded = np.pad(waveform, (pad_left, pad_right), mode="constant", constant_values=0.0)
        return padded, pad_left, pad_right
    else:
        start = (cur_len - target_samples) // 2
        cropped = waveform[start : start + target_samples]
        return cropped, 0, 0

def extract_mel_spectrogram(waveform_padded):
    mel_transform = T.MelSpectrogram(
        sample_rate=TARGET_SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )
    amp_to_db = T.AmplitudeToDB(top_db=80.0)
    wav_tensor = torch.from_numpy(waveform_padded).unsqueeze(0)  # (1, 32000)
    mel_spec = amp_to_db(mel_transform(wav_tensor))  # (1, 128, 63)
    return mel_spec

def main():
    os.makedirs("ADL/results", exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. PLOT 1: Amplitude over Time Feature across all 7 classes (Data_1400)
    # ---------------------------------------------------------
    print("Generating Plot 1: Amplitude over Time Feature across all 7 classes (Data_1400)...")
    class_order = ["KhoiDong", "KhongXacDinh", "Lui", "RePhai", "ReTrai", "TamDung", "Tien"]
    class_display_names = {
        "KhoiDong": "KhoiDong (Start / Initialize)",
        "KhongXacDinh": "KhongXacDinh (Unknown / Ambient Noise)",
        "Lui": "Lui (Backward)",
        "RePhai": "RePhai (Turn Right)",
        "ReTrai": "ReTrai (Turn Left)",
        "TamDung": "TamDung (Emergency Stop)",
        "Tien": "Tien (Forward)"
    }
    class_sample_files = {
        "KhoiDong": "Data_1400/KhoiDong/KhoiDong_001.wav",
        "KhongXacDinh": "Data_1400/KhongXacDinh/KhongXacDinh_001.wav",
        "Lui": "Data_1400/Lui/Lui_001.wav",
        "RePhai": "Data_1400/RePhai/RePhai_001.wav",  # Synchronized benchmark sample
        "ReTrai": "Data_1400/ReTrai/ReTrai_001.wav",
        "TamDung": "Data_1400/TamDung/TamDung_001.wav",
        "Tien": "Data_1400/Tien/Tien_001.wav"
    }

    fig, axes = plt.subplots(7, 1, figsize=(13, 14), sharex=False, sharey=True)
    plt.subplots_adjust(hspace=0.45)
    
    colors = ["#1f77b4", "#7f7f7f", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e", "#17becf"]
    letters = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)"]

    for idx, c_name in enumerate(class_order):
        ax = axes[idx]
        wav_path = class_sample_files[c_name]
        wav = load_audio_file(wav_path)
        dur = len(wav) / TARGET_SR
        t = np.linspace(0, dur, len(wav))
        
        ax.plot(t, wav, color=colors[idx], linewidth=0.9, alpha=0.9)
        ax.axhline(0, color="#888888", linestyle="--", linewidth=0.6, alpha=0.7)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlim(0, 2.05)
        ax.set_ylabel("Amplitude", fontsize=9.5, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)
        
        filename = os.path.basename(wav_path)
        ax.set_title(
            f"{letters[idx]} Class '{class_display_names[c_name]}' | File: '{filename}' | Duration: {dur:.2f}s ({len(wav):,} samples at 16 kHz)",
            fontsize=10.5, fontweight="bold", loc="left", pad=5, color="#102C57" if c_name != "RePhai" else "#b30000"
        )
        if idx == 6:
            ax.set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")

    plt.suptitle("Amplitude over Time Acoustic Features across 7 Robot Command Classes (Data_1400, 16 kHz)", fontsize=13.5, fontweight="bold", y=0.995, color="#102C57")
    plt.tight_layout()
    plt.savefig("ADL/results/waveform_all_classes.png", dpi=250, bbox_inches="tight")
    plt.close()
    print(" -> Saved: ADL/results/waveform_all_classes.png")

    # ---------------------------------------------------------
    # 2. PLOT 2: Original audio file of RePhai (Raw vs Center-Padded)
    # ---------------------------------------------------------
    print("Generating Plot 2: Benchmark Sample 'RePhai_001.wav' Raw vs Center-Padded...")
    benchmark_file = "Data_1400/RePhai/RePhai_001.wav"
    raw_wav = load_audio_file(benchmark_file)
    padded_wav, pad_l, pad_r = center_pad_crop(raw_wav, TARGET_SAMPLES)
    
    dur_raw = len(raw_wav) / TARGET_SR
    t_raw = np.linspace(0, dur_raw, len(raw_wav))
    t_padded = np.linspace(0, TARGET_DURATION, TARGET_SAMPLES)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6.8), sharey=True)
    plt.subplots_adjust(hspace=0.4)

    # (a) Raw Waveform
    ax1.plot(t_raw, raw_wav, color="#102C57", linewidth=1.0)
    ax1.axhline(0, color="#888888", linestyle="--", linewidth=0.7)
    ax1.set_ylim(-1.05, 1.05)
    ax1.set_xlim(0, 2.05)
    ax1.set_title(f"(a) Original Audio Waveform: 'RePhai_001.wav' (Class: RePhai | Raw Duration: {dur_raw:.2f}s | {len(raw_wav):,} samples)",
                  fontsize=11.5, fontweight="bold", pad=8, color="#102C57")
    ax1.set_xlabel("Time (seconds)", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Normalized Amplitude", fontsize=10.5, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.annotate(
        f"Raw End: {dur_raw:.2f}s",
        (dur_raw, 0),
        xytext=(dur_raw + 0.08, 0.45),
        textcoords="data",
        fontsize=9.5, fontweight="bold", color="#b30000",
        arrowprops=dict(arrowstyle="->", color="#b30000", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff0f0", edgecolor="#b30000", lw=1)
    )

    # (b) Center-Padded Waveform
    ax2.plot(t_padded, padded_wav, color="#d62728", linewidth=1.0, label="Centered Vocal Signal")
    ax2.axhline(0, color="#888888", linestyle="--", linewidth=0.7)
    pad_l_dur = pad_l / TARGET_SR
    pad_r_dur = pad_r / TARGET_SR
    ax2.axvspan(0, pad_l_dur, color="#e0e0e0", alpha=0.6, label=f"Left Pad: {pad_l_dur:.3f}s ({pad_l:,} samples)")
    ax2.axvspan(TARGET_DURATION - pad_r_dur, TARGET_DURATION, color="#e0e0e0", alpha=0.6, label=f"Right Pad: {pad_r_dur:.3f}s ({pad_r:,} samples)")
    ax2.set_ylim(-1.05, 1.05)
    ax2.set_xlim(0, 2.05)
    ax2.set_title(f"(b) Standardized 2.0s Audio Waveform via Center-Padding (Eq. 1) (Uniform: 32,000 samples at 16 kHz)",
                  fontsize=11.5, fontweight="bold", pad=8, color="#102C57")
    ax2.set_xlabel("Time (seconds)", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Normalized Amplitude", fontsize=10.5, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=9.5, framealpha=0.9)

    plt.tight_layout()
    plt.savefig("ADL/results/waveform_rephai_original.png", dpi=250, bbox_inches="tight")
    plt.close()
    print(" -> Saved: ADL/results/waveform_rephai_original.png")

    # ---------------------------------------------------------
    # 3. PLOT 3: SpecAugment Demonstration on the SAME RePhai file
    # ---------------------------------------------------------
    print("Generating Plot 3: SpecAugment Demonstration on 'RePhai_001.wav'...")
    mel_spec_tensor = extract_mel_spectrogram(padded_wav)  # (1, 128, 63)
    mel_orig = mel_spec_tensor.squeeze(0).numpy()          # (128, 63)

    torch.manual_seed(42)
    freq_masker = T.FrequencyMasking(freq_mask_param=16)
    time_masker = T.TimeMasking(time_mask_param=12)
    
    mel_masked_tensor = time_masker(freq_masker(mel_spec_tensor.clone()))
    mel_masked_tensor = time_masker(freq_masker(mel_masked_tensor))
    mel_masked = mel_masked_tensor.squeeze(0).numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    vmin = min(mel_orig.min(), mel_masked.min())
    vmax = max(mel_orig.max(), mel_masked.max())

    im1 = ax1.imshow(mel_orig, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    ax1.set_title("(a) Original Mel-Spectrogram (dB) - 'RePhai_001.wav'", fontsize=11.5, fontweight="bold", pad=10, color="#102C57")
    ax1.set_xlabel("Time Frames (63 frames ≈ 2.0s)", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Mel Frequency Bins (128 bins)", fontsize=10.5, fontweight="bold")
    cbar1 = fig.colorbar(im1, ax=ax1)
    cbar1.set_label("Magnitude (dB)", fontsize=9.5, fontweight="bold")

    im2 = ax2.imshow(mel_masked, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    ax2.set_title("(b) SpecAugment: Frequency & Time Masked Spectrogram", fontsize=11.5, fontweight="bold", pad=10, color="#102C57")
    ax2.set_xlabel("Time Frames (63 frames ≈ 2.0s)", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Mel Frequency Bins (128 bins)", fontsize=10.5, fontweight="bold")
    cbar2 = fig.colorbar(im2, ax=ax2)
    cbar2.set_label("Magnitude (dB)", fontsize=9.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig("ADL/results/specaugment_demo.png", dpi=250, bbox_inches="tight")
    plt.close()
    print(" -> Saved: ADL/results/specaugment_demo.png (Synchronized with RePhai_001.wav)")

    # ---------------------------------------------------------
    # 4, 5, 6: Grad-CAM Heatmaps (RePhai, KhongXacDinh, Tien)
    # ---------------------------------------------------------
    m5_ckpt_path = "ADL/results/M5_SE_VoiceResNet_AdamW.pth"
    if os.path.exists(m5_ckpt_path):
        print("Loading M5_SE_VoiceResNet model for Grad-CAM generation...")
        m5_model = get_model("M5_SE_VoiceResNet", num_classes=7)
        ckpt = torch.load(m5_ckpt_path, map_location="cpu", weights_only=False)
        m5_model.load_state_dict(ckpt["state_dict"])
        m5_model.eval()

        gradcam = GradCAM(m5_model, target_layer=m5_model.layer3.conv2)

        gradcam_targets = [
            {
                "file": "Data_1400/RePhai/RePhai_001.wav",
                "class_name": "RePhai",
                "target_idx": 3,
                "save_path": "ADL/results/gradcam_sample_rephai.png",
                "title_vi": "Khẩu lệnh 'Rẽ Phải' (Mẫu chuẩn đồng bộ)",
                "title_en": "Command 'RePhai' (Synchronized Benchmark Sample)"
            },
            {
                "file": "Data_1400/KhongXacDinh/KhongXacDinh_001.wav",
                "class_name": "KhongXacDinh",
                "target_idx": 1,
                "save_path": "ADL/results/gradcam_sample_khongxacdinh.png",
                "title_vi": "Lớp 'KhongXacDinh' (Tạp âm môi trường / Tiếng ồn)",
                "title_en": "Class 'KhongXacDinh' (Ambient Noise / Unknown)"
            },
            {
                "file": "Data_1400/Tien/Tien_001.wav",
                "class_name": "Tien",
                "target_idx": 6,
                "save_path": "ADL/results/gradcam_sample_tien.png",
                "title_vi": "Khẩu lệnh 'Tiến' (Khẩu lệnh đơn âm)",
                "title_en": "Command 'Tien' (Monosyllabic Forward Command)"
            }
        ]

        for item in gradcam_targets:
            print(f"Generating Grad-CAM for {item['class_name']} ({os.path.basename(item['file'])})...")
            w = load_audio_file(item["file"])
            w_pad, _, _ = center_pad_crop(w, TARGET_SAMPLES)
            mel_t = extract_mel_spectrogram(w_pad)
            input_t = mel_t.unsqueeze(0)

            cam_map, pred_class = gradcam.generate(input_t, target_class=item["target_idx"])
            pred_name = class_order[pred_class]
            mel_np = mel_t.squeeze(0).numpy()

            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            
            im0 = axes[0].imshow(mel_np, aspect="auto", origin="lower", cmap="viridis")
            axes[0].set_title(f"(a) 2.0s Mel-Spectrogram (dB) - '{os.path.basename(item['file'])}'", fontsize=11.5, fontweight="bold", pad=10, color="#102C57")
            axes[0].set_xlabel("Time Frames (63 frames ≈ 2.0s)", fontsize=10.5, fontweight="bold")
            axes[0].set_ylabel("Mel Frequency Bins (128 bins)", fontsize=10.5, fontweight="bold")
            fig.colorbar(im0, ax=axes[0])

            im1 = axes[1].imshow(mel_np, aspect="auto", origin="lower", cmap="gray", alpha=0.6)
            im2 = axes[1].imshow(cam_map, aspect="auto", origin="lower", cmap="jet", alpha=0.55)
            axes[1].set_title(f"(b) Grad-CAM Heatmap (True: {item['class_name']} | Pred: {pred_name})", fontsize=11.5, fontweight="bold", pad=10, color="#102C57")
            axes[1].set_xlabel("Time Frames (Acoustic Phoneme Attention)", fontsize=10.5, fontweight="bold")
            axes[1].set_ylabel("Mel Frequency Bins", fontsize=10.5, fontweight="bold")
            fig.colorbar(im2, ax=axes[1])

            plt.tight_layout()
            plt.savefig(item["save_path"], dpi=250, bbox_inches="tight")
            plt.close()
            print(f" -> Saved: {item['save_path']}")
    else:
        print(f"Warning: {m5_ckpt_path} not found yet, skipping Grad-CAM until training completes.")

    print("\n[SUCCESS] Audio visualizations script updated for Data_1400!")

if __name__ == "__main__":
    main()
