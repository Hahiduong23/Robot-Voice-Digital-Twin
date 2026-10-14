# =========================================================================
# FINAL TERM: Real-time Audio Preprocessing & Inference Pipeline Simulator
# Simulates exact Web Audio API: Sinc Resample + Silence Guard + Peak Norm 0.65 + Center-Pad 2.0s
# =========================================================================
import os
import sys
import glob
import time
import json
import torch
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import torchaudio.transforms as T

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from FN.models import get_model

TARGET_SR = 16000
TARGET_DURATION = 2.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION) # 32,000

class RealtimeAudioProcessor:
    def __init__(self, model_path='FN/results/BC_ResNet_AdamW.pth', device='cpu'):
        self.device = device
        self.model = get_model('BC_ResNet', num_classes=7).to(device)
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.eval()

        self.mel_trans = T.MelSpectrogram(
            sample_rate=TARGET_SR,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            power=2.0
        ).to(device)
        self.amp_to_db = T.AmplitudeToDB(top_db=80.0).to(device)

        self.class_names = ['KhoiDong', 'KhongXacDinh', 'Lui', 'RePhai', 'ReTrai', 'TamDung', 'Tien']
        self.class_names_vi = {
            'KhoiDong': 'Khởi Động',
            'KhongXacDinh': 'Không Xác Định',
            'Lui': 'Lùi Lại',
            'RePhai': 'Rẽ Phải',
            'ReTrai': 'Rẽ Trái',
            'TamDung': 'Tạm Dừng',
            'Tien': 'Tiến Thẳng'
        }

    def process_and_predict(self, raw_audio, sample_rate):
        """
        Input: raw_audio (1D numpy array, float32 or int16)
        Output: dict with prediction result, probabilities, and telemetry
        """
        t0 = time.time()

        # Convert to float32 [-1, 1]
        if raw_audio.dtype == np.int16:
            audio = raw_audio.astype(np.float32) / 32768.0
        else:
            audio = raw_audio.astype(np.float32)

        # 1. Hardware Sinc Resampling to 16,000 Hz if needed
        if sample_rate != TARGET_SR:
            num_target_samples = int(round(len(audio) * TARGET_SR / sample_rate))
            audio_16k = signal.resample(audio, num_target_samples)
        else:
            audio_16k = audio

        # 2. Measure energy
        peak_before = float(np.max(np.abs(audio_16k))) if len(audio_16k) > 0 else 0.0
        rms_before = float(np.sqrt(np.mean(audio_16k ** 2))) if len(audio_16k) > 0 else 0.0

        # 3. SILENCE GUARD: If audio is pure silence / room noise, bypass AI!
        if peak_before < 0.025 or rms_before < 0.003:
            return {
                'is_silence': True,
                'predicted_class': 'ImLang',
                'predicted_class_vi': 'Im Lặng (Không có tiếng nói)',
                'confidence': 1.0,
                'peak_before': peak_before,
                'rms_before': rms_before,
                'latency_ms': (time.time() - t0) * 1000
            }

        # 4. Adaptive Peak Normalization to 0.65 (Matching Data_MT mean peak 0.630)
        target_peak = 0.65
        gain = min(target_peak / max(peak_before, 1e-5), 25.0)
        normalized = audio_16k * gain
        final_peak = peak_before * gain

        # 5. Exact Center-Padding into 32,000 samples (2.0s)
        centered = np.zeros(TARGET_SAMPLES, dtype=np.float32)
        cur_len = len(normalized)
        if cur_len < TARGET_SAMPLES:
            pad_left = (TARGET_SAMPLES - cur_len) // 2
            centered[pad_left:pad_left + cur_len] = normalized
        else:
            start = (cur_len - TARGET_SAMPLES) // 2
            centered = normalized[start:start + TARGET_SAMPLES]

        # 6. Extract Log-Mel Spectrogram
        wav_t = torch.from_numpy(centered).unsqueeze(0).to(self.device)
        spec = self.amp_to_db(self.mel_trans(wav_t)).unsqueeze(0) # [1, 1, 128, 63]

        # 7. SOTA BC-ResNet Inference
        t_infer = time.time()
        with torch.no_grad():
            logits = self.model(spec)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        latency_ai = (time.time() - t_infer) * 1000
        latency_total = (time.time() - t0) * 1000

        pred_idx = int(np.argmax(probs))
        pred_class = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])

        return {
            'is_silence': False,
            'predicted_class': pred_class,
            'predicted_class_vi': self.class_names_vi[pred_class],
            'confidence': confidence,
            'probabilities': {cls: float(probs[i]) for i, cls in enumerate(self.class_names)},
            'peak_before': peak_before,
            'final_peak': final_peak,
            'latency_ai_ms': latency_ai,
            'latency_total_ms': latency_total
        }

def run_benchmark_test():
    print('=========================================================================')
    print('  KIỂM THỬ REAL-TIME SIMULATOR: SINC RESAMPLING + SILENCE GUARD + NORM')
    print('=========================================================================')

    proc = RealtimeAudioProcessor()

    # Test 1: Silence
    print('\n[TEST 1] Kiểm tra Silence Guard trên âm thanh im lặng (Zeros):')
    silence = np.zeros(32000, dtype=np.float32)
    res_silence = proc.process_and_predict(silence, 16000)
    print(f' - Kết quả: {res_silence["predicted_class_vi"]} | Peak: {res_silence["peak_before"]:.4f} | Bỏ qua AI thành công: {res_silence["is_silence"]}')

    # Test 2: Commands from Data_MT
    print('\n[TEST 2] Kiểm tra nhận diện trên 7 khẩu lệnh thực tế (Data_MT):')
    for cls in proc.class_names:
        files = sorted(glob.glob(f'Data_MT/{cls}/*.wav'))
        if not files: continue
        sr, data = wavfile.read(files[0])
        res = proc.process_and_predict(data, sr)
        icon = '✅' if res['predicted_class'] == cls else '❌'
        print(f' {icon} Lớp gốc: {cls:12s} ➜ Dự đoán: {res["predicted_class_vi"]:20s} ({res["confidence"]*100:5.1f}%) | Độ trễ AI: {res["latency_ai_ms"]:.2f}ms')

if __name__ == '__main__':
    run_benchmark_test()
