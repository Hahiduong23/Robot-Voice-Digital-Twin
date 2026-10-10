# =========================================================================
# RNNoise Denoising Module for 16kHz Audio Pipeline
# Wraps librnnoise C engine with multi-rate resampling support (16kHz <-> 48kHz)
# =========================================================================
import numpy as np
import torch
import torchaudio
import pyrnnoise.rnnoise as r

class RNNoiseDenoiser:
    """
    RNNoise Neural Denoiser:
    - Input: 16kHz mono audio (numpy array or torch Tensor)
    - Output: Denoised 16kHz mono audio of matching length
    - Internal processing: 48kHz Bark-scale GRU network (10ms frames)
    """
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.target_sr = r.SAMPLE_RATE  # 48000
        self.frame_size = r.FRAME_SIZE  # 480 samples = 10ms
        
        # Polyphase Resamplers (accurate & fast)
        self.resample_up = torchaudio.transforms.Resample(self.sample_rate, self.target_sr)
        self.resample_down = torchaudio.transforms.Resample(self.target_sr, self.sample_rate)

    def denoise_waveform(self, audio):
        """
        Denoise full audio waveform (numpy 1D array or torch Tensor).
        Returns:
            cleaned_audio: numpy 1D array (float32, normalized [-1.0, 1.0])
            avg_speech_prob: float (voice activity score 0.0 - 1.0)
        """
        is_tensor = isinstance(audio, torch.Tensor)
        if is_tensor:
            audio_np = audio.detach().cpu().squeeze().numpy()
        else:
            audio_np = np.asarray(audio, dtype=np.float32)

        orig_len = len(audio_np)
        if orig_len == 0:
            return np.zeros(0, dtype=np.float32), 0.0

        # Ensure normalized float32
        if audio_np.dtype == np.int16:
            audio_np = audio_np.astype(np.float32) / 32768.0
        elif np.abs(audio_np).max() > 1.0:
            audio_np = audio_np / (np.abs(audio_np).max() + 1e-8)

        # 1. Resample to 48kHz
        audio_tensor = torch.from_numpy(audio_np).float().unsqueeze(0)
        audio_48k = self.resample_up(audio_tensor).squeeze(0).numpy()

        # 2. Process frames through C RNNoise engine
        state = r.create()
        n_frames = len(audio_48k) // self.frame_size
        cleaned_frames = []
        speech_probs = []

        audio_48k_int16 = (np.clip(audio_48k, -1.0, 1.0) * 32767).astype(np.int16)

        for i in range(n_frames):
            frame = audio_48k_int16[i * self.frame_size : (i + 1) * self.frame_size]
            clean_frame, prob = r.process_mono_frame(state, frame)
            cleaned_frames.append(clean_frame)
            speech_probs.append(prob)

        # Handle remaining samples if any
        rem = len(audio_48k) % self.frame_size
        if rem > 0:
            frame = np.pad(audio_48k_int16[n_frames * self.frame_size :], (0, self.frame_size - rem))
            clean_frame, prob = r.process_mono_frame(state, frame)
            cleaned_frames.append(clean_frame[:rem])
            speech_probs.append(prob)

        r.destroy(state)

        # 3. Concatenate and resample back to 16kHz
        if cleaned_frames:
            cleaned_48k = np.concatenate(cleaned_frames).astype(np.float32) / 32768.0
            cleaned_48k_tensor = torch.from_numpy(cleaned_48k).float().unsqueeze(0)
            cleaned_16k_tensor = self.resample_down(cleaned_48k_tensor).squeeze(0)
            cleaned_16k = cleaned_16k_tensor.numpy()
        else:
            cleaned_16k = np.zeros(orig_len, dtype=np.float32)

        # Align length
        if len(cleaned_16k) > orig_len:
            cleaned_16k = cleaned_16k[:orig_len]
        elif len(cleaned_16k) < orig_len:
            cleaned_16k = np.pad(cleaned_16k, (0, orig_len - len(cleaned_16k)))

        avg_speech_prob = float(np.mean(speech_probs)) if speech_probs else 0.0
        return cleaned_16k, avg_speech_prob

if __name__ == '__main__':
    denoiser = RNNoiseDenoiser()
    test_sig = np.random.randn(32000).astype(np.float32) * 0.1
    clean, prob = denoiser.denoise_waveform(test_sig)
    print(f'[SUCCESS] RNNoiseDenoiser operational! Input: {len(test_sig)}, Output: {len(clean)}, Prob: {prob:.4f}')
