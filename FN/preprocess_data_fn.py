import os
import glob
import torch
import numpy as np
import scipy.io.wavfile as wavfile
import torchaudio.transforms as T
from sklearn.model_selection import train_test_split

TARGET_SR = 16000
TARGET_DURATION = 2.0  # seconds
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)  # 32,000 samples
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128

def load_and_preprocess_audio(file_path):
    sr, data = wavfile.read(file_path)
    if sr != TARGET_SR:
        raise ValueError(f"Expected sample rate {TARGET_SR}, got {sr} in {file_path}")

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

    cur_len = len(waveform)
    if cur_len < TARGET_SAMPLES:
        pad_total = TARGET_SAMPLES - cur_len
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        waveform = np.pad(waveform, (pad_left, pad_right), mode="constant", constant_values=0.0)
    else:
        start = (cur_len - TARGET_SAMPLES) // 2
        waveform = waveform[start : start + TARGET_SAMPLES]

    return torch.from_numpy(waveform).unsqueeze(0)

def main():
    data_dir = "Data_FN"
    output_path = "FN/dataset_processed_2800.pt"
    
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith(".")])
    label_map = {name: idx for idx, name in enumerate(class_names)}
    print(f"Discovered {len(class_names)} classes: {label_map}")

    mel_transform = T.MelSpectrogram(
        sample_rate=TARGET_SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )
    amp_to_db = T.AmplitudeToDB(top_db=80.0)

    all_specs = []
    all_labels = []

    print("")
    print("Processing audio files from Data_FN (2,900 samples: 500 RePhai + 400 each for others)...")
    for class_name in class_names:
        class_folder = os.path.join(data_dir, class_name)
        files = sorted(glob.glob(os.path.join(class_folder, "*.wav")))
        label = label_map[class_name]
        print(f" - Class {class_name:15s}: {len(files)} files")

        for f in files:
            wav_tensor = load_and_preprocess_audio(f)
            mel_spec = amp_to_db(mel_transform(wav_tensor))
            all_specs.append(mel_spec)
            all_labels.append(label)

    X = torch.stack(all_specs)
    y = torch.tensor(all_labels, dtype=torch.long)
    print("")
    print(f"Total processed samples: {X.shape[0]}, Tensor shape: {X.shape}, Labels shape: {y.shape}")

    indices = np.arange(len(y))
    # Stratified 80:10:10 split
    train_val_idx, test_idx, train_val_y, test_y = train_test_split(
        indices, y.numpy(), test_size=0.10, stratify=y.numpy(), random_state=42
    )
    train_idx, val_idx, _, _ = train_test_split(
        train_val_idx, train_val_y, test_size=0.1111, stratify=train_val_y, random_state=42
    )

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print("Splits summary (Stratified 80:10:10):")
    print(f" - Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(y)*100:.1f}%)")
    print(f" - Val:   {X_val.shape[0]} samples ({X_val.shape[0]/len(y)*100:.1f}%)")
    print(f" - Test:  {X_test.shape[0]} samples ({X_test.shape[0]/len(y)*100:.1f}%)")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "label_map": label_map,
        "class_names": class_names,
        "mel_shape": list(X_train.shape[1:]),
        "target_sr": TARGET_SR,
        "target_duration": TARGET_DURATION,
    }
    torch.save(payload, output_path)
    print("")
    print(f"[SUCCESS] Processed dataset saved to: {output_path} (Size: {os.path.getsize(output_path)/1024/1024:.2f} MB)")

if __name__ == "__main__":
    main()
