import os
import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def count_parameters(model):
    """Returns total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_flops(model, input_size=(1, 1, 128, 63), device="cpu"):
    """
    Estimates MACs/FLOPs by tracing Conv2d and Linear layers forward passes.
    """
    flops = 0
    hooks = []

    def conv_hook(module, input, output):
        nonlocal flops
        batch_size, out_c, out_h, out_w = output.shape
        in_c = module.in_channels
        k_h, k_w = module.kernel_size
        groups = module.groups
        macs_per_item = out_c * (in_c // groups) * k_h * k_w * out_h * out_w
        flops += macs_per_item * 2  # 1 MAC = 2 FLOPs

    def linear_hook(module, input, output):
        nonlocal flops
        flops += module.in_features * module.out_features * 2

    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    model.eval()
    dummy = torch.randn(*input_size).to(device)
    with torch.no_grad():
        model(dummy)

    for h in hooks:
        h.remove()

    return flops

def measure_latency(model, input_size=(1, 1, 128, 63), num_warmup=20, num_runs=100, device="cpu"):
    """
    Measures average inference latency in milliseconds (ms) per sample.
    """
    model.eval()
    model.to(device)
    dummy = torch.randn(*input_size).to(device)

    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy)

        start = time.perf_counter()
        for _ in range(num_runs):
            _ = model(dummy)
        end = time.perf_counter()

    avg_latency_ms = ((end - start) / num_runs) * 1000.0
    return avg_latency_ms

def plot_learning_curves(history, save_path, title="Learning Curves"):
    """
    Plots and saves Training & Validation Accuracy and Loss curves with (a) and (b) subtitles.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) Accuracy
    axes[0].plot(history["train_acc"], label="Train Acc", linewidth=2.2, color="#1f77b4")
    axes[0].plot(history["val_acc"], label="Val Acc", linewidth=2.2, linestyle="--", color="#ff7f0e")
    axes[0].set_title(f"(a) Training & Validation Accuracy ({title})", fontsize=12, fontweight="bold", pad=10)
    axes[0].set_xlabel("Epoch", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=10, loc="lower right", framealpha=0.9)

    # (b) Loss
    axes[1].plot(history["train_loss"], label="Train Loss", linewidth=2.2, color="#2ca02c")
    axes[1].plot(history["val_loss"], label="Val Loss", linewidth=2.2, linestyle="--", color="#d62728")
    axes[1].set_title(f"(b) Training & Validation Loss ({title})", fontsize=12, fontweight="bold", pad=10)
    axes[1].set_xlabel("Epoch", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Cross-Entropy Loss", fontsize=11, fontweight="bold")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=10, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path, title="Confusion Matrix"):
    """
    Computes and plots Confusion Matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8.5, 6.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                cbar=True, annot_kws={"size": 11})
    plt.title(f"{title}", fontsize=12.5, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11, fontweight="bold")
    plt.ylabel("True Class", fontsize=11, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, target_class

def plot_gradcam_overlay(mel_spec, cam_map, title, save_path):
    """
    Visualizes raw Mel-Spectrogram alongside Grad-CAM heatmap with (a) and (b) subtitles.
    """
    if isinstance(mel_spec, torch.Tensor):
        mel_spec = mel_spec.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) Mel-Spectrogram
    im0 = axes[0].imshow(mel_spec, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_title("(a) Original Mel-Spectrogram (dB)", fontsize=11.5, fontweight="bold", pad=10)
    axes[0].set_xlabel("Time Frames (1.5s - 2.0s)", fontsize=10.5, fontweight="bold")
    axes[0].set_ylabel("Mel Frequency Bins (128)", fontsize=10.5, fontweight="bold")
    fig.colorbar(im0, ax=axes[0])

    # (b) Grad-CAM Overlay
    im1 = axes[1].imshow(mel_spec, aspect="auto", origin="lower", cmap="gray", alpha=0.6)
    im2 = axes[1].imshow(cam_map, aspect="auto", origin="lower", cmap="jet", alpha=0.5)
    axes[1].set_title(f"(b) Grad-CAM Heatmap ({title})", fontsize=11.5, fontweight="bold", pad=10)
    axes[1].set_xlabel("Time Frames", fontsize=10.5, fontweight="bold")
    axes[1].set_ylabel("Mel Frequency Bins", fontsize=10.5, fontweight="bold")
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()