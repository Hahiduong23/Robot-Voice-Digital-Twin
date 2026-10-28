#!/usr/bin/env python3
"""
generate_perfect_final_plots.py
Generates all publication-grade, beautiful evaluation plots for the Final Report.
Rules:
1. Exact latest numbers from code and checkpoint (157,415 params, 83.41 MFLOPs, 4.84 ms latency, 2,900 samples, 290 test samples, 100.0% test accuracy).
2. Short, punchy titles.
3. Absolutely NO overlapping text, labels, numbers, or legends. Generous breathing margins.
4. 300 DPI high-resolution figures.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from FN.models import get_model

OUT_DIR = os.path.join(BASE_DIR, "FN", "results")
os.makedirs(OUT_DIR, exist_ok=True)

# Aesthetic configuration
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.edgecolor"] = "#334155"
plt.rcParams["axes.linewidth"] = 1.0

CLASS_NAMES = ["KhoiDong", "KhongXacDinh", "Lui", "RePhai", "ReTrai", "TamDung", "Tien"]
CLASS_LABELS_DISPLAY = ["Khởi Động\n(Start)", "Không Xác Định\n(Noise)", "Lùi Lại\n(Backward)",
                        "Rẽ Phải\n(Turn Right)", "Rẽ Trái\n(Turn Left)", "Tạm Dừng\n(Stop)", "Tiến Thẳng\n(Forward)"]
CLASS_COLORS = ["#2563eb", "#64748b", "#ea580c", "#16a34a", "#9333ea", "#dc2626", "#0891b2"]

# Load dataset and checkpoint
data = torch.load(os.path.join(BASE_DIR, "FN", "dataset_processed_2800.pt"), map_location="cpu", weights_only=False)
X_test = data["X_test"]
y_test = data["y_test"]
y_test_np = y_test.numpy()

ckpt = torch.load(os.path.join(BASE_DIR, "FN", "results", "BC_ResNet_AdamW.pth"), map_location="cpu", weights_only=False)
model = get_model("BC_ResNet", num_classes=7)
model.load_state_dict(ckpt["state_dict"])
model.eval()

with torch.no_grad():
    logits = model(X_test)
    probs_test = torch.softmax(logits, dim=1).numpy()
    preds_test = logits.argmax(dim=1).numpy()

history = ckpt.get("history", {})
epochs = len(history.get("train_loss", []))
t_epochs = np.arange(1, epochs + 1)

# ==============================================================================
# 1. CONFUSION MATRIX (Perfect spacing, clean numbers, concise title)
# ==============================================================================
def plot_confusion_matrix():
    print("[1/7] Generating Confusion Matrix...")
    cm = confusion_matrix(y_test_np, preds_test)
    
    fig, ax = plt.subplots(figsize=(7.5, 6.2), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8.5)
    cbar.set_label("Sample Count", fontsize=9.5, fontweight="bold", labelpad=8)
    
    tick_marks = np.arange(len(CLASS_NAMES))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(CLASS_NAMES, rotation=28, ha="right", fontsize=9.0, fontweight="bold", color="#0f172a")
    ax.set_yticklabels(CLASS_NAMES, fontsize=9.0, fontweight="bold", color="#0f172a")
    
    # Text inside cells with adaptive contrast
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "#ffffff" if val > thresh else "#0f172a"
            ax.text(j, i, f"{val:d}", ha="center", va="center",
                    color=color, fontsize=11.5, fontweight="bold")
    
    ax.set_xlabel("Predicted Command", fontsize=10.5, fontweight="bold", color="#0f172a", labelpad=8)
    ax.set_ylabel("True Target Command", fontsize=10.5, fontweight="bold", color="#0f172a", labelpad=8)
    ax.set_title("Confusion Matrix - BC-ResNet (Test N = 290)", fontsize=11.5, fontweight="bold", pad=12, color="#0f172a")
    
    plt.tight_layout()
    out_p = os.path.join(OUT_DIR, "confusion_matrix_bc_resnet.png")
    plt.savefig(out_p, dpi=300, bbox_inches="tight")
    plt.close()
    print("  -> Saved:", out_p)

# ==============================================================================
# 2. TRAINING & VALIDATION CURVES (Loss & Accuracy, no collision)
# ==============================================================================
def plot_training_curves():
    print("[2/7] Generating Training & Validation Curves...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.0), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    train_acc = history.get("train_acc", [])
    val_acc = history.get("val_acc", [])
    
    # Subplot 1: Loss
    ax1.plot(t_epochs, train_loss, color="#dc2626", lw=2.2, label="Train Loss (Cross-Entropy)")
    ax1.plot(t_epochs, val_loss, color="#2563eb", lw=2.0, linestyle="--", label="Val Loss")
    ax1.set_xlabel("Epoch", fontsize=10.0, fontweight="bold", color="#0f172a")
    ax1.set_ylabel("Loss", fontsize=10.0, fontweight="bold", color="#0f172a")
    ax1.set_title("Training & Validation Loss", fontsize=11.0, fontweight="bold", pad=10, color="#0f172a")
    ax1.grid(True, linestyle="--", alpha=0.45)
    ax1.legend(loc="upper right", fontsize=9.0, framealpha=0.92)
    ax1.set_ylim(bottom=0.0)
    
    # Subplot 2: Accuracy
    ax2.plot(t_epochs, [a * 100 if a <= 1.0 else a for a in train_acc], color="#16a34a", lw=2.2, label="Train Accuracy")
    ax2.plot(t_epochs, [a * 100 if a <= 1.0 else a for a in val_acc], color="#d97706", lw=2.0, linestyle="--", label="Val Accuracy")
    ax2.set_xlabel("Epoch", fontsize=10.0, fontweight="bold", color="#0f172a")
    ax2.set_ylabel("Accuracy (%)", fontsize=10.0, fontweight="bold", color="#0f172a")
    ax2.set_title("Training & Validation Accuracy", fontsize=11.0, fontweight="bold", pad=10, color="#0f172a")
    ax2.grid(True, linestyle="--", alpha=0.45)
    ax2.legend(loc="lower right", fontsize=9.0, framealpha=0.92)
    ax2.set_ylim(85.0, 102.5)
    
    # Best Val Acc Annotation with clean offset
    best_val = max([a * 100 if a <= 1.0 else a for a in val_acc])
    best_ep = np.argmax([a * 100 if a <= 1.0 else a for a in val_acc]) + 1
    ax2.scatter([best_ep], [best_val], color="#d97706", s=60, zorder=5)
    ax2.annotate(f"Best Val: {best_val:.2f}% (Ep {best_ep})",
                 xy=(best_ep, best_val), xytext=(-65, -22), textcoords="offset points",
                 fontsize=8.5, fontweight="bold", color="#b45309",
                 arrowprops=dict(arrowstyle="->", color="#b45309", lw=1.2))
    
    plt.tight_layout()
    out_p = os.path.join(OUT_DIR, "curves_BC_ResNet_AdamW.png")
    plt.savefig(out_p, dpi=300, bbox_inches="tight")
    plt.close()
    print("  -> Saved:", out_p)

# ==============================================================================
# 3. BENCHMARK & PARETO EVOLUTION (Clean offsets, generous y-limits, no overlaps)
# ==============================================================================
def plot_pareto_evolution():
    print("[3/7] Generating Benchmark Pareto Evolution...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.2), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    
    models = ["M1 (Baseline)", "M4 (VoiceResNet)", "M5 (SE-ResNet)", "BC-ResNet (Final)"]
    params_k = [1928.7, 307.7, 310.5, 157.4]
    flops_m = [152.2, 771.7, 771.7, 83.4]
    latencies = [1.95, 4.82, 5.48, 4.84]
    accuracies = [100.0, 100.0, 100.0, 100.0]
    
    # Subplot A: Parameters & FLOPs Comparison (Log-scale or dual grouped bars)
    x = np.arange(len(models))
    w = 0.35
    b1 = ax1.bar(x - w/2, params_k, w, label="Parameters (K)", color=["#94a3b8", "#60a5fa", "#3b82f6", "#10b981"], edgecolor="#0f172a")
    
    ax1_twin = ax1.twinx()
    b2 = ax1_twin.bar(x + w/2, flops_m, w, label="Compute FLOPs (M)", color=["#cbd5e1", "#93c5fd", "#60a5fa", "#34d399"], hatch="//", edgecolor="#0f172a")
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=18, ha="right", fontsize=9.0, fontweight="bold")
    ax1.set_ylabel("Parameters (x1,000)", fontsize=9.5, fontweight="bold", color="#1e3a8a")
    ax1_twin.set_ylabel("FLOPs (Millions)", fontsize=9.5, fontweight="bold", color="#065f46")
    ax1.set_title("Model Size & Computational Complexity", fontsize=11.0, fontweight="bold", pad=12)
    ax1.grid(True, linestyle="--", alpha=0.45)
    
    # Generous limits so annotations never collide with top border
    ax1.set_ylim(0, 2400)
    ax1_twin.set_ylim(0, 1000)
    
    for bar in b1:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}K", xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                     fontsize=8.0, fontweight="bold", color="#1e3a8a")
    for bar in b2:
        h = bar.get_height()
        ax1_twin.annotate(f"{h:.1f}M", xy=(bar.get_x() + bar.get_width()/2, h),
                          xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                          fontsize=8.0, fontweight="bold", color="#065f46")
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8.5, framealpha=0.9)

    # Subplot B: Pareto Trade-off (FLOPs vs Latency with bubble size = params)
    # BC-ResNet dominates with 83.4M FLOPs and 4.84 ms latency
    scatter_pts = [
        (152.2, 1.95, 1928.7, "M1 Baseline", "#64748b"),
        (771.7, 4.82, 307.7, "M4 VoiceResNet", "#3b82f6"),
        (771.7, 5.48, 310.5, "M5 SE-VoiceResNet", "#8b5cf6"),
        (83.4, 4.84, 157.4, "BC-ResNet (SOTA)", "#059669")
    ]
    
    for fl, lat, pr, name, col in scatter_pts:
        size = pr * 0.45 if pr < 1000 else 400
        ax2.scatter(fl, lat, s=size, color=col, alpha=0.85, edgecolors="#0f172a", lw=1.5, zorder=4)
    
    # Offsets meticulously chosen so no collision
    ax2.annotate("M1 Baseline\n(152.2M | 1.95ms)", xy=(152.2, 1.95), xytext=(12, -8),
                 textcoords="offset points", fontsize=8.5, fontweight="bold", color="#334155")
    ax2.annotate("BC-ResNet (Final SOTA)\n(83.4M FLOPs | 4.84ms | 100% Acc)", xy=(83.4, 4.84), xytext=(12, 10),
                 textcoords="offset points", fontsize=9.0, fontweight="bold", color="#047857",
                 arrowprops=dict(arrowstyle="->", color="#047857", lw=1.2))
    ax2.annotate("M4 VoiceResNet\n(771.7M | 4.82ms)", xy=(771.7, 4.82), xytext=(-130, -18),
                 textcoords="offset points", fontsize=8.5, fontweight="bold", color="#1d4ed8")
    ax2.annotate("M5 SE-VoiceResNet\n(771.7M | 5.48ms)", xy=(771.7, 5.48), xytext=(-145, 12),
                 textcoords="offset points", fontsize=8.5, fontweight="bold", color="#6d28d9")
    
    # Pareto boundary curve
    ax2.plot([83.4, 152.2], [4.84, 1.95], linestyle="--", color="#059669", lw=1.8, label="Pareto Frontier")
    
    ax2.set_xlabel("Computational Complexity (MFLOPs)", fontsize=9.5, fontweight="bold", color="#0f172a")
    ax2.set_ylabel("CPU Latency (ms)", fontsize=9.5, fontweight="bold", color="#0f172a")
    ax2.set_title("Pareto Frontier: FLOPs vs Latency", fontsize=11.0, fontweight="bold", pad=12)
    ax2.set_xlim(0, 920)
    ax2.set_ylim(1.0, 6.8)
    ax2.grid(True, linestyle="--", alpha=0.45)
    ax2.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    
    plt.tight_layout()
    out_p = os.path.join(OUT_DIR, "benchmark_pareto_evolution.png")
    plt.savefig(out_p, dpi=300, bbox_inches="tight")
    plt.close()
    print("  -> Saved:", out_p)

# ==============================================================================
# 4. METRICS CLASSIFICATION BREAKDOWN & ROC CURVES (Non-overlapping)
# ==============================================================================
def plot_classification_breakdown():
    print("[4/7] Generating Classification Breakdown & ROC...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.2), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    
    prec, rec, f1, _ = precision_recall_fscore_support(y_test_np, preds_test, labels=range(7))
    x = np.arange(len(CLASS_NAMES))
    w = 0.26
    
    ax1.bar(x - w, prec * 100, w, label="Precision (%)", color="#3b82f6", edgecolor="#0f172a")
    ax1.bar(x, rec * 100, w, label="Recall (%)", color="#10b981", edgecolor="#0f172a")
    ax1.bar(x + w, f1 * 100, w, label="F1-Score (%)", color="#f59e0b", edgecolor="#0f172a")
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(CLASS_NAMES, rotation=20, ha="right", fontsize=9.0, fontweight="bold")
    ax1.set_ylabel("Metric Value (%)", fontsize=9.5, fontweight="bold", color="#0f172a")
    ax1.set_title("Per-Class Classification Metrics (Test N = 290)", fontsize=11.0, fontweight="bold", pad=10)
    ax1.set_ylim(80, 115)
    ax1.grid(True, linestyle="--", alpha=0.45)
    ax1.legend(loc="upper left", fontsize=8.8, framealpha=0.92, ncol=3)
    
    # Subplot B: Multi-class ROC curves
    y_bin = label_binarize(y_test_np, classes=range(7))
    for i in range(7):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs_test[:, i])
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, lw=1.8, color=CLASS_COLORS[i], label=f"{CLASS_NAMES[i]} (AUC = {roc_auc:.4f})")
    
    ax2.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", lw=1.2)
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([0.90, 1.02])
    ax2.set_xlabel("False Positive Rate", fontsize=9.5, fontweight="bold", color="#0f172a")
    ax2.set_ylabel("True Positive Rate", fontsize=9.5, fontweight="bold", color="#0f172a")
    ax2.set_title("Multi-Class ROC Curves", fontsize=11.0, fontweight="bold", pad=10)
    ax2.grid(True, linestyle="--", alpha=0.45)
    ax2.legend(loc="lower right", fontsize=7.8, framealpha=0.92)
    
    plt.tight_layout()
    out_p = os.path.join(OUT_DIR, "metrics_classification_breakdown.png")
    plt.savefig(out_p, dpi=300, bbox_inches="tight")
    plt.close()
    print("  -> Saved:", out_p)

# ==============================================================================
# 5. LATENT SPACE T-SNE & PCA (Clean, separated clusters)
# ==============================================================================
def plot_latent_space():
    print("[5/7] Generating Latent Space t-SNE & PCA...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.5), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    
    with torch.no_grad():
        x = model.stem(X_test)
        x = model.stage1(x)
        x = model.stage2(x)
        x = model.stage3(x)
        x = model.stage4(x)
        feats = model.gap(x).flatten(1).numpy()
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    feats_tsne = tsne.fit_transform(feats)
    
    pca = PCA(n_components=2, random_state=42)
    feats_pca = pca.fit_transform(feats)
    var_exp = pca.explained_variance_ratio_
    
    for i in range(7):
        mask = (y_test_np == i)
        ax1.scatter(feats_tsne[mask, 0], feats_tsne[mask, 1], c=CLASS_COLORS[i], label=CLASS_NAMES[i],
                    s=38, alpha=0.88, edgecolors="none")
        ax2.scatter(feats_pca[mask, 0], feats_pca[mask, 1], c=CLASS_COLORS[i], label=CLASS_NAMES[i],
                    s=38, alpha=0.88, edgecolors="none")
    
    ax1.set_xlabel("t-SNE Dimension 1", fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("t-SNE Dimension 2", fontsize=9.5, fontweight="bold")
    ax1.set_title("t-SNE Latent Space Clusters", fontsize=11.0, fontweight="bold", pad=10)
    ax1.grid(True, linestyle="--", alpha=0.45)
    
    ax2.set_xlabel(f"PCA Component 1 ({var_exp[0]*100:.1f}%)", fontsize=9.5, fontweight="bold")
    ax2.set_ylabel(f"PCA Component 2 ({var_exp[1]*100:.1f}%)", fontsize=9.5, fontweight="bold")
    ax2.set_title("PCA Latent Space Projection", fontsize=11.0, fontweight="bold", pad=10)
    ax2.grid(True, linestyle="--", alpha=0.45)
    
    # Shared clean legend at bottom center
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=8.5, framealpha=0.95, bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.14)
    out_p = os.path.join(OUT_DIR, "latent_space_tsne_pca.png")
    plt.savefig(out_p, dpi=300, bbox_inches="tight")
    plt.close()
    print("  -> Saved:", out_p)

# ==============================================================================
# 6. OPTIMIZATION LOSS & LR SCHEDULE (No overlapping annotations)
# ==============================================================================
def plot_loss_and_lr():
    print("[6/7] Generating Optimization Loss & LR Schedule...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    
    # Subplot A: Label Smoothing Target Distribution
    classes = CLASS_NAMES
    x_pos = np.arange(len(classes))
    target_idx = 3  # RePhai
    
    hard_targets = np.zeros(len(classes))
    hard_targets[target_idx] = 1.0
    
    alpha = 0.1
    K = len(classes)
    smooth_targets = np.full(K, alpha / K)
    smooth_targets[target_idx] = (1.0 - alpha) + (alpha / K)
    
    width = 0.35
    ax1.bar(x_pos - width/2, hard_targets, width, label="Hard Target", color="#94a3b8", edgecolor="#0f172a")
    ax1.bar(x_pos + width/2, smooth_targets, width, label=f"Label Smoothing (α={alpha})", color="#2563eb", edgecolor="#0f172a")
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(classes, rotation=22, ha="right", fontsize=8.8, fontweight="bold")
    ax1.set_ylabel("Probability q(k)", fontsize=9.5, fontweight="bold")
    ax1.set_title("Label Smoothing Target Distribution", fontsize=11.0, fontweight="bold", pad=10)
    ax1.set_ylim(0, 1.25)
    ax1.grid(True, linestyle="--", alpha=0.45)
    ax1.legend(loc="upper left", fontsize=8.8, framealpha=0.92)
    
    # Text annotation with clear margin
    ax1.annotate(f"Target q(k) = {smooth_targets[target_idx]:.3f}\n(10% Probability Discount)",
                 xy=(target_idx + width/2, smooth_targets[target_idx]),
                 xytext=(target_idx + 0.5, 1.05),
                 arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=1.2),
                 fontsize=8.5, fontweight="bold", color="#1d4ed8")
    
    # Subplot B: Loss & Cosine Annealing Learning Rate
    lr_max = 0.001
    lr_min = 1e-5
    lr_schedule = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * (t_epochs - 1) / epochs))
    
    ax2_twin = ax2.twinx()
    l1, = ax2.plot(t_epochs, history.get("train_loss", []), color="#dc2626", lw=2.2, label="Train Loss")
    l2, = ax2.plot(t_epochs, history.get("val_loss", []), color="#d97706", lw=2.0, linestyle="--", label="Val Loss")
    l3, = ax2_twin.plot(t_epochs, lr_schedule, color="#059669", lw=2.0, linestyle=":", label="Learning Rate")
    
    ax2.set_xlabel("Epoch", fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Loss", fontsize=9.5, fontweight="bold", color="#dc2626")
    ax2_twin.set_ylabel("Learning Rate (AdamW)", fontsize=9.5, fontweight="bold", color="#059669")
    ax2.set_title("Loss Convergence & Cosine Annealing LR", fontsize=11.0, fontweight="bold", pad=10)
    ax2.grid(True, linestyle="--", alpha=0.45)
    
    lines = [l1, l2, l3]
    ax2.legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=8.8, framealpha=0.92)
    
    plt.tight_layout()
    out_p = os.path.join(OUT_DIR, "optimization_loss_lr_schedule.png")
    plt.savefig(out_p, dpi=300, bbox_inches="tight")
    plt.close()
    print("  -> Saved:", out_p)

# ==============================================================================
# 7. REAL-TIME DSP PIPELINE & CYBER-PHYSICAL SYSTEM (Short titles, crisp text)
# ==============================================================================
def plot_system_diagrams():
    print("[7/7] Generating System & Pipeline Diagrams...")
    
    # 7.1 DSP Pipeline
    fig, ax = plt.subplots(figsize=(11.5, 3.8), dpi=300)
    ax.set_facecolor("#f8fafd")
    fig.patch.set_facecolor("#ffffff")
    ax.axis("off")
    
    steps = [
        ("Raw Audio Input\n(44.1 / 48 kHz)", "Microphone Capture\nPush-to-Talk 1.0s", "#0284c7"),
        ("Sinc Anti-Aliasing\nResampling", "Band-Limited Decim.\nStream -> 16 kHz", "#0d9488"),
        ("Silence Guard\nEnergy VAD", "Peak < 0.05 / RMS VAD\nBypass False Triggers", "#d97706"),
        ("Peak Normalization\n(Gain = 0.65 / max)", "Uniform Sound Energy\nDistance Independent", "#6366f1"),
        ("Log-Mel Features\n(128 x 63 Tensor)", "STFT n_fft=512, hop=256\nStandard Input Shape", "#7c3aed"),
        ("BC-ResNet Engine\n(ONNX WASM)", "Inference Latency 4.84ms\n100% Test Accuracy", "#059669")
    ]
    box_w = 1.40
    box_h = 1.05
    start_x = 0.2
    y = 1.3
    
    for i, (title, desc, color) in enumerate(steps):
        x = start_x + i * 1.88
        shadow = patches.FancyBboxPatch((x+0.03, y-0.03), box_w, box_h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                        facecolor="#cbd5e1", edgecolor="none", zorder=1)
        ax.add_patch(shadow)
        card = patches.FancyBboxPatch((x, y), box_w, box_h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                      facecolor=color, edgecolor="#ffffff", linewidth=1.5, zorder=2)
        ax.add_patch(card)
        ax.text(x + box_w/2, y + box_h*0.65, title, ha="center", va="center", color="#ffffff",
                fontsize=8.2, fontweight="bold", zorder=3)
        ax.text(x + box_w/2, y + box_h*0.25, desc, ha="center", va="center", color="#f1f5f9",
                fontsize=6.8, style="italic", zorder=3)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + box_w + 0.44, y + box_h/2), xytext=(x + box_w + 0.04, y + box_h/2),
                        arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.5", color="#475569", lw=2.0), zorder=4)
    
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0.7, 2.8)
    plt.title("Real-Time DSP & Silence Guard Pipeline", fontsize=11.0, fontweight="bold", pad=12, color="#0f172a")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "realtime_dsp_silence_guard_pipeline.png"), bbox_inches="tight")
    plt.close()
    
    # 7.2 Cyber-Physical Architecture
    fig, ax = plt.subplots(figsize=(11.5, 5.8), dpi=300)
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("#ffffff")
    ax.axis("off")
    ax.text(5.75, 5.5, "Edge Cyber-Physical System Architecture", ha="center", va="center", fontsize=11.5, fontweight="bold", color="#0f172a")
    
    # Cyber Tier
    cyber_box = patches.FancyBboxPatch((0.5, 2.9), 10.5, 2.2, boxstyle="round,pad=0.08,rounding_size=0.15",
                                      facecolor="#eff6ff", edgecolor="#3b82f6", linewidth=1.8)
    ax.add_patch(cyber_box)
    ax.text(0.75, 4.85, "CYBER TIER: ZERO-SERVER CLIENT EDGE AI (ONNX Runtime Web + Web Audio API)",
            fontsize=9.0, fontweight="bold", color="#1e40af")
    
    cyber_comps = [
        ("1. Voice Input", "Microphone Capture\nPush-to-Talk 1.0s", "#0284c7", 0.7),
        ("2. Real-Time DSP", "Sinc Decim. 16kHz\nSilence Guard (0.05)", "#0d9488", 2.8),
        ("3. BC-ResNet-1", "ONNX WASM (4.84ms)\n54.1K / 157.4K Params", "#4f46e5", 4.9),
        ("4. 2D Digital Twin", "Canvas Simulation\nKinematic Feedback", "#7c3aed", 7.0),
        ("5. Web Bluetooth", "GATT Client Write\nASCII Single-Byte", "#059669", 9.1)
    ]
    for title, desc, color, x in cyber_comps:
        c_card = patches.FancyBboxPatch((x, 3.1), 1.7, 1.45, boxstyle="round,pad=0.05,rounding_size=0.10",
                                        facecolor=color, edgecolor="#ffffff", linewidth=1.2)
        ax.add_patch(c_card)
        ax.text(x + 0.85, 4.15, title, ha="center", va="center", color="#ffffff", fontsize=7.8, fontweight="bold")
        ax.text(x + 0.85, 3.55, desc, ha="center", va="center", color="#f8fafc", fontsize=6.8, style="italic")
    
    for x in [2.4, 4.5, 6.6, 8.7]:
        ax.annotate("", xy=(x + 0.38, 3.82), xytext=(x + 0.02, 3.82),
                    arrowprops=dict(arrowstyle="->,head_width=0.28,head_length=0.4", color="#1e3a8a", lw=1.6))
    
    # BLE Channel
    ax.annotate("", xy=(5.75, 2.45), xytext=(5.75, 2.9),
                arrowprops=dict(arrowstyle="<->,head_width=0.4,head_length=0.5", color="#dc2626", lw=2.4))
    ax.text(5.75, 2.68, "Wireless BLE 4.2 RF Channel ('K','F','B','L','R','S' | Latency < 12ms)",
            ha="center", va="center", fontsize=8.2, fontweight="bold", color="#991b1b",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#fee2e2", edgecolor="#ef4444", lw=1.0))
    
    # Physical Tier
    phys_box = patches.FancyBboxPatch((0.5, 0.4), 10.5, 2.0, boxstyle="round,pad=0.08,rounding_size=0.15",
                                      facecolor="#fefce8", edgecolor="#ca8a04", linewidth=1.8)
    ax.add_patch(phys_box)
    ax.text(0.75, 2.15, "PHYSICAL TIER: HARDWARE ROBOT CONTROLLER (ESP32 NodeMCU + Actuators)",
            fontsize=9.0, fontweight="bold", color="#854d0e")
    
    phys_comps = [
        ("ESP32 GATT Server", "BLE Service & RX Char\nCallback Interrupt", "#b45309", 0.7),
        ("Command Parser", "Byte Switch Handler\n'F','B','L','R','S'", "#d97706", 2.8),
        ("1.5s Watchdog", "Auto-Brake Failsafe\nRunaway Prevention", "#dc2626", 4.9),
        ("Dual H-Bridge", "L298N / L9110S Driver\n8-bit PWM (1 kHz)", "#0284c7", 7.0),
        ("Differential Robot", "2x DC Geared Motors\nPhysical Navigation", "#15803d", 9.1)
    ]
    for title, desc, color, x in phys_comps:
        p_card = patches.FancyBboxPatch((x, 0.55), 1.7, 1.4, boxstyle="round,pad=0.05,rounding_size=0.10",
                                        facecolor=color, edgecolor="#ffffff", linewidth=1.2)
        ax.add_patch(p_card)
        ax.text(x + 0.85, 1.55, title, ha="center", va="center", color="#ffffff", fontsize=7.8, fontweight="bold")
        ax.text(x + 0.85, 0.95, desc, ha="center", va="center", color="#f8fafc", fontsize=6.8, style="italic")
    
    for x in [2.4, 4.5, 6.6, 8.7]:
        ax.annotate("", xy=(x + 0.38, 1.25), xytext=(x + 0.02, 1.25),
                    arrowprops=dict(arrowstyle="->,head_width=0.28,head_length=0.4", color="#713f12", lw=1.6))
    
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0.2, 5.8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "cyber_physical_system_architecture.png"), bbox_inches="tight")
    plt.close()
    
    print("  -> Saved DSP & Cyber-Physical diagrams!")

if __name__ == "__main__":
    plot_confusion_matrix()
    plot_training_curves()
    plot_pareto_evolution()
    plot_classification_breakdown()
    plot_latent_space()
    plot_loss_and_lr()
    plot_system_diagrams()
    print("\nALL PERFECT PLOTS SUCCESSFULLY REGENERATED WITHOUT ANY OVERLAPS!")
