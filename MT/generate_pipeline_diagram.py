import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

# Set up figure
fig, ax = plt.subplots(figsize=(15.5, 9.0), dpi=300)
fig.patch.set_facecolor('#FFFFFF')
ax.set_facecolor('#FFFFFF')
ax.set_xlim(0, 15.5)
ax.set_ylim(0, 9.0)
ax.axis('off')

# Title & Subtitle with high aesthetic styling
ax.text(7.75, 8.55, "END-TO-END ADVANCED DEEP LEARNING ENGINEERING PIPELINE", 
        ha='center', va='center', fontsize=16, fontweight='bold', color='#0F172A', fontfamily='DejaVu Sans')
ax.text(7.75, 8.18, "Vietnamese Voice Command Recognition for Autonomous Mobile Robots (Data_1400 Benchmark)", 
        ha='center', va='center', fontsize=11, fontstyle='italic', color='#475569', fontfamily='DejaVu Sans')

# Helper function to draw rounded box
def draw_stage(x, y, w, h, header_text, bg_color, border_color, badge_color, items, step_num):
    # Outer box
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.18",
                         edgecolor=border_color, facecolor=bg_color, linewidth=1.8, zorder=2)
    ax.add_patch(box)
    
    # Header badge
    badge = FancyBboxPatch((x, y + h - 0.6), w, 0.6, boxstyle="round,pad=0.04,rounding_size=0.14",
                           edgecolor=badge_color, facecolor=badge_color, linewidth=1, zorder=3)
    ax.add_patch(badge)
    
    # Step circle
    circle = patches.Circle((x + 0.38, y + h - 0.3), 0.22, facecolor='#FFFFFF', edgecolor=badge_color, linewidth=1.5, zorder=4)
    ax.add_patch(circle)
    ax.text(x + 0.38, y + h - 0.3, str(step_num), ha='center', va='center', fontsize=9.5, fontweight='bold', color=badge_color, zorder=5)
    
    # Header title
    ax.text(x + 0.72, y + h - 0.3, header_text, ha='left', va='center', fontsize=10.5, fontweight='bold', color='#FFFFFF', zorder=4)
    
    # Items
    item_y = y + h - 0.92
    for item in items:
        ax.text(x + 0.2, item_y, "• " + item[0], ha='left', va='top', fontsize=8.8, fontweight='bold', color='#1E293B', zorder=4)
        if len(item) > 1 and item[1]:
            ax.text(x + 0.36, item_y - 0.26, item[1], ha='left', va='top', fontsize=8.0, color='#475569', zorder=4)
            item_y -= 0.56
        else:
            item_y -= 0.35

# Helper function for connectors
def draw_arrow(x1, y1, x2, y2, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#0F172A", lw=2.2, mutation_scale=15), zorder=6)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + 0.18
        ax.text(mx, my, label, ha='center', va='center', fontsize=8.0, fontweight='bold', color='#0F172A',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor="#94A3B8", lw=0.9), zorder=7)

# Top Row (Stage 1 to 3)
# Stage 1: Raw Audio Acquisition
draw_stage(0.6, 4.6, 4.1, 3.2, "ACOUSTIC ACQUISITION", "#F8FAFC", "#334155", "#1E293B", [
    ("7 Robot Navigation Classes", "KhoiDong, Lui, RePhai, ReTrai, Tien, TamDung, KhongXacDinh"),
    ("Clean Dataset: 1,400 WAVs", "fs = 16,000 Hz, Single-channel Mono (200 files/class)"),
    ("100% Unique (Zero Leakage)", "Verified MD5 & raw PCM audio hash (0 duplicates)"),
    ("Temporal Duration Spread", "Natural speech duration: 0.82s - 2.32s")
], 1)

# Stage 2: Audio Preprocessing
draw_stage(5.7, 4.6, 4.1, 3.2, "STANDARDIZATION & STFT", "#F0FDF4", "#16A34A", "#15803D", [
    ("Uniform 2.0s Center-Pad/Crop", "N_target = 32,000 samples (zero phoneme truncation)"),
    ("Short-Time Fourier Transform", "Hann window: N_fft = 1024, Hop = 512 (~32ms)"),
    ("Log-Mel Spectrogram (dB)", "128 Mel frequency bins × 63 time frames"),
    ("Offline Binary Tensor (.pt)", "dataset_processed.pt (43.08 MB, Train 1120, Val 140, Test 140)")
], 2)

# Stage 3: Data Augmentation
draw_stage(10.8, 4.6, 4.1, 3.2, "SPEC-AUGMENTATION", "#FFFBEB", "#D97706", "#B45309", [
    ("Frequency Masking (F_max=16)", "Invariance against vocal pitch & speaker timbre"),
    ("Time Masking (T_max=12)", "Robustness against audio dropout & ambient gaps"),
    ("Stochastic Mini-Batch Loader", "p = 0.8 during training (strictly inactive during eval)")
], 3)

# Connectors for Top Row
draw_arrow(4.7, 6.2, 5.7, 6.2, "Raw Audio (16kHz)")
draw_arrow(9.8, 6.2, 10.8, 6.2, "Log-Mel (128x63)")

# Connector Top to Bottom
ax.annotate("", xy=(12.85, 3.85), xytext=(12.85, 4.6),
            arrowprops=dict(arrowstyle="-|>", color="#0F172A", lw=2.2, mutation_scale=15), zorder=6)
ax.text(12.85, 4.22, "Augmented Batches", ha='center', va='center', fontsize=8.0, fontweight='bold', color='#0F172A',
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor="#94A3B8", lw=0.9), zorder=7)

# Bottom Row (Stage 4 to 6)
# Stage 4: Deep Neural Network
draw_stage(10.8, 0.45, 4.1, 3.4, "ARCHITECTURAL MODELING", "#F5F3FF", "#7C3AED", "#6D28D9", [
    ("Ablation Study (M1 -> M5)", "M1: Baseline CNN (1.93M params, 5.51M FLOPs)"),
    ("Regularization: M2 & M3", "+Dropout (0.4) & +BatchNorm2d"),
    ("M4: VoiceResNet + GAP", "Residual F(x)+x, cuts 84% params (307.7K)"),
    ("M5: SE_VoiceResNet (SOTA)", "Channel Attention (310.5K params, Test Loss 0.4826)")
], 4)

# Stage 5: Advanced Training
draw_stage(5.7, 0.45, 4.1, 3.4, "OPTIMIZATION & LOSS", "#FEF2F2", "#DC2626", "#B91C1C", [
    ("Loss Function Formulation", "Label Smoothing Cross-Entropy (alpha = 0.1)"),
    ("Optimizer Benchmark", "SGD (69.3%) vs Momentum (100%) vs Adam vs AdamW"),
    ("AdamW + Cosine Annealing", "LR 1e-3 -> 1e-6, weight decay = 1e-4 (Flat optimum)"),
    ("Batch Size: 32 | Epochs: 20-35", "Hardware-accelerated training (Apple Silicon MPS / CUDA)")
], 5)

# Stage 6: Scientific Evaluation & Deployment
draw_stage(0.6, 0.45, 4.1, 3.4, "EVALUATION & DEPLOYMENT", "#EFF6FF", "#2563EB", "#1D4ED8", [
    ("Pareto Frontier Dominance", "M5 achieves top Pareto trade-off (Params vs Latency)"),
    ("100% Accuracy on Test Set", "140/140 correct (20/20 each class on diagonal)"),
    ("Explainable AI (Grad-CAM)", "Sharp focal activation on Formant bands (F1, F2)"),
    ("Real-Time Edge Roadmap", "CPU Latency 5.48ms, INT8 Quantization (~310KB), ROS2 VAD")
], 6)

# Connectors for Bottom Row
draw_arrow(10.8, 2.15, 9.8, 2.15, "Model Graphs")
draw_arrow(5.7, 2.15, 4.7, 2.15, "Trained Weights")

plt.tight_layout()
output_path = "ADL/results/end_to_end_pipeline.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"[SUCCESS] Pipeline diagram saved to {output_path}")
