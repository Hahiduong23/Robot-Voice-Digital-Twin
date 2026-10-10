import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

os.makedirs("ADL/results/architectures", exist_ok=True)

# -------------------------------------------------------------
# Color Palette matching Figure 8
# -------------------------------------------------------------
C_INPUT     = "#FFA000"  # Amber/Yellow
C_OUTPUT    = "#FFA000"  # Amber/Yellow
C_CONV      = "#4A90E2"  # Slate Blue
C_BN        = "#2E7D32"  # Forest Green
C_POOL      = "#42A5F5"  # Sky Blue
C_DROPOUT   = "#0D2538"  # Deep Navy Blue
C_DENSE     = "#6A1B9A"  # Royal Purple
C_GAP       = "#00838F"  # Teal
C_RES       = "#0097A7"  # Teal/Cyan for Residual Add
C_SE        = "#C2185B"  # Deep Magenta/Rose
C_CONTAINER = "#E65100"  # Burnt Orange Border
C_ARROW     = "#1976D2"  # Vivid Blue Arrow
C_BG        = "#FFFFFF"

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

def draw_box(ax, x, y, w, h, text, facecolor, edgecolor="#263238", 
             textcolor="white", fontsize=8.0, bold=True, pad=0.15):
    weight = "bold" if bold else "normal"
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=f"round,pad={pad},rounding_size=0.12",
                         facecolor=facecolor, edgecolor=edgecolor, linewidth=1.2,
                         zorder=3)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', color=textcolor,
            fontsize=fontsize, fontweight=weight, zorder=4, multialignment='center')

def draw_container(ax, x, y, w, h, label="", border_color=C_CONTAINER):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.18,rounding_size=0.2",
                         facecolor="#FFF8E1", edgecolor=border_color, linewidth=1.5,
                         zorder=1, alpha=0.5)
    ax.add_patch(box)
    if label:
        ax.text(x, y + h/2 + 0.12, label, ha='center', va='bottom', color=border_color,
                fontsize=8.5, fontweight='bold', zorder=4)

def draw_arrow(ax, p1, p2, color=C_ARROW, rad=0.0, lw=1.8):
    ax.annotate("", xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}", shrinkA=3, shrinkB=3),
                zorder=2)

def draw_corner_arrow(ax, p1, p2, p3, color=C_ARROW, lw=1.8):
    # p1 -> p2 -> p3
    ax.plot([p1[0], p2[0], p3[0]], [p1[1], p2[1], p3[1]], color=color, lw=lw, zorder=2)
    ax.annotate("", xy=p3, xytext=(p3[0] + (0.01 if p2[0] < p3[0] else -0.01 if p2[0] > p3[0] else 0), 
                                   p3[1] + (0.01 if p2[1] < p3[1] else -0.01 if p2[1] > p3[1] else 0)),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=2),
                zorder=2)

# =========================================================================
# 1. MODEL M1: Baseline CNN
# =========================================================================
def plot_m1():
    fig, ax = plt.subplots(figsize=(13.5, 4.2), dpi=300)
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-0.8, 3.8)
    ax.axis('off')

    # Input
    draw_box(ax, 0.6, 2.7, 1.4, 0.9, "Input\nMel-Spec\n(1, 128, 63)", C_INPUT, textcolor="#000000", fontsize=7.5)

    # Block 1
    draw_container(ax, 2.5, 2.7, 1.6, 1.9, "Conv Block 1")
    draw_box(ax, 2.5, 3.1, 1.4, 0.7, "Conv2d (32)\nkernel: 3x3, pad: 1\nReLU", C_CONV, fontsize=6.8)
    draw_box(ax, 2.5, 2.1, 1.4, 0.6, "MaxPool2d\n(2, 2) -> (32, 64, 31)", C_POOL, fontsize=6.8)
    draw_arrow(ax, (2.5, 2.75), (2.5, 2.4))

    # Block 2
    draw_container(ax, 4.6, 2.7, 1.6, 1.9, "Conv Block 2")
    draw_box(ax, 4.6, 3.1, 1.4, 0.7, "Conv2d (64)\nkernel: 3x3, pad: 1\nReLU", C_CONV, fontsize=6.8)
    draw_box(ax, 4.6, 2.1, 1.4, 0.6, "MaxPool2d\n(2, 2) -> (64, 32, 15)", C_POOL, fontsize=6.8)
    draw_arrow(ax, (4.6, 2.75), (4.6, 2.4))

    # Block 3
    draw_container(ax, 6.7, 2.7, 1.6, 1.9, "Conv Block 3")
    draw_box(ax, 6.7, 3.1, 1.4, 0.7, "Conv2d (128)\nkernel: 3x3, pad: 1\nReLU", C_CONV, fontsize=6.8)
    draw_box(ax, 6.7, 2.1, 1.4, 0.6, "MaxPool2d\n(2, 2) -> (128, 16, 7)", C_POOL, fontsize=6.8)
    draw_arrow(ax, (6.7, 2.75), (6.7, 2.4))

    # Flatten
    draw_box(ax, 8.7, 2.7, 1.4, 0.8, "Flatten Layer\nVector dim:\n14,336", "#455A64", fontsize=7.2)

    # Fully Connected 1
    draw_box(ax, 10.8, 3.0, 1.8, 0.8, "Fully Connected\n(out: 128, act: ReLU)", C_DENSE, fontsize=7.5)
    # Fully Connected 2
    draw_box(ax, 10.8, 1.7, 1.8, 0.8, "Fully Connected\n(out: 7, act: Softmax)", C_DENSE, fontsize=7.5)
    # Output
    draw_box(ax, 10.8, 0.4, 1.4, 0.7, "Output\n(7 Classes)", C_OUTPUT, textcolor="#000000", fontsize=7.5)

    # Connecting Arrows
    draw_arrow(ax, (1.3, 2.7), (1.7, 2.7))
    draw_arrow(ax, (3.3, 2.7), (3.8, 2.7))
    draw_arrow(ax, (5.4, 2.7), (5.9, 2.7))
    draw_arrow(ax, (7.5, 2.7), (8.0, 2.7))
    draw_arrow(ax, (9.4, 2.7), (9.9, 3.0), rad=-0.1)
    draw_arrow(ax, (10.8, 2.6), (10.8, 2.1))
    draw_arrow(ax, (10.8, 1.3), (10.8, 0.75))

    plt.title("Figure 8. Architecture of Model M1 (Baseline CNN)", fontsize=11.5, fontweight='bold', color="#0D2538", y=-0.08)
    plt.tight_layout()
    plt.savefig("ADL/results/architectures/arch_M1_BaselineCNN.png", bbox_inches='tight', dpi=300)
    plt.close()

# =========================================================================
# 2. MODEL M2: Baseline + Dropout
# =========================================================================
def plot_m2():
    fig, ax = plt.subplots(figsize=(13.5, 4.4), dpi=300)
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-0.8, 4.0)
    ax.axis('off')

    # Input
    draw_box(ax, 0.6, 2.8, 1.4, 0.9, "Input\nMel-Spec\n(1, 128, 63)", C_INPUT, textcolor="#000000", fontsize=7.5)

    # Block 1
    draw_container(ax, 2.5, 2.8, 1.6, 1.9, "Conv Block 1")
    draw_box(ax, 2.5, 3.2, 1.4, 0.7, "Conv2d (32)\nkernel: 3x3, pad: 1\nReLU", C_CONV, fontsize=6.8)
    draw_box(ax, 2.5, 2.2, 1.4, 0.6, "MaxPool2d\n(2, 2) -> (32, 64, 31)", C_POOL, fontsize=6.8)
    draw_arrow(ax, (2.5, 2.85), (2.5, 2.5))

    # Block 2
    draw_container(ax, 4.6, 2.8, 1.6, 1.9, "Conv Block 2")
    draw_box(ax, 4.6, 3.2, 1.4, 0.7, "Conv2d (64)\nkernel: 3x3, pad: 1\nReLU", C_CONV, fontsize=6.8)
    draw_box(ax, 4.6, 2.2, 1.4, 0.6, "MaxPool2d\n(2, 2) -> (64, 32, 15)", C_POOL, fontsize=6.8)
    draw_arrow(ax, (4.6, 2.85), (4.6, 2.5))

    # Block 3
    draw_container(ax, 6.7, 2.8, 1.6, 1.9, "Conv Block 3")
    draw_box(ax, 6.7, 3.2, 1.4, 0.7, "Conv2d (128)\nkernel: 3x3, pad: 1\nReLU", C_CONV, fontsize=6.8)
    draw_box(ax, 6.7, 2.2, 1.4, 0.6, "MaxPool2d\n(2, 2) -> (128, 16, 7)", C_POOL, fontsize=6.8)
    draw_arrow(ax, (6.7, 2.85), (6.7, 2.5))

    # Flatten
    draw_box(ax, 8.7, 2.8, 1.4, 0.8, "Flatten Layer\nVector dim:\n14,336", "#455A64", fontsize=7.2)

    # Fully Connected 1
    draw_box(ax, 10.8, 3.3, 1.8, 0.7, "Fully Connected\n(out: 128, act: ReLU)", C_DENSE, fontsize=7.2)
    # Dropout (Highlighted novelty)
    draw_box(ax, 10.8, 2.3, 1.6, 0.6, "Dropout Layer\n(rate = 0.4)", C_DROPOUT, fontsize=7.2, edgecolor="#FFD54F")
    # Fully Connected 2
    draw_box(ax, 10.8, 1.3, 1.8, 0.7, "Fully Connected\n(out: 7, act: Softmax)", C_DENSE, fontsize=7.2)
    # Output
    draw_box(ax, 10.8, 0.2, 1.4, 0.6, "Output\n(7 Classes)", C_OUTPUT, textcolor="#000000", fontsize=7.5)

    # Connecting Arrows
    draw_arrow(ax, (1.3, 2.8), (1.7, 2.8))
    draw_arrow(ax, (3.3, 2.8), (3.8, 2.8))
    draw_arrow(ax, (5.4, 2.8), (5.9, 2.8))
    draw_arrow(ax, (7.5, 2.8), (8.0, 2.8))
    draw_arrow(ax, (9.4, 2.8), (9.9, 3.3), rad=-0.1)
    draw_arrow(ax, (10.8, 2.95), (10.8, 2.6))
    draw_arrow(ax, (10.8, 2.0), (10.8, 1.65))
    draw_arrow(ax, (10.8, 0.95), (10.8, 0.5))

    plt.title("Figure 9. Architecture of Model M2 (Dropout CNN)", fontsize=11.5, fontweight='bold', color="#0D2538", y=-0.08)
    plt.tight_layout()
    plt.savefig("ADL/results/architectures/arch_M2_DropoutCNN.png", bbox_inches='tight', dpi=300)
    plt.close()

# =========================================================================
# 3. MODEL M3: Baseline + Dropout + BatchNorm2d
# =========================================================================
def plot_m3():
    fig, ax = plt.subplots(figsize=(14.0, 4.8), dpi=300)
    ax.set_xlim(-0.5, 12.8)
    ax.set_ylim(-0.8, 4.4)
    ax.axis('off')

    # Input
    draw_box(ax, 0.6, 2.9, 1.4, 0.9, "Input\nMel-Spec\n(1, 128, 63)", C_INPUT, textcolor="#000000", fontsize=7.5)

    # Block 1
    draw_container(ax, 2.5, 2.9, 1.7, 2.4, "Conv-BN Block 1")
    draw_box(ax, 2.5, 3.7, 1.5, 0.55, "Conv2d (32)\nk: 3x3, pad: 1", C_CONV, fontsize=6.8)
    draw_box(ax, 2.5, 2.95, 1.5, 0.55, "BatchNorm2d (32)\n+ ReLU", C_BN, fontsize=6.8)
    draw_box(ax, 2.5, 2.1, 1.5, 0.55, "MaxPool2d (2, 2)\n-> (32, 64, 31)", C_POOL, fontsize=6.8)
    draw_arrow(ax, (2.5, 3.42), (2.5, 3.22))
    draw_arrow(ax, (2.5, 2.67), (2.5, 2.37))

    # Block 2
    draw_container(ax, 4.7, 2.9, 1.7, 2.4, "Conv-BN Block 2")
    draw_box(ax, 4.7, 3.7, 1.5, 0.55, "Conv2d (64)\nk: 3x3, pad: 1", C_CONV, fontsize=6.8)
    draw_box(ax, 4.7, 2.95, 1.5, 0.55, "BatchNorm2d (64)\n+ ReLU", C_BN, fontsize=6.8)
    draw_box(ax, 4.7, 2.1, 1.5, 0.55, "MaxPool2d (2, 2)\n-> (64, 32, 15)", C_POOL, fontsize=6.8)
    draw_arrow(ax, (4.7, 3.42), (4.7, 3.22))
    draw_arrow(ax, (4.7, 2.67), (4.7, 2.37))

    # Block 3
    draw_container(ax, 6.9, 2.9, 1.7, 2.4, "Conv-BN Block 3")
    draw_box(ax, 6.9, 3.7, 1.5, 0.55, "Conv2d (128)\nk: 3x3, pad: 1", C_CONV, fontsize=6.8)
    draw_box(ax, 6.9, 2.95, 1.5, 0.55, "BatchNorm2d (128)\n+ ReLU", C_BN, fontsize=6.8)
    draw_box(ax, 6.9, 2.1, 1.5, 0.55, "MaxPool2d (2, 2)\n-> (128, 16, 7)", C_POOL, fontsize=6.8)
    draw_arrow(ax, (6.9, 3.42), (6.9, 3.22))
    draw_arrow(ax, (6.9, 2.67), (6.9, 2.37))

    # Flatten
    draw_box(ax, 9.0, 2.9, 1.4, 0.8, "Flatten Layer\nVector dim:\n14,336", "#455A64", fontsize=7.2)

    # Fully Connected 1
    draw_box(ax, 11.2, 3.5, 1.8, 0.7, "Fully Connected\n(out: 128, act: ReLU)", C_DENSE, fontsize=7.2)
    # Dropout
    draw_box(ax, 11.2, 2.45, 1.6, 0.6, "Dropout Layer\n(rate = 0.4)", C_DROPOUT, fontsize=7.2)
    # Fully Connected 2
    draw_box(ax, 11.2, 1.4, 1.8, 0.7, "Fully Connected\n(out: 7, act: Softmax)", C_DENSE, fontsize=7.2)
    # Output
    draw_box(ax, 11.2, 0.25, 1.4, 0.6, "Output\n(7 Classes)", C_OUTPUT, textcolor="#000000", fontsize=7.5)

    # Connecting Arrows
    draw_arrow(ax, (1.3, 2.9), (1.65, 2.9))
    draw_arrow(ax, (3.35, 2.9), (3.85, 2.9))
    draw_arrow(ax, (5.55, 2.9), (6.05, 2.9))
    draw_arrow(ax, (7.75, 2.9), (8.3, 2.9))
    draw_arrow(ax, (9.7, 2.9), (10.3, 3.5), rad=-0.1)
    draw_arrow(ax, (11.2, 3.15), (11.2, 2.75))
    draw_arrow(ax, (11.2, 2.15), (11.2, 1.75))
    draw_arrow(ax, (11.2, 1.05), (11.2, 0.55))

    plt.title("Figure 10. Architecture of Model M3 (BatchNorm CNN)", fontsize=11.5, fontweight='bold', color="#0D2538", y=-0.08)
    plt.tight_layout()
    plt.savefig("ADL/results/architectures/arch_M3_BatchNormCNN.png", bbox_inches='tight', dpi=300)
    plt.close()

# =========================================================================
# 4. MODEL M4: VoiceResNet (Residual Blocks + GAP)
# =========================================================================
def plot_m4():
    fig, ax = plt.subplots(figsize=(14.2, 4.8), dpi=300)
    ax.set_xlim(-0.5, 13.0)
    ax.set_ylim(-0.8, 4.4)
    ax.axis('off')

    # Input
    draw_box(ax, 0.6, 2.8, 1.3, 0.9, "Input\nMel-Spec\n(1, 128, 63)", C_INPUT, textcolor="#000000", fontsize=7.5)

    # Stem
    draw_box(ax, 2.1, 2.8, 1.4, 1.2, "Stem Block\nConv2d(32)\n+ BN + ReLU\n(32, 128, 63)", "#00695C", fontsize=7.0)

    # ResBlock 1
    draw_container(ax, 4.2, 2.8, 2.0, 2.4, "Residual Block 1 (s=1)")
    draw_box(ax, 4.2, 3.6, 1.7, 0.55, "Conv-BN-ReLU (32)", C_CONV, fontsize=6.8)
    draw_box(ax, 4.2, 2.8, 1.7, 0.55, "Conv-BN (32)", C_CONV, fontsize=6.8)
    draw_box(ax, 4.2, 2.0, 1.7, 0.55, "Add [F(x)+x] + ReLU\n(32, 128, 63)", C_RES, fontsize=6.8)
    draw_arrow(ax, (4.2, 3.32), (4.2, 3.08))
    draw_arrow(ax, (4.2, 2.52), (4.2, 2.28))
    # Identity shortcut
    draw_corner_arrow(ax, (3.2, 3.6), (3.0, 2.0), (3.35, 2.0), color="#E65100", lw=1.5)

    # ResBlock 2
    draw_container(ax, 6.7, 2.8, 2.1, 2.4, "Residual Block 2 (s=2)")
    draw_box(ax, 6.7, 3.6, 1.8, 0.55, "Conv-BN-ReLU (64, s=2)", C_CONV, fontsize=6.8)
    draw_box(ax, 6.7, 2.8, 1.8, 0.55, "Conv-BN (64)", C_CONV, fontsize=6.8)
    draw_box(ax, 6.7, 2.0, 1.8, 0.55, "Add [F(x)+W_s*x] + ReLU\n(64, 64, 32)", C_RES, fontsize=6.8)
    draw_arrow(ax, (6.7, 3.32), (6.7, 3.08))
    draw_arrow(ax, (6.7, 2.52), (6.7, 2.28))
    # 1x1 Conv shortcut
    draw_corner_arrow(ax, (5.65, 3.6), (5.5, 2.0), (5.8, 2.0), color="#E65100", lw=1.5)

    # ResBlock 3
    draw_container(ax, 9.3, 2.8, 2.1, 2.4, "Residual Block 3 (s=2)")
    draw_box(ax, 9.3, 3.6, 1.8, 0.55, "Conv-BN-ReLU (128, s=2)", C_CONV, fontsize=6.8)
    draw_box(ax, 9.3, 2.8, 1.8, 0.55, "Conv-BN (128)", C_CONV, fontsize=6.8)
    draw_box(ax, 9.3, 2.0, 1.8, 0.55, "Add [F(x)+W_s*x] + ReLU\n(128, 32, 16)", C_RES, fontsize=6.8)
    draw_arrow(ax, (9.3, 3.32), (9.3, 3.08))
    draw_arrow(ax, (9.3, 2.52), (9.3, 2.28))
    # 1x1 Conv shortcut
    draw_corner_arrow(ax, (8.25, 3.6), (8.1, 2.0), (8.4, 2.0), color="#E65100", lw=1.5)

    # GAP + Head
    draw_box(ax, 11.6, 3.6, 1.8, 0.7, "Global Avg Pooling\nAdaptiveAvgPool2d(1,1)\n-> (128)", C_GAP, fontsize=7.0, edgecolor="#FFD54F")
    draw_box(ax, 11.6, 2.5, 1.6, 0.55, "Dropout Layer\n(rate = 0.15)", C_DROPOUT, fontsize=7.0)
    draw_box(ax, 11.6, 1.5, 1.8, 0.65, "Fully Connected\n(out: 7, act: Softmax)", C_DENSE, fontsize=7.2)
    draw_box(ax, 11.6, 0.4, 1.3, 0.6, "Output\n(7 Classes)", C_OUTPUT, textcolor="#000000", fontsize=7.5)

    # Connecting arrows
    draw_arrow(ax, (1.25, 2.8), (1.4, 2.8))
    draw_arrow(ax, (2.8, 2.8), (3.2, 2.8))
    draw_arrow(ax, (5.2, 2.8), (5.65, 2.8))
    draw_arrow(ax, (7.75, 2.8), (8.25, 2.8))
    draw_arrow(ax, (10.35, 2.8), (10.7, 3.6), rad=-0.1)
    draw_arrow(ax, (11.6, 3.25), (11.6, 2.78))
    draw_arrow(ax, (11.6, 2.22), (11.6, 1.83))
    draw_arrow(ax, (11.6, 1.17), (11.6, 0.7))

    plt.title("Figure 11. Architecture of Model M4 (VoiceResNet with Residual Blocks & GAP)", fontsize=11.5, fontweight='bold', color="#0D2538", y=-0.08)
    plt.tight_layout()
    plt.savefig("ADL/results/architectures/arch_M4_VoiceResNet.png", bbox_inches='tight', dpi=300)
    plt.close()

# =========================================================================
# 5. MODEL M5: SE_VoiceResNet (SOTA with Channel Attention)
# =========================================================================
def plot_m5():
    fig, ax = plt.subplots(figsize=(14.5, 5.2), dpi=300)
    ax.set_xlim(-0.5, 13.2)
    ax.set_ylim(-0.8, 4.7)
    ax.axis('off')

    # Input
    draw_box(ax, 0.6, 3.0, 1.3, 0.9, "Input\nMel-Spec\n(1, 128, 63)", C_INPUT, textcolor="#000000", fontsize=7.5)

    # Stem
    draw_box(ax, 2.1, 3.0, 1.4, 1.2, "Stem Block\nConv2d(32)\n+ BN + ReLU\n(32, 128, 63)", "#00695C", fontsize=7.0)

    # SE-ResBlock 1
    draw_container(ax, 4.3, 3.0, 2.2, 2.9, "SE-Residual Block 1 (s=1)")
    draw_box(ax, 4.3, 4.0, 1.8, 0.5, "Conv-BN-ReLU (32)", C_CONV, fontsize=6.8)
    draw_box(ax, 4.3, 3.3, 1.8, 0.5, "Conv-BN (32)", C_CONV, fontsize=6.8)
    draw_box(ax, 4.3, 2.5, 1.8, 0.55, "SE Attention Block\nSqueeze & Excitation (r=16)", C_SE, fontsize=6.8, edgecolor="#FFD54F")
    draw_box(ax, 4.3, 1.7, 1.8, 0.5, "Add [F_se(x)+x] + ReLU\n(32, 128, 63)", C_RES, fontsize=6.8)
    draw_arrow(ax, (4.3, 3.75), (4.3, 3.55))
    draw_arrow(ax, (4.3, 3.05), (4.3, 2.78))
    draw_arrow(ax, (4.3, 2.22), (4.3, 1.95))
    draw_corner_arrow(ax, (3.2, 4.0), (3.0, 1.7), (3.4, 1.7), color="#E65100", lw=1.5)

    # SE-ResBlock 2
    draw_container(ax, 6.9, 3.0, 2.2, 2.9, "SE-Residual Block 2 (s=2)")
    draw_box(ax, 6.9, 4.0, 1.8, 0.5, "Conv-BN-ReLU (64, s=2)", C_CONV, fontsize=6.8)
    draw_box(ax, 6.9, 3.3, 1.8, 0.5, "Conv-BN (64)", C_CONV, fontsize=6.8)
    draw_box(ax, 6.9, 2.5, 1.8, 0.55, "SE Attention Block\nSqueeze & Excitation (r=16)", C_SE, fontsize=6.8, edgecolor="#FFD54F")
    draw_box(ax, 6.9, 1.7, 1.8, 0.5, "Add [F_se(x)+W_s*x] + ReLU\n(64, 64, 32)", C_RES, fontsize=6.8)
    draw_arrow(ax, (6.9, 3.75), (6.9, 3.55))
    draw_arrow(ax, (6.9, 3.05), (6.9, 2.78))
    draw_arrow(ax, (6.9, 2.22), (6.9, 1.95))
    draw_corner_arrow(ax, (5.8, 4.0), (5.6, 1.7), (6.0, 1.7), color="#E65100", lw=1.5)

    # SE-ResBlock 3
    draw_container(ax, 9.5, 3.0, 2.2, 2.9, "SE-Residual Block 3 (s=2)")
    draw_box(ax, 9.5, 4.0, 1.8, 0.5, "Conv-BN-ReLU (128, s=2)", C_CONV, fontsize=6.8)
    draw_box(ax, 9.5, 3.3, 1.8, 0.5, "Conv-BN (128)", C_CONV, fontsize=6.8)
    draw_box(ax, 9.5, 2.5, 1.8, 0.55, "SE Attention Block\nSqueeze & Excitation (r=16)", C_SE, fontsize=6.8, edgecolor="#FFD54F")
    draw_box(ax, 9.5, 1.7, 1.8, 0.5, "Add [F_se(x)+W_s*x] + ReLU\n(128, 32, 16)", C_RES, fontsize=6.8)
    draw_arrow(ax, (9.5, 3.75), (9.5, 3.55))
    draw_arrow(ax, (9.5, 3.05), (9.5, 2.78))
    draw_arrow(ax, (9.5, 2.22), (9.5, 1.95))
    draw_corner_arrow(ax, (8.4, 4.0), (8.2, 1.7), (8.6, 1.7), color="#E65100", lw=1.5)

    # GAP + Head
    draw_box(ax, 11.9, 3.8, 1.8, 0.7, "Global Avg Pooling\nAdaptiveAvgPool2d(1,1)\n-> (128)", C_GAP, fontsize=7.0, edgecolor="#FFD54F")
    draw_box(ax, 11.9, 2.7, 1.6, 0.55, "Dropout Layer\n(rate = 0.15)", C_DROPOUT, fontsize=7.0)
    draw_box(ax, 11.9, 1.6, 1.8, 0.65, "Fully Connected\n(out: 7, act: Softmax)", C_DENSE, fontsize=7.2)
    draw_box(ax, 11.9, 0.45, 1.3, 0.6, "Output\n(7 Classes)", C_OUTPUT, textcolor="#000000", fontsize=7.5)

    # Connecting arrows
    draw_arrow(ax, (1.25, 3.0), (1.4, 3.0))
    draw_arrow(ax, (2.8, 3.0), (3.2, 3.0))
    draw_arrow(ax, (5.4, 3.0), (5.8, 3.0))
    draw_arrow(ax, (8.0, 3.0), (8.4, 3.0))
    draw_arrow(ax, (10.6, 3.0), (11.0, 3.8), rad=-0.1)
    draw_arrow(ax, (11.9, 3.45), (11.9, 2.98))
    draw_arrow(ax, (11.9, 2.42), (11.9, 1.93))
    draw_arrow(ax, (11.9, 1.27), (11.9, 0.75))

    plt.title("Figure 12. Architecture of Model M5 (SE-VoiceResNet SOTA with Channel Attention & GAP)", fontsize=11.5, fontweight='bold', color="#0D2538", y=-0.08)
    plt.tight_layout()
    plt.savefig("ADL/results/architectures/arch_M5_SE_VoiceResNet.png", bbox_inches='tight', dpi=300)
    plt.close()

# =========================================================================
# 6. DETAILED ZOOM: SE Channel Attention Mechanism
# =========================================================================
def plot_se_mechanism():
    fig, ax = plt.subplots(figsize=(11.5, 3.8), dpi=300)
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-0.8, 3.2)
    ax.axis('off')

    # Input feature map X
    draw_box(ax, 0.8, 1.5, 1.4, 1.0, "Input Tensor\nX in R^(C x H x W)\n(e.g., 128 x 32 x 16)", C_CONV, fontsize=7.2)

    # Squeeze
    draw_box(ax, 3.0, 1.5, 1.8, 1.0, "Squeeze Operator (F_sq)\nGlobal Avg Pooling (1x1)\nz_c = 1/(HxW) sum(u_c)\n-> (C x 1 x 1)", C_GAP, fontsize=7.2)

    # Excitation
    draw_container(ax, 6.2, 1.5, 3.2, 1.6, "Excitation Operator (F_ex)")
    draw_box(ax, 5.3, 1.5, 1.3, 0.8, "Linear (C -> C/r)\n+ ReLU", C_DENSE, fontsize=7.0)
    draw_box(ax, 7.1, 1.5, 1.4, 0.8, "Linear (C/r -> C)\n+ Sigmoid", C_DENSE, fontsize=7.0)
    draw_arrow(ax, (6.0, 1.5), (6.4, 1.5))

    # Scale
    draw_box(ax, 9.2, 1.5, 1.6, 1.0, "Scale Operator (F_scale)\nChannel-wise Multiply\nx_tilde = s_c * u_c", C_SE, fontsize=7.2)

    # Output feature map
    draw_box(ax, 11.0, 1.5, 1.0, 0.8, "Calibrated\nOutput\nX_tilde", C_OUTPUT, textcolor="#000000", fontsize=7.2)

    # Connecting arrows
    draw_arrow(ax, (1.5, 1.5), (2.1, 1.5))
    draw_arrow(ax, (3.9, 1.5), (4.6, 1.5))
    draw_arrow(ax, (7.8, 1.5), (8.4, 1.5))
    draw_arrow(ax, (10.0, 1.5), (10.5, 1.5))

    plt.title("Figure 13. Squeeze-and-Excitation (SE) Channel Attention Mechanism", fontsize=11.5, fontweight='bold', color="#0D2538", y=-0.08)
    plt.tight_layout()
    plt.savefig("ADL/results/architectures/arch_SE_Block_Mechanism.png", bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_m1()
    plot_m2()
    plot_m3()
    plot_m4()
    plot_m5()
    plot_se_mechanism()
    print("[SUCCESS] All 5 model architectures + SE Mechanism plotted successfully in ADL/results/architectures/!")
