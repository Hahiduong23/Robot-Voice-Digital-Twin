import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, PathPatch
import matplotlib.path as mpath

os.makedirs("ADL/results/architectures", exist_ok=True)

# -------------------------------------------------------------
# Color Palette strictly matching Figure 8 Reference
# -------------------------------------------------------------
CLR_YELLOW  = "#F5A623"  # Input & Output boxes (Gold / Amber)
CLR_GREEN   = "#27AE60"  # Preprocessing / Stem / Transform (Emerald Green)
CLR_BLUE    = "#4A90E2"  # Conv2d / Core Layer (Soft Blue)
CLR_BN      = "#2E7D32"  # BatchNorm2d (Forest Green)
CLR_POOL    = "#5C9CE6"  # MaxPool2d (Light Blue)
CLR_TEAL    = "#00838F"  # GAP / ResAdd (Deep Teal)
CLR_NAVY    = "#0D233A"  # Dropout layer (Dark Navy Blue - matching Fig 8)
CLR_PURPLE  = "#6A1B9A"  # Fully connected layers (Purple - matching Fig 8)
CLR_PINK    = "#D81B60"  # Squeeze-and-Excitation Attention (Magenta Rose)
CLR_BORDER  = "#E65100"  # Container outline (Burnt Orange - matching Fig 8)
CLR_ARROW   = "#1976D2"  # Connecting arrows (Crisp Azure Blue)

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

class Block:
    def __init__(self, cx, cy, w, h, text, facecolor, textcolor="white", fontsize=7.8, bold=True, edgecolor="#1A252C", pad=0.15):
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.pad = pad
        self.text = text
        self.facecolor = facecolor
        self.textcolor = textcolor
        self.fontsize = fontsize
        self.bold = bold
        self.edgecolor = edgecolor

    # EXACT visible outer boundaries taking into account pad
    @property
    def left(self):   return (self.cx - self.w/2 - self.pad, self.cy)
    @property
    def right(self):  return (self.cx + self.w/2 + self.pad, self.cy)
    @property
    def top(self):    return (self.cx, self.cy + self.h/2 + self.pad)
    @property
    def bottom(self): return (self.cx, self.cy - self.h/2 - self.pad)

    def draw(self, ax):
        weight = "bold" if self.bold else "normal"
        box = FancyBboxPatch((self.cx - self.w/2, self.cy - self.h/2), self.w, self.h,
                             boxstyle=f"round,pad={self.pad},rounding_size=0.15",
                             facecolor=self.facecolor, edgecolor=self.edgecolor, linewidth=1.4,
                             zorder=3)
        ax.add_patch(box)
        ax.text(self.cx, self.cy, self.text, ha='center', va='center', color=self.textcolor,
                fontsize=self.fontsize, fontweight=weight, zorder=5, multialignment='center')

def draw_container(ax, cx, cy, w, h, label="", border_color=CLR_BORDER, pad=0.22):
    box = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                         boxstyle=f"round,pad={pad},rounding_size=0.25",
                         facecolor="#FFFBF5", edgecolor=border_color, linewidth=1.6,
                         zorder=1, alpha=0.95)
    ax.add_patch(box)
    
    if label:
        badge_w = min(w * 0.90, max(len(label) * 0.11, 2.0))
        badge_h = 0.44
        badge_y = cy + h/2 + pad + 0.08
        badge = FancyBboxPatch((cx - badge_w/2, badge_y - badge_h/2), badge_w, badge_h,
                               boxstyle="round,pad=0.06,rounding_size=0.1",
                               facecolor="#FFE0B2", edgecolor=border_color, linewidth=1.2,
                               zorder=4)
        ax.add_patch(badge)
        ax.text(cx, badge_y, label, ha='center', va='center', color="#BF360C",
                fontsize=8.8, fontweight='bold', zorder=5)

    # Return visible left & right coordinates of container
    return (cx - w/2 - pad, cx + w/2 + pad)

def arrow_straight(ax, p1, p2, color=CLR_ARROW, lw=2.2):
    """
    Arrow with high zorder=6 and shrink=3 so it is ALWAYS drawn on top,
    starting cleanly outside box 1 and ending cleanly outside box 2.
    """
    ax.annotate("", xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=18, shrinkA=3, shrinkB=3),
                zorder=6)

def arrow_orthogonal_s(ax, p1, p2, color=CLR_ARROW, lw=2.2):
    x1, y1 = p1
    x2, y2 = p2
    x_mid = (x1 + x2) / 2
    path_data = [
        (mpath.Path.MOVETO, (x1 + 0.02, y1)),
        (mpath.Path.LINETO, (x_mid, y1)),
        (mpath.Path.LINETO, (x_mid, y2)),
        (mpath.Path.LINETO, (x2 - 0.05, y2))
    ]
    codes, verts = zip(*path_data)
    path = mpath.Path(verts, codes)
    patch = PathPatch(path, facecolor='none', edgecolor=color, lw=lw, zorder=6)
    ax.add_patch(patch)
    ax.annotate("", xy=p2, xytext=(x2 - 0.18, y2),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=18, shrinkA=0, shrinkB=3),
                zorder=6)

def arrow_top_route(ax, p1, p2, y_top, color=CLR_ARROW, lw=2.2):
    x1, y1 = p1
    x2, y2 = p2
    x_mid = x1 + (x2 - x1) * 0.45
    path_data = [
        (mpath.Path.MOVETO, (x1 + 0.02, y1)),
        (mpath.Path.LINETO, (x_mid, y1)),
        (mpath.Path.LINETO, (x_mid, y_top)),
        (mpath.Path.LINETO, (x2, y_top)),
        (mpath.Path.LINETO, (x2, y2 + 0.05))
    ]
    codes, verts = zip(*path_data)
    path = mpath.Path(verts, codes)
    patch = PathPatch(path, facecolor='none', edgecolor=color, lw=lw, zorder=6)
    ax.add_patch(patch)
    ax.annotate("", xy=p2, xytext=(x2, y2 + 0.18),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=18, shrinkA=0, shrinkB=3),
                zorder=6)

# =========================================================================
# 1. MODEL M1: Baseline CNN (No bottom title, expansive arrows)
# =========================================================================
def generate_m1():
    fig, ax = plt.subplots(figsize=(18.0, 6.8), dpi=300)
    ax.set_xlim(-0.6, 20.0)
    ax.set_ylim(-0.4, 7.2)
    ax.axis('off')

    b_input = Block(1.5, 5.0, 1.8, 1.0, "Input\nInput shape:\n(1, 128, 63)", CLR_YELLOW, textcolor="#000000", fontsize=8.2)
    b_prep  = Block(1.5, 2.6, 1.8, 0.85, "Mel-Spectrogram\n(128 x 63)", CLR_GREEN, fontsize=8.0)

    # Conv Blocks (Generous vertical spacing inside container)
    b_c1 = Block(4.9, 4.4, 2.0, 0.85, "Conv2d layer\n(32 filters, 3x3)\nactivation: relu", CLR_BLUE, fontsize=7.5)
    b_p1 = Block(4.9, 2.2, 2.0, 0.75, "MaxPool2d layer\n(output: 32, 64, 31)", CLR_POOL, fontsize=7.5)

    b_c2 = Block(8.5, 4.4, 2.0, 0.85, "Conv2d layer\n(64 filters, 3x3)\nactivation: relu", CLR_BLUE, fontsize=7.5)
    b_p2 = Block(8.5, 2.2, 2.0, 0.75, "MaxPool2d layer\n(output: 64, 32, 15)", CLR_POOL, fontsize=7.5)

    b_c3 = Block(12.1, 4.4, 2.0, 0.85, "Conv2d layer\n(128 filters, 3x3)\nactivation: relu", CLR_BLUE, fontsize=7.5)
    b_p3 = Block(12.1, 2.2, 2.0, 0.75, "MaxPool2d layer\n(output: 128, 16, 7)", CLR_POOL, fontsize=7.5)

    b_flat = Block(14.9, 3.3, 1.6, 0.85, "Flatten layer\n(dim: 14,336)", "#455A64", fontsize=7.5)

    b_fc1 = Block(17.8, 5.2, 2.1, 0.9, "Fully connected layer\n(output: 128,\nactivation: relu)", CLR_PURPLE, fontsize=7.5)
    b_fc2 = Block(17.8, 3.0, 2.1, 0.9, "Fully connected layer\n(output: 7,\nactivation: softmax)", CLR_PURPLE, fontsize=7.5)
    b_out = Block(17.8, 1.0, 1.6, 0.75, "Output\n(7 classes)", CLR_YELLOW, textcolor="#000000", fontsize=8.2)

    c1_l, c1_r = draw_container(ax, 4.9, 3.3, 2.5, 3.6, "Conv Block 1")
    c2_l, c2_r = draw_container(ax, 8.5, 3.3, 2.5, 3.6, "Conv Block 2")
    c3_l, c3_r = draw_container(ax, 12.1, 3.3, 2.5, 3.6, "Conv Block 3")

    for b in [b_input, b_prep, b_c1, b_p1, b_c2, b_p2, b_c3, b_p3, b_flat, b_fc1, b_fc2, b_out]:
        b.draw(ax)

    arrow_straight(ax, b_input.bottom, b_prep.top)
    arrow_orthogonal_s(ax, b_prep.right, b_c1.left)

    arrow_straight(ax, b_c1.bottom, b_p1.top)
    arrow_straight(ax, b_c2.bottom, b_p2.top)
    arrow_straight(ax, b_c3.bottom, b_p3.top)

    # Clear arrows between containers
    arrow_straight(ax, (c1_r, 3.3), (c2_l, 3.3))
    arrow_straight(ax, (c2_r, 3.3), (c3_l, 3.3))
    arrow_straight(ax, (c3_r, 3.3), b_flat.left)

    arrow_top_route(ax, b_flat.right, b_fc1.top, y_top=6.4)
    arrow_straight(ax, b_fc1.bottom, b_fc2.top)
    arrow_straight(ax, b_fc2.bottom, b_out.top)

    plt.savefig("ADL/results/architectures/arch_M1_BaselineCNN.png", bbox_inches='tight', pad_inches=0.25, dpi=300)
    plt.close()

# =========================================================================
# 2. MODEL M2: Baseline + Dropout
# =========================================================================
def generate_m2():
    fig, ax = plt.subplots(figsize=(18.0, 7.2), dpi=300)
    ax.set_xlim(-0.6, 20.0)
    ax.set_ylim(-0.4, 7.4)
    ax.axis('off')

    b_input = Block(1.5, 5.2, 1.8, 1.0, "Input\nInput shape:\n(1, 128, 63)", CLR_YELLOW, textcolor="#000000", fontsize=8.2)
    b_prep  = Block(1.5, 2.8, 1.8, 0.85, "Mel-Spectrogram\n(128 x 63)", CLR_GREEN, fontsize=8.0)

    b_c1 = Block(4.9, 4.4, 2.0, 0.85, "Conv2d layer\n(32 filters, 3x3)\nactivation: relu", CLR_BLUE, fontsize=7.5)
    b_p1 = Block(4.9, 2.2, 2.0, 0.75, "MaxPool2d layer\n(output: 32, 64, 31)", CLR_POOL, fontsize=7.5)

    b_c2 = Block(8.5, 4.4, 2.0, 0.85, "Conv2d layer\n(64 filters, 3x3)\nactivation: relu", CLR_BLUE, fontsize=7.5)
    b_p2 = Block(8.5, 2.2, 2.0, 0.75, "MaxPool2d layer\n(output: 64, 32, 15)", CLR_POOL, fontsize=7.5)

    b_c3 = Block(12.1, 4.4, 2.0, 0.85, "Conv2d layer\n(128 filters, 3x3)\nactivation: relu", CLR_BLUE, fontsize=7.5)
    b_p3 = Block(12.1, 2.2, 2.0, 0.75, "MaxPool2d layer\n(output: 128, 16, 7)", CLR_POOL, fontsize=7.5)

    b_flat = Block(14.9, 3.3, 1.6, 0.85, "Flatten layer\n(dim: 14,336)", "#455A64", fontsize=7.5)

    b_fc1  = Block(17.8, 5.5, 2.1, 0.8, "Fully connected layer\n(output: 128, relu)", CLR_PURPLE, fontsize=7.5)
    b_drop = Block(17.8, 4.0, 1.9, 0.7, "Dropout layer\n(rate = 0.4)", CLR_NAVY, fontsize=7.8, edgecolor="#FFD54F")
    b_fc2  = Block(17.8, 2.5, 2.1, 0.8, "Fully connected layer\n(output: 7, softmax)", CLR_PURPLE, fontsize=7.5)
    b_out  = Block(17.8, 0.9, 1.6, 0.7, "Output\n(7 classes)", CLR_YELLOW, textcolor="#000000", fontsize=8.2)

    c1_l, c1_r = draw_container(ax, 4.9, 3.3, 2.5, 3.6, "Conv Block 1")
    c2_l, c2_r = draw_container(ax, 8.5, 3.3, 2.5, 3.6, "Conv Block 2")
    c3_l, c3_r = draw_container(ax, 12.1, 3.3, 2.5, 3.6, "Conv Block 3")

    for b in [b_input, b_prep, b_c1, b_p1, b_c2, b_p2, b_c3, b_p3, b_flat, b_fc1, b_drop, b_fc2, b_out]:
        b.draw(ax)

    arrow_straight(ax, b_input.bottom, b_prep.top)
    arrow_orthogonal_s(ax, b_prep.right, b_c1.left)

    arrow_straight(ax, b_c1.bottom, b_p1.top)
    arrow_straight(ax, b_c2.bottom, b_p2.top)
    arrow_straight(ax, b_c3.bottom, b_p3.top)

    arrow_straight(ax, (c1_r, 3.3), (c2_l, 3.3))
    arrow_straight(ax, (c2_r, 3.3), (c3_l, 3.3))
    arrow_straight(ax, (c3_r, 3.3), b_flat.left)

    arrow_top_route(ax, b_flat.right, b_fc1.top, y_top=6.6)

    arrow_straight(ax, b_fc1.bottom, b_drop.top)
    arrow_straight(ax, b_drop.bottom, b_fc2.top)
    arrow_straight(ax, b_fc2.bottom, b_out.top)

    plt.savefig("ADL/results/architectures/arch_M2_DropoutCNN.png", bbox_inches='tight', pad_inches=0.25, dpi=300)
    plt.close()

# =========================================================================
# 3. MODEL M3: Baseline + Dropout + BatchNorm2d
# =========================================================================
def generate_m3():
    fig, ax = plt.subplots(figsize=(18.5, 8.0), dpi=300)
    ax.set_xlim(-0.6, 20.2)
    ax.set_ylim(-0.4, 8.0)
    ax.axis('off')

    b_input = Block(1.5, 5.6, 1.8, 1.0, "Input\nInput shape:\n(1, 128, 63)", CLR_YELLOW, textcolor="#000000", fontsize=8.2)
    b_prep  = Block(1.5, 3.4, 1.8, 0.85, "Mel-Spectrogram\n(128 x 63)", CLR_GREEN, fontsize=8.0)

    # 3 stacked layers inside container with generous vertical gap
    b_c1  = Block(4.9, 5.6, 2.0, 0.75, "Conv2d layer\n(32 filters, 3x3)", CLR_BLUE, fontsize=7.5)
    b_bn1 = Block(4.9, 4.0, 2.0, 0.75, "BatchNorm2d (32)\n+ ReLU activation", CLR_BN, fontsize=7.5)
    b_p1  = Block(4.9, 2.4, 2.0, 0.75, "MaxPool2d layer\n(output: 32, 64, 31)", CLR_POOL, fontsize=7.5)

    b_c2  = Block(8.5, 5.6, 2.0, 0.75, "Conv2d layer\n(64 filters, 3x3)", CLR_BLUE, fontsize=7.5)
    b_bn2 = Block(8.5, 4.0, 2.0, 0.75, "BatchNorm2d (64)\n+ ReLU activation", CLR_BN, fontsize=7.5)
    b_p2  = Block(8.5, 2.4, 2.0, 0.75, "MaxPool2d layer\n(output: 64, 32, 15)", CLR_POOL, fontsize=7.5)

    b_c3  = Block(12.1, 5.6, 2.0, 0.75, "Conv2d layer\n(128 filters, 3x3)", CLR_BLUE, fontsize=7.5)
    b_bn3 = Block(12.1, 4.0, 2.0, 0.75, "BatchNorm2d (128)\n+ ReLU activation", CLR_BN, fontsize=7.5)
    b_p3  = Block(12.1, 2.4, 2.0, 0.75, "MaxPool2d layer\n(output: 128, 16, 7)", CLR_POOL, fontsize=7.5)

    b_flat = Block(14.9, 4.0, 1.6, 0.85, "Flatten layer\n(dim: 14,336)", "#455A64", fontsize=7.5)

    b_fc1  = Block(17.8, 6.0, 2.1, 0.8, "Fully connected layer\n(output: 128, relu)", CLR_PURPLE, fontsize=7.5)
    b_drop = Block(17.8, 4.5, 1.9, 0.7, "Dropout layer\n(rate = 0.4)", CLR_NAVY, fontsize=7.8)
    b_fc2  = Block(17.8, 3.0, 2.1, 0.8, "Fully connected layer\n(output: 7, softmax)", CLR_PURPLE, fontsize=7.5)
    b_out  = Block(17.8, 1.4, 1.6, 0.7, "Output\n(7 classes)", CLR_YELLOW, textcolor="#000000", fontsize=8.2)

    c1_l, c1_r = draw_container(ax, 4.9, 4.0, 2.5, 4.8, "Conv-BN Block 1")
    c2_l, c2_r = draw_container(ax, 8.5, 4.0, 2.5, 4.8, "Conv-BN Block 2")
    c3_l, c3_r = draw_container(ax, 12.1, 4.0, 2.5, 4.8, "Conv-BN Block 3")

    for b in [b_input, b_prep, b_c1, b_bn1, b_p1, b_c2, b_bn2, b_p2, b_c3, b_bn3, b_p3, b_flat, b_fc1, b_drop, b_fc2, b_out]:
        b.draw(ax)

    arrow_straight(ax, b_input.bottom, b_prep.top)
    arrow_orthogonal_s(ax, b_prep.right, b_c1.left)

    arrow_straight(ax, b_c1.bottom, b_bn1.top)
    arrow_straight(ax, b_bn1.bottom, b_p1.top)

    arrow_straight(ax, b_c2.bottom, b_bn2.top)
    arrow_straight(ax, b_bn2.bottom, b_p2.top)

    arrow_straight(ax, b_c3.bottom, b_bn3.top)
    arrow_straight(ax, b_bn3.bottom, b_p3.top)

    arrow_straight(ax, (c1_r, 4.0), (c2_l, 4.0))
    arrow_straight(ax, (c2_r, 4.0), (c3_l, 4.0))
    arrow_straight(ax, (c3_r, 4.0), b_flat.left)

    arrow_top_route(ax, b_flat.right, b_fc1.top, y_top=7.2)

    arrow_straight(ax, b_fc1.bottom, b_drop.top)
    arrow_straight(ax, b_drop.bottom, b_fc2.top)
    arrow_straight(ax, b_fc2.bottom, b_out.top)

    plt.savefig("ADL/results/architectures/arch_M3_BatchNormCNN.png", bbox_inches='tight', pad_inches=0.25, dpi=300)
    plt.close()

# =========================================================================
# 4. MODEL M4: VoiceResNet (Residual Learning + GAP)
# =========================================================================
def generate_m4():
    fig, ax = plt.subplots(figsize=(19.0, 8.0), dpi=300)
    ax.set_xlim(-0.6, 20.6)
    ax.set_ylim(-0.4, 8.0)
    ax.axis('off')

    b_input = Block(1.5, 5.6, 1.8, 1.0, "Input\nInput shape:\n(1, 128, 63)", CLR_YELLOW, textcolor="#000000", fontsize=8.2)
    b_stem  = Block(1.5, 3.4, 1.8, 0.95, "Stem Layer\nConv2d(32) + BN\n+ ReLU", CLR_GREEN, fontsize=7.8)

    b_r1_c1  = Block(5.0, 5.7, 2.0, 0.72, "Conv-BN-ReLU\n(32 filters, 3x3)", CLR_BLUE, fontsize=7.5)
    b_r1_c2  = Block(5.0, 4.1, 2.0, 0.72, "Conv-BN layer\n(32 filters, 3x3)", CLR_BLUE, fontsize=7.5)
    b_r1_add = Block(5.0, 2.5, 2.0, 0.72, "Add [F(x) + x]\n+ ReLU (32,128,63)", CLR_TEAL, fontsize=7.5)

    b_r2_c1  = Block(8.7, 5.7, 2.0, 0.72, "Conv-BN-ReLU\n(64 flt, 3x3, s=2)", CLR_BLUE, fontsize=7.5)
    b_r2_c2  = Block(8.7, 4.1, 2.0, 0.72, "Conv-BN layer\n(64 filters, 3x3)", CLR_BLUE, fontsize=7.5)
    b_r2_add = Block(8.7, 2.5, 2.0, 0.72, "Add [F(x) + Ws*x]\n+ ReLU (64,64,32)", CLR_TEAL, fontsize=7.5)

    b_r3_c1  = Block(12.4, 5.7, 2.0, 0.72, "Conv-BN-ReLU\n(128 flt, 3x3, s=2)", CLR_BLUE, fontsize=7.5)
    b_r3_c2  = Block(12.4, 4.1, 2.0, 0.72, "Conv-BN layer\n(128 filters, 3x3)", CLR_BLUE, fontsize=7.5)
    b_r3_add = Block(12.4, 2.5, 2.0, 0.72, "Add [F(x) + Ws*x]\n+ ReLU (128,32,16)", CLR_TEAL, fontsize=7.5)

    b_gap = Block(15.3, 4.1, 1.9, 1.0, "Global Avg Pool\nAdaptiveAvgPool\n(output: 128)", CLR_TEAL, fontsize=7.8, edgecolor="#FFD54F")

    b_drop = Block(18.0, 5.7, 1.9, 0.72, "Dropout layer\n(rate = 0.15)", CLR_NAVY, fontsize=7.8)
    b_fc   = Block(18.0, 4.0, 2.1, 0.85, "Fully connected\n(out: 7, softmax)", CLR_PURPLE, fontsize=7.8)
    b_out  = Block(18.0, 2.2, 1.6, 0.72, "Output\n(7 classes)", CLR_YELLOW, textcolor="#000000", fontsize=8.2)

    c1_l, c1_r = draw_container(ax, 5.0, 4.1, 2.6, 4.8, "Residual Block 1 (s=1)")
    c2_l, c2_r = draw_container(ax, 8.7, 4.1, 2.6, 4.8, "Residual Block 2 (s=2)")
    c3_l, c3_r = draw_container(ax, 12.4, 4.1, 2.6, 4.8, "Residual Block 3 (s=2)")

    all_blocks = [b_input, b_stem, b_r1_c1, b_r1_c2, b_r1_add, b_r2_c1, b_r2_c2, b_r2_add,
                  b_r3_c1, b_r3_c2, b_r3_add, b_gap, b_drop, b_fc, b_out]
    for b in all_blocks:
        b.draw(ax)

    arrow_straight(ax, b_input.bottom, b_stem.top)
    arrow_orthogonal_s(ax, b_stem.right, b_r1_c1.left)

    for (c1, c2, add) in [(b_r1_c1, b_r1_c2, b_r1_add), (b_r2_c1, b_r2_c2, b_r2_add), (b_r3_c1, b_r3_c2, b_r3_add)]:
        arrow_straight(ax, c1.bottom, c2.top)
        arrow_straight(ax, c2.bottom, add.top)

    def draw_shortcut(cx, label="Shortcut"):
        p_start = (cx - 1.05 - 0.15, 5.7)
        p_end   = (cx - 1.05 - 0.15, 2.5)
        path_data = [
            (mpath.Path.MOVETO, p_start),
            (mpath.Path.LINETO, (cx - 1.5, 5.7)),
            (mpath.Path.LINETO, (cx - 1.5, 2.5)),
            (mpath.Path.LINETO, (cx - 1.25, 2.5))
        ]
        codes, verts = zip(*path_data)
        path = mpath.Path(verts, codes)
        patch = PathPatch(path, facecolor='none', edgecolor="#E65100", lw=2.0, linestyle="--", zorder=6)
        ax.add_patch(patch)
        ax.annotate("", xy=(cx - 1.2, 2.5), xytext=(cx - 1.35, 2.5),
                    arrowprops=dict(arrowstyle="-|>", color="#E65100", lw=2.0, mutation_scale=16), zorder=6)
        ax.text(cx - 1.6, 4.1, label, rotation=90, ha='right', va='center', fontsize=7.2, color="#BF360C", fontweight='bold')

    draw_shortcut(5.0, label="Identity x")
    draw_shortcut(8.7, label="1x1 Conv (s=2)")
    draw_shortcut(12.4, label="1x1 Conv (s=2)")

    arrow_straight(ax, (c1_r, 4.1), (c2_l, 4.1))
    arrow_straight(ax, (c2_r, 4.1), (c3_l, 4.1))
    arrow_straight(ax, (c3_r, 4.1), b_gap.left)

    arrow_top_route(ax, b_gap.right, b_drop.top, y_top=7.1)
    arrow_straight(ax, b_drop.bottom, b_fc.top)
    arrow_straight(ax, b_fc.bottom, b_out.top)

    plt.savefig("ADL/results/architectures/arch_M4_VoiceResNet.png", bbox_inches='tight', pad_inches=0.25, dpi=300)
    plt.close()

# =========================================================================
# 5. MODEL M5: SE_VoiceResNet (SOTA with Channel Attention)
# =========================================================================
def generate_m5():
    fig, ax = plt.subplots(figsize=(19.5, 8.8), dpi=300)
    ax.set_xlim(-0.6, 21.0)
    ax.set_ylim(-0.4, 8.8)
    ax.axis('off')

    b_input = Block(1.5, 6.4, 1.8, 1.0, "Input\nInput shape:\n(1, 128, 63)", CLR_YELLOW, textcolor="#000000", fontsize=8.2)
    b_stem  = Block(1.5, 4.0, 1.8, 0.95, "Stem Layer\nConv2d(32) + BN\n+ ReLU", CLR_GREEN, fontsize=7.8)

    # 4 vertically stacked layers in SE-ResBlock with generous gap
    b_r1_c1  = Block(5.1, 6.8, 2.1, 0.68, "Conv-BN-ReLU\n(32 filters, 3x3)", CLR_BLUE, fontsize=7.4)
    b_r1_c2  = Block(5.1, 5.3, 2.1, 0.68, "Conv-BN layer\n(32 filters, 3x3)", CLR_BLUE, fontsize=7.4)
    b_r1_se  = Block(5.1, 3.8, 2.1, 0.74, "SE Attention Block\nSqueeze-&-Excitation", CLR_PINK, fontsize=7.6, edgecolor="#FFD54F")
    b_r1_add = Block(5.1, 2.3, 2.1, 0.68, "Add [F_se(x) + x]\n+ ReLU (32,128,63)", CLR_TEAL, fontsize=7.4)

    b_r2_c1  = Block(9.0, 6.8, 2.1, 0.68, "Conv-BN-ReLU\n(64 flt, 3x3, s=2)", CLR_BLUE, fontsize=7.4)
    b_r2_c2  = Block(9.0, 5.3, 2.1, 0.68, "Conv-BN layer\n(64 filters, 3x3)", CLR_BLUE, fontsize=7.4)
    b_r2_se  = Block(9.0, 3.8, 2.1, 0.74, "SE Attention Block\nSqueeze-&-Excitation", CLR_PINK, fontsize=7.6, edgecolor="#FFD54F")
    b_r2_add = Block(9.0, 2.3, 2.1, 0.68, "Add [F_se(x) + Ws*x]\n+ ReLU (64,64,32)", CLR_TEAL, fontsize=7.4)

    b_r3_c1  = Block(12.9, 6.8, 2.1, 0.68, "Conv-BN-ReLU\n(128 flt, 3x3, s=2)", CLR_BLUE, fontsize=7.4)
    b_r3_c2  = Block(12.9, 5.3, 2.1, 0.68, "Conv-BN layer\n(128 filters, 3x3)", CLR_BLUE, fontsize=7.4)
    b_r3_se  = Block(12.9, 3.8, 2.1, 0.74, "SE Attention Block\nSqueeze-&-Excitation", CLR_PINK, fontsize=7.6, edgecolor="#FFD54F")
    b_r3_add = Block(12.9, 2.3, 2.1, 0.68, "Add [F_se(x) + Ws*x]\n+ ReLU (128,32,16)", CLR_TEAL, fontsize=7.4)

    b_gap = Block(16.0, 4.55, 1.9, 1.0, "Global Avg Pool\nAdaptiveAvgPool\n(output: 128)", CLR_TEAL, fontsize=7.8, edgecolor="#FFD54F")

    b_drop = Block(18.8, 6.4, 1.9, 0.72, "Dropout layer\n(rate = 0.15)", CLR_NAVY, fontsize=7.8)
    b_fc   = Block(18.8, 4.4, 2.1, 0.85, "Fully connected\n(out: 7, softmax)", CLR_PURPLE, fontsize=7.8)
    b_out  = Block(18.8, 2.4, 1.6, 0.72, "Output\n(7 classes)", CLR_YELLOW, textcolor="#000000", fontsize=8.2)

    c1_l, c1_r = draw_container(ax, 5.1, 4.55, 2.7, 5.8, "SE-Residual Block 1 (s=1)")
    c2_l, c2_r = draw_container(ax, 9.0, 4.55, 2.7, 5.8, "SE-Residual Block 2 (s=2)")
    c3_l, c3_r = draw_container(ax, 12.9, 4.55, 2.7, 5.8, "SE-Residual Block 3 (s=2)")

    all_blocks = [b_input, b_stem, b_r1_c1, b_r1_c2, b_r1_se, b_r1_add,
                  b_r2_c1, b_r2_c2, b_r2_se, b_r2_add,
                  b_r3_c1, b_r3_c2, b_r3_se, b_r3_add, b_gap, b_drop, b_fc, b_out]
    for b in all_blocks:
        b.draw(ax)

    arrow_straight(ax, b_input.bottom, b_stem.top)
    arrow_orthogonal_s(ax, b_stem.right, b_r1_c1.left)

    for (c1, c2, se, add) in [(b_r1_c1, b_r1_c2, b_r1_se, b_r1_add),
                               (b_r2_c1, b_r2_c2, b_r2_se, b_r2_add),
                               (b_r3_c1, b_r3_c2, b_r3_se, b_r3_add)]:
        arrow_straight(ax, c1.bottom, c2.top)
        arrow_straight(ax, c2.bottom, se.top)
        arrow_straight(ax, se.bottom, add.top)

    def draw_shortcut_m5(cx, label="Shortcut"):
        p_start = (cx - 1.1 - 0.15, 6.8)
        p_end   = (cx - 1.1 - 0.15, 2.3)
        path_data = [
            (mpath.Path.MOVETO, p_start),
            (mpath.Path.LINETO, (cx - 1.55, 6.8)),
            (mpath.Path.LINETO, (cx - 1.55, 2.3)),
            (mpath.Path.LINETO, (cx - 1.30, 2.3))
        ]
        codes, verts = zip(*path_data)
        path = mpath.Path(verts, codes)
        patch = PathPatch(path, facecolor='none', edgecolor="#E65100", lw=2.0, linestyle="--", zorder=6)
        ax.add_patch(patch)
        ax.annotate("", xy=(cx - 1.25, 2.3), xytext=(cx - 1.40, 2.3),
                    arrowprops=dict(arrowstyle="-|>", color="#E65100", lw=2.0, mutation_scale=16), zorder=6)
        ax.text(cx - 1.65, 4.55, label, rotation=90, ha='right', va='center', fontsize=7.2, color="#BF360C", fontweight='bold')

    draw_shortcut_m5(5.1, label="Identity x")
    draw_shortcut_m5(9.0, label="1x1 Conv (s=2)")
    draw_shortcut_m5(12.9, label="1x1 Conv (s=2)")

    arrow_straight(ax, (c1_r, 4.55), (c2_l, 4.55))
    arrow_straight(ax, (c2_r, 4.55), (c3_l, 4.55))
    arrow_straight(ax, (c3_r, 4.55), b_gap.left)

    arrow_top_route(ax, b_gap.right, b_drop.top, y_top=8.0)
    arrow_straight(ax, b_drop.bottom, b_fc.top)
    arrow_straight(ax, b_fc.bottom, b_out.top)

    plt.savefig("ADL/results/architectures/arch_M5_SE_VoiceResNet.png", bbox_inches='tight', pad_inches=0.25, dpi=300)
    plt.close()

# =========================================================================
# 6. SE CHANNEL ATTENTION MECHANISM (Detailed Zoom - Compact & Spacious)
# =========================================================================
def generate_se_detail():
    fig, ax = plt.subplots(figsize=(15.5, 4.2), dpi=300)
    ax.set_xlim(-0.6, 16.0)
    ax.set_ylim(-0.2, 3.8)
    ax.axis('off')

    b_x   = Block(1.4, 1.8, 1.6, 0.85, "Input Tensor\nX in R^(C x H x W)", CLR_BLUE, fontsize=7.5)
    b_sq  = Block(4.0, 1.8, 1.8, 0.85, "Squeeze (F_sq)\nGlobal Avg Pool\nz in R^(C x 1 x 1)", CLR_TEAL, fontsize=7.2)
    b_fc1 = Block(6.8, 1.8, 1.6, 0.78, "Dense (C -> C/r)\n+ ReLU (r=16)", CLR_PURPLE, fontsize=7.2)
    b_fc2 = Block(9.2, 1.8, 1.6, 0.78, "Dense (C/r -> C)\n+ Sigmoid (s in [0,1]^C)", CLR_PURPLE, fontsize=7.0)
    b_sc  = Block(12.0, 1.8, 1.8, 0.85, "Scale (F_scale)\nChannel Multiply\nx_tilde = s_c * u_c", CLR_PINK, fontsize=7.2, edgecolor="#FFD54F")
    b_out = Block(14.5, 1.8, 1.3, 0.8, "Calibrated\nOutput\nX_tilde", CLR_YELLOW, textcolor="#000000", fontsize=7.5)

    c_ex_l, c_ex_r = draw_container(ax, 8.0, 1.8, 4.6, 1.8, "Excitation Operator (F_ex)")

    for b in [b_x, b_sq, b_fc1, b_fc2, b_sc, b_out]:
        b.draw(ax)

    arrow_straight(ax, b_x.right, b_sq.left)
    arrow_straight(ax, b_sq.right, b_fc1.left)
    arrow_straight(ax, b_fc1.right, b_fc2.left)
    arrow_straight(ax, b_fc2.right, b_sc.left)
    arrow_straight(ax, b_sc.right, b_out.left)

    plt.savefig("ADL/results/architectures/arch_SE_Block_Mechanism.png", bbox_inches='tight', pad_inches=0.25, dpi=300)
    plt.close()

if __name__ == "__main__":
    generate_m1()
    generate_m2()
    generate_m3()
    generate_m4()
    generate_m5()
    generate_se_detail()
    print("[SUCCESS] All 6 architecture diagrams regenerated with titles removed and arrows 100% visible on top!")
