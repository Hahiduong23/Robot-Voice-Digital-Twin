# ==============================================================================
# FINAL TERM: MT-STYLED ARCHITECTURES & EDITABLE POWERPOINT GENERATOR
# Faithfully adheres to MT/results/architectures/ (containers, badges, curved shortcuts)
# 1. arch_BC_ResNet_Architecture.png (Matching arch_M5_SE_VoiceResNet.png)
# 2. arch_BC_ResBlock_Mechanism.png (Matching arch_SE_Block_Mechanism.png)
# 3. BC_ResNet_Architectures.pptx (Native editable PowerPoint shapes & text boxes)
# 4. All evaluation plots verified in 100% academic English
# ==============================================================================
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, PathPatch
import matplotlib.path as mpath

import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIR = os.path.join(BASE_DIR, "FN", "results")
ARCH_DIR = os.path.join(RES_DIR, "architectures")
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(ARCH_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

# ------------------------------------------------------------------------------
# EXACT COLOR PALETTE FROM MT/generate_pro_diagrams.py
# ------------------------------------------------------------------------------
CLR_YELLOW  = "#F5A623"  # Input & Output boxes (Gold / Amber)
CLR_GREEN   = "#27AE60"  # Preprocessing / Stem (Emerald Green)
CLR_BLUE    = "#4A90E2"  # Conv2d / Freq-Conv / Pointwise (Soft Blue)
CLR_BN      = "#2E7D32"  # BatchNorm2d (Forest Green)
CLR_POOL    = "#5C9CE6"  # MaxPool2d (Light Blue)
CLR_TEAL    = "#00838F"  # GAP / Residual Addition (Deep Teal)
CLR_NAVY    = "#0D233A"  # Dropout layer (Dark Navy Blue)
CLR_PURPLE  = "#6A1B9A"  # Fully connected layers (Purple)
CLR_PINK    = "#D81B60"  # Broadcast Frequency Attention (Magenta / Pink)
CLR_BORDER  = "#E65100"  # Container outline (Burnt Orange)
CLR_ARROW   = "#1976D2"  # Connecting arrows (Crisp Azure Blue)

class Block:
    def __init__(self, cx, cy, w, h, text, facecolor, textcolor="white", fontsize=7.6, bold=True, edgecolor="#1A252C", pad=0.14):
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
                             boxstyle=f"round,pad={self.pad},rounding_size=0.14",
                             facecolor=self.facecolor, edgecolor=self.edgecolor, linewidth=1.4,
                             zorder=4)
        ax.add_patch(box)
        ax.text(self.cx, self.cy, self.text, ha='center', va='center', color=self.textcolor,
                fontsize=self.fontsize, fontweight=weight, zorder=5, multialignment='center',
                linespacing=1.2)

def draw_container(ax, cx, cy, w, h, label="", border_color=CLR_BORDER):
    box = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                         boxstyle="round,pad=0.2,rounding_size=0.22",
                         facecolor="#FFFBF5", edgecolor=border_color, linewidth=1.6,
                         zorder=2, alpha=0.96)
    ax.add_patch(box)
    
    if label:
        badge_w = min(w * 0.92, max(len(label) * 0.11, 2.2))
        badge_h = 0.44
        badge_y = cy + h/2 + 0.08
        badge = FancyBboxPatch((cx - badge_w/2, badge_y - badge_h/2), badge_w, badge_h,
                               boxstyle="round,pad=0.06,rounding_size=0.1",
                               facecolor="#FFE0B2", edgecolor=border_color, linewidth=1.2,
                               zorder=4)
        ax.add_patch(badge)
        ax.text(cx, badge_y, label, ha='center', va='center', color="#BF360C",
                fontsize=8.5, fontweight='bold', zorder=5)
    return cx - w/2, cx + w/2

def arrow_straight(ax, p1, p2, color=CLR_ARROW, lw=2.0):
    ax.annotate("", xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=16, shrinkA=3, shrinkB=3),
                zorder=3)

def arrow_orthogonal_s(ax, p1, p2, color=CLR_ARROW, lw=2.0):
    mid_x = (p1[0] + p2[0]) / 2.0
    ax.annotate("", xy=p2, xytext=(mid_x, p2[1]),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=16, shrinkB=3),
                zorder=3)
    ax.plot([p1[0], mid_x, mid_x], [p1[1], p1[1], p2[1]], color=color, lw=lw, zorder=3)

def arrow_top_route(ax, p1, p2, y_top, color=CLR_ARROW, lw=2.0):
    ax.plot([p1[0], p1[0]+0.35, p1[0]+0.35], [p1[1], p1[1], y_top], color=color, lw=lw, zorder=3)
    ax.plot([p1[0]+0.35, p2[0]], [y_top, y_top], color=color, lw=lw, zorder=3)
    ax.annotate("", xy=p2, xytext=(p2[0], y_top),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=16, shrinkB=3),
                zorder=3)

# ------------------------------------------------------------------------------
# 1. GENERATE arch_BC_ResNet_Architecture.png (Matching MT M5 style)
# ------------------------------------------------------------------------------
def generate_m5_styled_bc_resnet_architecture():
    print("[1/3] Generating arch_BC_ResNet_Architecture.png matching MT/architectures style...")
    fig, ax = plt.subplots(figsize=(23.5, 9.2), dpi=300)
    ax.set_xlim(-0.6, 25.2)
    ax.set_ylim(-0.4, 9.2)
    ax.axis('off')

    # Column 1: Input & Stem
    b_input = Block(1.4, 6.6, 1.8, 1.05, "Input Audio\nMel-Spectrogram\n(1, 128, 63)", CLR_YELLOW, textcolor="#000000", fontsize=8.2)
    b_stem  = Block(1.4, 4.0, 1.8, 0.98, "Stem Layer\nConv2d(32) + BN\n+ ReLU (3x3, s=1)", CLR_GREEN, fontsize=7.8)

    # Column 2: BCResBlock 1 (Stride = 1, Channels = 32)
    b_s1_c1  = Block(5.0, 7.0, 2.1, 0.68, "Freq-Conv (3x1)\n(32 flt, groups=32)", CLR_BLUE, fontsize=7.4)
    b_s1_c2  = Block(5.0, 5.4, 2.1, 0.68, "Pointwise (1x1)\n(32 flt, + BN2)", CLR_BLUE, fontsize=7.4)
    b_s1_bfa = Block(5.0, 3.8, 2.1, 0.74, "Broadcast Attn (BFA)\nAvgPool F -> Conv 1x1", CLR_PINK, fontsize=7.4, edgecolor="#FFD54F")
    b_s1_add = Block(5.0, 2.2, 2.1, 0.68, "Add [F_bc(x) + x]\n+ ReLU (32,128,63)", CLR_TEAL, fontsize=7.4)

    # Column 3: BCResBlock 2 (Stride = 2, Channels = 64)
    b_s2_c1  = Block(8.9, 7.0, 2.1, 0.68, "Freq-Conv (3x1)\n(64 flt, s=2, groups=32)", CLR_BLUE, fontsize=7.4)
    b_s2_c2  = Block(8.9, 5.4, 2.1, 0.68, "Pointwise (1x1)\n(64 flt, + BN2)", CLR_BLUE, fontsize=7.4)
    b_s2_bfa = Block(8.9, 3.8, 2.1, 0.74, "Broadcast Attn (BFA)\nAvgPool F -> Conv 1x1", CLR_PINK, fontsize=7.4, edgecolor="#FFD54F")
    b_s2_add = Block(8.9, 2.2, 2.1, 0.68, "Add [F_bc(x) + Ws*x]\n+ ReLU (64,64,32)", CLR_TEAL, fontsize=7.4)

    # Column 4: BCResBlock 3 (Stride = 2, Channels = 128)
    b_s3_c1  = Block(12.8, 7.0, 2.1, 0.68, "Freq-Conv (3x1)\n(128 flt, s=2, groups=64)", CLR_BLUE, fontsize=7.4)
    b_s3_c2  = Block(12.8, 5.4, 2.1, 0.68, "Pointwise (1x1)\n(128 flt, + BN2)", CLR_BLUE, fontsize=7.4)
    b_s3_bfa = Block(12.8, 3.8, 2.1, 0.74, "Broadcast Attn (BFA)\nAvgPool F -> Conv 1x1", CLR_PINK, fontsize=7.4, edgecolor="#FFD54F")
    b_s3_add = Block(12.8, 2.2, 2.1, 0.68, "Add [F_bc(x) + Ws*x]\n+ ReLU (128,32,16)", CLR_TEAL, fontsize=7.4)

    # Column 5: BCResBlock 4 (Stride = 1, Channels = 128)
    b_s4_c1  = Block(16.7, 7.0, 2.1, 0.68, "Freq-Conv (3x1)\n(128 flt, s=1, groups=128)", CLR_BLUE, fontsize=7.4)
    b_s4_c2  = Block(16.7, 5.4, 2.1, 0.68, "Pointwise (1x1)\n(128 flt, + BN2)", CLR_BLUE, fontsize=7.4)
    b_s4_bfa = Block(16.7, 3.8, 2.1, 0.74, "Broadcast Attn (BFA)\nAvgPool F -> Conv 1x1", CLR_PINK, fontsize=7.4, edgecolor="#FFD54F")
    b_s4_add = Block(16.7, 2.2, 2.1, 0.68, "Add [F_bc(x) + x]\n+ ReLU (128,32,16)", CLR_TEAL, fontsize=7.4)

    # Column 6: GAP
    b_gap = Block(19.8, 4.6, 1.9, 1.05, "Global Avg Pool\nAdaptiveAvgPool2d\n(output: 128-d)", CLR_TEAL, fontsize=7.8, edgecolor="#FFD54F")

    # Column 7: Classifier Head
    b_drop = Block(22.8, 6.6, 2.0, 0.74, "Dropout layer\n(rate = 0.15)", CLR_NAVY, fontsize=7.8)
    b_fc   = Block(22.8, 4.6, 2.1, 0.88, "Fully connected\n(out: 7, softmax)", CLR_PURPLE, fontsize=7.8)
    b_out  = Block(22.8, 2.5, 1.8, 0.74, "Output\n(7 Command Classes)", CLR_YELLOW, textcolor="#000000", fontsize=8.0)

    # Containers
    c1_l, c1_r = draw_container(ax, 5.0, 4.6, 2.7, 6.0, "BC-Residual Block 1 (s=1)")
    c2_l, c2_r = draw_container(ax, 8.9, 4.6, 2.7, 6.0, "BC-Residual Block 2 (s=2)")
    c3_l, c3_r = draw_container(ax, 12.8, 4.6, 2.7, 6.0, "BC-Residual Block 3 (s=2)")
    c4_l, c4_r = draw_container(ax, 16.7, 4.6, 2.7, 6.0, "BC-Residual Block 4 (s=1)")

    all_blocks = [
        b_input, b_stem,
        b_s1_c1, b_s1_c2, b_s1_bfa, b_s1_add,
        b_s2_c1, b_s2_c2, b_s2_bfa, b_s2_add,
        b_s3_c1, b_s3_c2, b_s3_bfa, b_s3_add,
        b_s4_c1, b_s4_c2, b_s4_bfa, b_s4_add,
        b_gap, b_drop, b_fc, b_out
    ]
    for b in all_blocks:
        b.draw(ax)

    # Sequential Arrows
    arrow_straight(ax, b_input.bottom, b_stem.top)
    arrow_orthogonal_s(ax, b_stem.right, b_s1_c1.left)

    for (c1, c2, bfa, add) in [
        (b_s1_c1, b_s1_c2, b_s1_bfa, b_s1_add),
        (b_s2_c1, b_s2_c2, b_s2_bfa, b_s2_add),
        (b_s3_c1, b_s3_c2, b_s3_bfa, b_s3_add),
        (b_s4_c1, b_s4_c2, b_s4_bfa, b_s4_add)
    ]:
        arrow_straight(ax, c1.bottom, c2.top)
        arrow_straight(ax, c2.bottom, bfa.top)
        arrow_straight(ax, bfa.bottom, add.top)

    # Curved Residual Shortcuts
    def draw_shortcut(cx, label="Identity x"):
        path_data = [
            (mpath.Path.MOVETO, (cx - 1.25, 7.4)),
            (mpath.Path.CURVE3, (cx - 1.75, 4.6)),
            (mpath.Path.CURVE3, (cx - 1.25, 2.2))
        ]
        codes, verts = zip(*path_data)
        path = mpath.Path(verts, codes)
        patch = PathPatch(path, facecolor='none', edgecolor="#E65100", lw=2.0, linestyle="--", zorder=6)
        ax.add_patch(patch)
        ax.annotate("", xy=(cx - 1.15, 2.2), xytext=(cx - 1.35, 2.2),
                    arrowprops=dict(arrowstyle="-|>", color="#E65100", lw=2.0, mutation_scale=16), zorder=6)
        ax.text(cx - 1.7, 4.6, label, rotation=90, ha='right', va='center', fontsize=7.2, color="#BF360C", fontweight='bold')

    draw_shortcut(5.0, label="Identity x")
    draw_shortcut(8.9, label="1x1 Conv (s=2)")
    draw_shortcut(12.8, label="1x1 Conv (s=2)")
    draw_shortcut(16.7, label="Identity x")

    # Connectors between blocks
    arrow_straight(ax, (c1_r, 4.6), (c2_l, 4.6))
    arrow_straight(ax, (c2_r, 4.6), (c3_l, 4.6))
    arrow_straight(ax, (c3_r, 4.6), (c4_l, 4.6))
    arrow_straight(ax, (c4_r, 4.6), b_gap.left)

    arrow_top_route(ax, b_gap.right, b_drop.top, y_top=7.4)
    arrow_straight(ax, b_drop.bottom, b_fc.top)
    arrow_straight(ax, b_fc.bottom, b_out.top)

    # Figure Header
    plt.title("Figure: Qualcomm SOTA Broadcasted Residual Network (BC-ResNet) Architecture\n"
              "Total Parameters: 157,415 (~157K)  |  FLOPs: 83.41 M  |  CPU Latency: 4.84 ms  |  Trained on Data_FN (2,900 Samples)",
              fontsize=14, fontweight='bold', color="#102C57", pad=20)

    out_p = os.path.join(RES_DIR, "arch_BC_ResNet_Architecture.png")
    out_p_arch = os.path.join(ARCH_DIR, "arch_BC_ResNet_Architecture.png")
    plt.tight_layout()
    plt.savefig(out_p, bbox_inches='tight', pad_inches=0.25, dpi=300)
    plt.savefig(out_p_arch, bbox_inches='tight', pad_inches=0.25, dpi=300)
    plt.close()
    print(f"  -> Saved: {out_p}")
    print(f"  -> Saved: {out_p_arch}")

# ------------------------------------------------------------------------------
# 2. GENERATE arch_BC_ResBlock_Mechanism.png (Matching MT SE Mechanism style)
# ------------------------------------------------------------------------------
def generate_se_styled_bc_resblock_mechanism():
    print("[2/3] Generating arch_BC_ResBlock_Mechanism.png matching MT/architectures style...")
    fig, ax = plt.subplots(figsize=(16.8, 5.8), dpi=300)
    ax.set_xlim(-0.6, 17.6)
    ax.set_ylim(-0.6, 5.2)
    ax.axis('off')

    # Main blocks
    b_x    = Block(1.2, 2.8, 1.6, 0.95, "Input Tensor\nX in R^(C x F x T)", CLR_BLUE, fontsize=7.6)
    b_fcnv = Block(3.7, 2.8, 1.8, 0.95, "Frequency Conv\n3x1 (Groups=C)\n+ BN1 + ReLU", CLR_BLUE, fontsize=7.4)
    b_pcnv = Block(6.2, 2.8, 1.8, 0.95, "Pointwise Conv\n1x1 (C -> C')\n+ BN2", CLR_BLUE, fontsize=7.4)

    # Inside Broadcast Attention Operator Container
    b_favg = Block(9.2, 2.8, 1.7, 0.90, "Freq AvgPool (F_avg)\nz = (1/F)*sum_f x(f)\nz in R^(C' x 1 x T)", CLR_TEAL, fontsize=7.0)
    b_bcnv = Block(11.5, 2.8, 1.6, 0.85, "1x1 Conv + BN\nChannel Mixing", CLR_PURPLE, fontsize=7.2)
    b_sigm = Block(13.6, 2.8, 1.4, 0.85, "Sigmoid (σ)\ns in [0, 1]^(C'x1xT)", CLR_PURPLE, fontsize=7.2)

    # Multiply & Residual Addition
    b_mult = Block(15.7, 2.8, 1.1, 0.90, "Scale ⊗\nBroadcast\nacross F", CLR_PINK, fontsize=7.2, edgecolor="#FFD54F")
    b_out  = Block(17.2, 2.8, 1.2, 0.90, "Calibrated\nOutput\nX_tilde", CLR_YELLOW, textcolor="#000000", fontsize=7.4)

    # Shortcut branch at bottom
    b_sc   = Block(8.5, 0.6, 3.2, 0.85, "Residual Shortcut Branch\nIdentity (s=1)  or  1x1 Conv + BN (s=2)", CLR_TEAL, fontsize=7.4)

    # Container around Broadcast Attention
    draw_container(ax, 11.4, 2.8, 6.8, 2.3, "Broadcasted Frequency Attention (BFA) Operator", border_color=CLR_BORDER)

    all_blocks = [b_x, b_fcnv, b_pcnv, b_favg, b_bcnv, b_sigm, b_mult, b_out, b_sc]
    for b in all_blocks:
        b.draw(ax)

    # Connecting Arrows
    arrow_straight(ax, b_x.right, b_fcnv.left)
    arrow_straight(ax, b_fcnv.right, b_pcnv.left)
    arrow_straight(ax, b_pcnv.right, b_favg.left)
    arrow_straight(ax, b_favg.right, b_bcnv.left)
    arrow_straight(ax, b_bcnv.right, b_sigm.left)
    arrow_straight(ax, b_sigm.right, b_mult.left)
    arrow_straight(ax, b_mult.right, b_out.left)

    # Shortcut path routing
    ax.plot([b_x.cx, b_x.cx], [b_x.bottom[1], 0.6], color="#E65100", lw=2.0, linestyle="--", zorder=3)
    arrow_straight(ax, (b_x.cx, 0.6), b_sc.left, color="#E65100", lw=2.0)
    ax.plot([b_sc.right[0], b_out.cx], [0.6, 0.6], color="#E65100", lw=2.0, linestyle="--", zorder=3)
    arrow_straight(ax, (b_out.cx, 0.6), b_out.bottom, color="#E65100", lw=2.0)
    ax.text(8.5, 0.05, "⊕ Residual Addition before Final ReLU Activation", ha='center', va='bottom', fontsize=7.5, color="#BF360C", fontweight='bold')

    plt.title("Figure: Detailed Architecture of Broadcasted Residual Block (BC-ResBlock)\n"
              "Coupling Frequency-wise Separable Convolution with 1D Broadcasted Frequency Attention (BFA)",
              fontsize=12.5, fontweight='bold', color="#102C57", pad=15)

    out_p = os.path.join(RES_DIR, "arch_BC_ResBlock_Mechanism.png")
    out_p_arch = os.path.join(ARCH_DIR, "arch_BC_ResBlock_Mechanism.png")
    plt.tight_layout()
    plt.savefig(out_p, bbox_inches='tight', pad_inches=0.25, dpi=300)
    plt.savefig(out_p_arch, bbox_inches='tight', pad_inches=0.25, dpi=300)
    plt.close()
    print(f"  -> Saved: {out_p}")
    print(f"  -> Saved: {out_p_arch}")

# ------------------------------------------------------------------------------
# 3. GENERATE NATIVE EDITABLE POWERPOINT (BC_ResNet_Architectures.pptx)
# ------------------------------------------------------------------------------
def generate_native_editable_pptx():
    print("[3/3] Generating native editable PowerPoint file: BC_ResNet_Architectures.pptx...")
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def hex_to_rgb(hex_str):
        h = hex_str.lstrip('#')
        return RGBColor(*(int(h[i:i+2], 16) for i in (0, 2, 4)))

    # -----------------------------
    # SLIDE 1: BC-ResNet Full Model (MT-style with Containers)
    # -----------------------------
    slide1 = prs.slides.add_slide(blank_layout)

    # Title Box
    title_box1 = slide1.shapes.add_textbox(Inches(0.6), Inches(0.35), Inches(12.133), Inches(0.9))
    tf1 = title_box1.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "Qualcomm SOTA Broadcasted Residual Network (BC-ResNet) Architecture"
    p1.font.bold = True
    p1.font.size = Pt(20)
    p1.font.color.rgb = hex_to_rgb("#102C57")
    
    p2 = tf1.add_paragraph()
    p2.text = "157,415 Parameters (~157K)  |  FLOPs: 83.41 M  |  CPU Latency: 4.84 ms  |  Trained on Data_FN (2,900 Samples)"
    p2.font.size = Pt(11)
    p2.font.color.rgb = hex_to_rgb("#475569")

    # Helper function to add styled block
    def add_block_shape(slide, x, y, w, h, text, color_hex, text_color_hex="#FFFFFF", font_size=8.5):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        shape.fill.solid()
        shape.fill.fore_color.rgb = hex_to_rgb(color_hex)
        shape.line.color.rgb = hex_to_rgb("#1A252C")
        shape.line.width = Pt(1.2)
        tf = shape.text_frame
        tf.word_wrap = True
        tf.margin_top = Inches(0.04)
        tf.margin_bottom = Inches(0.04)
        tf.margin_left = Inches(0.04)
        tf.margin_right = Inches(0.04)
        p = tf.paragraphs[0]
        p.text = text
        p.font.size = Pt(font_size)
        p.font.bold = True
        p.font.color.rgb = hex_to_rgb(text_color_hex)
        p.alignment = PP_ALIGN.CENTER
        return shape

    # Helper function to add container
    def add_container_shape(slide, x, y, w, h, label):
        cont = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        cont.fill.solid()
        cont.fill.fore_color.rgb = hex_to_rgb("#FFFBF5")
        cont.line.color.rgb = hex_to_rgb(CLR_BORDER)
        cont.line.width = Pt(1.5)
        
        # Badge
        badge = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x + 0.1), Inches(y - 0.16), Inches(w - 0.2), Inches(0.32))
        badge.fill.solid()
        badge.fill.fore_color.rgb = hex_to_rgb("#FFE0B2")
        badge.line.color.rgb = hex_to_rgb(CLR_BORDER)
        badge.line.width = Pt(1.0)
        tf = badge.text_frame
        p = tf.paragraphs[0]
        p.text = label
        p.font.size = Pt(8.5)
        p.font.bold = True
        p.font.color.rgb = hex_to_rgb("#BF360C")
        p.alignment = PP_ALIGN.CENTER
        return cont

    # 1. Input & Stem
    add_block_shape(slide1, 0.6, 1.8, 1.4, 1.3, "Input Audio\nMel-Spectrogram\n(1, 128, 63)", CLR_YELLOW, "#000000", font_size=9.0)
    add_block_shape(slide1, 0.6, 3.8, 1.4, 1.3, "Stem Layer\nConv2d (3x3, c=32)\n+ BN + ReLU", CLR_GREEN, "#FFFFFF", font_size=8.5)

    # 2. Stage 1 Container & sub-blocks
    add_container_shape(slide1, 2.3, 1.7, 1.8, 4.3, "BC-ResBlock 1 (s=1)")
    add_block_shape(slide1, 2.45, 2.0, 1.5, 0.75, "Freq-Conv (3x1)\n(32 flt, g=32)", CLR_BLUE)
    add_block_shape(slide1, 2.45, 3.0, 1.5, 0.75, "Pointwise (1x1)\n(32 flt, + BN)", CLR_BLUE)
    add_block_shape(slide1, 2.45, 4.0, 1.5, 0.75, "Broadcast Attn\n(BFA Operator)", CLR_PINK)
    add_block_shape(slide1, 2.45, 5.0, 1.5, 0.75, "Add [F(x) + x]\n+ ReLU (32,128,63)", CLR_TEAL)

    # 3. Stage 2 Container & sub-blocks
    add_container_shape(slide1, 4.4, 1.7, 1.8, 4.3, "BC-ResBlock 2 (s=2)")
    add_block_shape(slide1, 4.55, 2.0, 1.5, 0.75, "Freq-Conv (3x1)\n(64 flt, s=2)", CLR_BLUE)
    add_block_shape(slide1, 4.55, 3.0, 1.5, 0.75, "Pointwise (1x1)\n(64 flt, + BN)", CLR_BLUE)
    add_block_shape(slide1, 4.55, 4.0, 1.5, 0.75, "Broadcast Attn\n(BFA Operator)", CLR_PINK)
    add_block_shape(slide1, 4.55, 5.0, 1.5, 0.75, "Add [F(x) + Ws*x]\n+ ReLU (64,64,32)", CLR_TEAL)

    # 4. Stage 3 Container & sub-blocks
    add_container_shape(slide1, 6.5, 1.7, 1.8, 4.3, "BC-ResBlock 3 (s=2)")
    add_block_shape(slide1, 6.65, 2.0, 1.5, 0.75, "Freq-Conv (3x1)\n(128 flt, s=2)", CLR_BLUE)
    add_block_shape(slide1, 6.65, 3.0, 1.5, 0.75, "Pointwise (1x1)\n(128 flt, + BN)", CLR_BLUE)
    add_block_shape(slide1, 6.65, 4.0, 1.5, 0.75, "Broadcast Attn\n(BFA Operator)", CLR_PINK)
    add_block_shape(slide1, 6.65, 5.0, 1.5, 0.75, "Add [F(x) + Ws*x]\n+ ReLU (128,32,16)", CLR_TEAL)

    # 5. Stage 4 Container & sub-blocks
    add_container_shape(slide1, 8.6, 1.7, 1.8, 4.3, "BC-ResBlock 4 (s=1)")
    add_block_shape(slide1, 8.75, 2.0, 1.5, 0.75, "Freq-Conv (3x1)\n(128 flt, s=1)", CLR_BLUE)
    add_block_shape(slide1, 8.75, 3.0, 1.5, 0.75, "Pointwise (1x1)\n(128 flt, + BN)", CLR_BLUE)
    add_block_shape(slide1, 8.75, 4.0, 1.5, 0.75, "Broadcast Attn\n(BFA Operator)", CLR_PINK)
    add_block_shape(slide1, 8.75, 5.0, 1.5, 0.75, "Add [F(x) + x]\n+ ReLU (128,32,16)", CLR_TEAL)

    # 6. Global Avg Pool
    add_block_shape(slide1, 10.6, 3.2, 1.1, 1.3, "Global Avg Pool\nAdaptivePool\n(128-d)", CLR_TEAL, "#FFFFFF", font_size=8.5)

    # 7. Head & Output
    add_block_shape(slide1, 11.9, 1.8, 1.1, 1.0, "Dropout\n(p = 0.15)", CLR_NAVY, "#FFFFFF", font_size=8.5)
    add_block_shape(slide1, 11.9, 3.3, 1.1, 1.1, "Dense FC\nLinear(128→7)\n+ Softmax", CLR_PURPLE, "#FFFFFF", font_size=8.5)
    add_block_shape(slide1, 11.9, 4.9, 1.1, 1.1, "Output\n(7 Command\nClasses)", CLR_YELLOW, "#000000", font_size=8.5)

    # Footer Notes
    note_box1 = slide1.shapes.add_textbox(Inches(0.6), Inches(6.5), Inches(12.133), Inches(0.5))
    tf_n1 = note_box1.text_frame
    p_n1 = tf_n1.paragraphs[0]
    p_n1.text = "Native PowerPoint Shapes: All blocks, labels, and text can be freely selected, moved, colored, or edited directly in PowerPoint."
    p_n1.font.size = Pt(10)
    p_n1.font.italic = True
    p_n1.font.color.rgb = hex_to_rgb("#64748B")

    # -----------------------------
    # SLIDE 2: Detailed BC-ResBlock Mechanism (Matching SE Mechanism)
    # -----------------------------
    slide2 = prs.slides.add_slide(blank_layout)

    # Title Box
    title_box2 = slide2.shapes.add_textbox(Inches(0.6), Inches(0.35), Inches(12.133), Inches(0.9))
    tf2 = title_box2.text_frame
    tf2.word_wrap = True
    p1 = tf2.paragraphs[0]
    p1.text = "Detailed Architecture of Broadcasted Residual Block (BC-ResBlock)"
    p1.font.bold = True
    p1.font.size = Pt(20)
    p1.font.color.rgb = hex_to_rgb("#102C57")
    
    p2 = tf2.add_paragraph()
    p2.text = "Coupling Frequency-wise Separable Convolution with 1D Broadcasted Frequency Attention (BFA)"
    p2.font.size = Pt(11)
    p2.font.color.rgb = hex_to_rgb("#475569")

    # Pipeline blocks on Slide 2
    add_block_shape(slide2, 0.6, 2.5, 1.5, 1.2, "Input Tensor\nX ∈ R^(C×F×T)", CLR_BLUE, font_size=9.5)
    add_block_shape(slide2, 2.4, 2.5, 1.5, 1.2, "Frequency Conv\n3×1 (Groups=C)\n+ BN1 + ReLU", CLR_BLUE, font_size=9.0)
    add_block_shape(slide2, 4.2, 2.5, 1.5, 1.2, "Pointwise Conv\n1×1 (C → C')\n+ BN2", CLR_BLUE, font_size=9.0)

    # Container around BFA
    add_container_shape(slide2, 6.0, 1.8, 4.9, 2.6, "Broadcasted Frequency Attention (BFA) Operator")
    add_block_shape(slide2, 6.2, 2.5, 1.4, 1.2, "Freq AvgPool (F_avg)\nz = (1/F) Σ_f x(f)\nz ∈ R^(C'×1×T)", CLR_TEAL, font_size=8.5)
    add_block_shape(slide2, 7.8, 2.5, 1.4, 1.2, "1×1 Conv + BN\nChannel Mixing", CLR_PURPLE, font_size=8.5)
    add_block_shape(slide2, 9.4, 2.5, 1.3, 1.2, "Sigmoid (σ)\ns ∈ [0, 1]^(C'×1×T)", CLR_PURPLE, font_size=8.5)

    add_block_shape(slide2, 11.2, 2.5, 0.8, 1.2, "Scale ⊗\nBroadcast", CLR_PINK, font_size=8.5)
    add_block_shape(slide2, 12.2, 2.5, 0.8, 1.2, "Output\nX_tilde", CLR_YELLOW, "#000000", font_size=8.5)

    # Residual Shortcut below
    add_block_shape(slide2, 3.5, 4.8, 5.5, 0.9, "Residual Shortcut Branch\nIdentity (s=1)  or  1×1 Conv + BN (s=2)", CLR_TEAL, font_size=9.5)

    note_box2 = slide2.shapes.add_textbox(Inches(0.6), Inches(6.5), Inches(12.133), Inches(0.5))
    tf_n2 = note_box2.text_frame
    p_n2 = tf_n2.paragraphs[0]
    p_n2.text = "All mathematical labels and component boxes are fully editable PowerPoint vectors."
    p_n2.font.size = Pt(10)
    p_n2.font.italic = True
    p_n2.font.color.rgb = hex_to_rgb("#64748B")

    pptx_path1 = os.path.join(RES_DIR, "BC_ResNet_Architectures.pptx")
    pptx_path2 = os.path.join(ARCH_DIR, "BC_ResNet_Architectures.pptx")
    prs.save(pptx_path1)
    prs.save(pptx_path2)
    print(f"  -> Saved: {pptx_path1}")
    print(f"  -> Saved: {pptx_path2}")

if __name__ == "__main__":
    generate_m5_styled_bc_resnet_architecture()
    generate_se_styled_bc_resblock_mechanism()
    generate_native_editable_pptx()
    print("\n🎉 [SUCCESS] All MT-styled architecture diagrams and editable .pptx generated successfully!")
