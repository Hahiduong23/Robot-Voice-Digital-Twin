import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, PathPatch
import matplotlib.path as mpath

# Palette
CLR_YELLOW  = "#F5A623"
CLR_BLUE    = "#4A90E2"
CLR_TEAL    = "#00838F"
CLR_PURPLE  = "#6A1B9A"
CLR_PINK    = "#D81B60"
CLR_BORDER  = "#E65100"
CLR_ARROW   = "#1976D2"

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

class Block:
    def __init__(self, cx, cy, w, h, text, facecolor, textcolor="white", fontsize=7.6, bold=True, edgecolor="#1A252C"):
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.text = text
        self.facecolor = facecolor
        self.textcolor = textcolor
        self.fontsize = fontsize
        self.bold = bold
        self.edgecolor = edgecolor

    @property
    def left(self): return (self.cx - self.w/2, self.cy)
    @property
    def right(self): return (self.cx + self.w/2, self.cy)
    @property
    def top(self): return (self.cx, self.cy + self.h/2)
    @property
    def bottom(self): return (self.cx, self.cy - self.h/2)

    def draw(self, ax):
        weight = "bold" if self.bold else "normal"
        box = FancyBboxPatch((self.cx - self.w/2, self.cy - self.h/2), self.w, self.h,
                             boxstyle="round,pad=0.14,rounding_size=0.14",
                             facecolor=self.facecolor, edgecolor=self.edgecolor, linewidth=1.4,
                             zorder=4)
        ax.add_patch(box)
        ax.text(self.cx, self.cy, self.text, ha='center', va='center', color=self.textcolor,
                fontsize=self.fontsize, fontweight=weight, zorder=5, multialignment='center')

def draw_container(ax, cx, cy, w, h, label="", border_color=CLR_BORDER):
    box = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                         boxstyle="round,pad=0.2,rounding_size=0.22",
                         facecolor="#FFFBF5", edgecolor=border_color, linewidth=1.6,
                         zorder=2, alpha=0.95)
    ax.add_patch(box)
    
    if label:
        badge_w = min(w * 0.88, max(len(label) * 0.11, 2.0))
        badge_h = 0.42
        badge_y = cy + h/2 + 0.06
        badge = FancyBboxPatch((cx - badge_w/2, badge_y - badge_h/2), badge_w, badge_h,
                               boxstyle="round,pad=0.06,rounding_size=0.1",
                               facecolor="#FFE0B2", edgecolor=border_color, linewidth=1.2,
                               zorder=4)
        ax.add_patch(badge)
        ax.text(cx, badge_y, label, ha='center', va='center', color="#BF360C",
                fontsize=8.5, fontweight='bold', zorder=5)

def arrow_straight(ax, p1, p2, color=CLR_ARROW, lw=2.0):
    ax.annotate("", xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=16, shrinkA=3, shrinkB=3),
                zorder=3)

def generate_se_detail():
    fig, ax = plt.subplots(figsize=(15.0, 4.6), dpi=300)
    # Generous limits so blocks never clip or stick out
    ax.set_xlim(-0.6, 15.6)
    ax.set_ylim(-0.8, 4.0)
    ax.axis('off')

    # Compact blocks
    b_x   = Block(1.4, 2.0, 1.6, 0.9, "Input Tensor\nX in R^(C x H x W)", CLR_BLUE, fontsize=7.6)
    b_sq  = Block(3.9, 2.0, 1.8, 0.9, "Squeeze (F_sq)\nGlobal Avg Pool\nz in R^(C x 1 x 1)", CLR_TEAL, fontsize=7.2)
    b_fc1 = Block(6.6, 2.0, 1.6, 0.8, "Dense (C -> C/r)\n+ ReLU (r=16)", CLR_PURPLE, fontsize=7.2)
    b_fc2 = Block(9.0, 2.0, 1.6, 0.8, "Dense (C/r -> C)\n+ Sigmoid (s in [0,1]^C)", CLR_PURPLE, fontsize=7.0)
    b_sc  = Block(11.7, 2.0, 1.8, 0.9, "Scale (F_scale)\nChannel Multiply\nx_tilde = s_c * u_c", CLR_PINK, fontsize=7.2, edgecolor="#FFD54F")
    b_out = Block(14.0, 2.0, 1.3, 0.85, "Calibrated\nOutput\nX_tilde", CLR_YELLOW, textcolor="#000000", fontsize=7.6)

    # Container around Excitation
    draw_container(ax, 7.8, 2.0, 4.4, 1.7, "Excitation Operator (F_ex)")

    for b in [b_x, b_sq, b_fc1, b_fc2, b_sc, b_out]:
        b.draw(ax)

    arrow_straight(ax, b_x.right, b_sq.left)
    arrow_straight(ax, b_sq.right, b_fc1.left)
    arrow_straight(ax, b_fc1.right, b_fc2.left)
    arrow_straight(ax, b_fc2.right, b_sc.left)
    arrow_straight(ax, b_sc.right, b_out.left)

    plt.title("Figure 13. Detailed Squeeze-and-Excitation (SE) Channel Attention Mechanism", 
              fontsize=11.5, fontweight='bold', color="#102C57", y=-0.08)
    
    plt.savefig("ADL/results/architectures/arch_SE_Block_Mechanism.png", bbox_inches='tight', pad_inches=0.3, dpi=300)
    plt.close()
    print("[SUCCESS] SE Block detail regenerated with compact blocks and generous margins!")

if __name__ == "__main__":
    generate_se_detail()
