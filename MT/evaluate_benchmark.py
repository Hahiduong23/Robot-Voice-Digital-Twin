import os
import json
import matplotlib.pyplot as plt

def main():
    summary_path = "MT/results/ablation_summary.json"
    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found.")
        return

    with open(summary_path) as f:
        results = json.load(f)

    model_styles = {
        "M1_BaselineCNN": {"color": "#1f77b4", "marker": "o", "label": "M1: Baseline CNN"},
        "M2_DropoutCNN": {"color": "#2ca02c", "marker": "s", "label": "M2: + Dropout"},
        "M3_BatchNormCNN": {"color": "#ff7f0e", "marker": "^", "label": "M3: + BatchNorm"},
        "M4_VoiceResNet": {"color": "#9467bd", "marker": "D", "label": "M4: VoiceResNet"},
        "M5_SE_VoiceResNet": {"color": "#d62728", "marker": "*", "label": "M5: SE_VoiceResNet (SOTA)"},
    }

    # Offsets with arrows so annotations do not collide
    offsets_params = {
        "M1_BaselineCNN": (-46, -22),
        "M2_DropoutCNN": (0, 18),
        "M3_BatchNormCNN": (42, -22),
        "M4_VoiceResNet": (46, -18),
        "M5_SE_VoiceResNet": (46, 15),
    }

    offsets_flops = {
        "M1_BaselineCNN": (-44, -22),
        "M2_DropoutCNN": (0, 18),
        "M3_BatchNormCNN": (44, -22),
        "M4_VoiceResNet": (-48, -18),
        "M5_SE_VoiceResNet": (-48, 15),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.2))

    # Subplot 1: Test Acc vs Params
    for r in results:
        m = r["model_name"]
        style = model_styles[m]
        p_k = r["params"] / 1000.0
        acc = r["test_acc"]
        
        ax1.scatter(p_k, acc, color=style["color"], marker=style["marker"], 
                    s=160 if style["marker"] != "*" else 290, 
                    edgecolors="black", linewidth=1.2, zorder=4, label=style["label"])
        
        ox, oy = offsets_params[m]
        ax1.annotate(
            f"{m[:2]}: {acc:.1f}%",
            (p_k, acc),
            xytext=(ox, oy),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=style["color"],
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#ffffff", alpha=0.95, edgecolor=style["color"], linewidth=1.1),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.08", color=style["color"], lw=1.1)
        )

    ax1.set_title("(a) Trade-off: Test Accuracy vs. Parameters", fontsize=12, fontweight="bold", pad=14)
    ax1.set_xlabel("Number of Parameters (Kilo-Params)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Test Accuracy (%)", fontsize=11, fontweight="bold")
    ax1.set_ylim(96.5, 103.5)
    ax1.set_xlim(0, 2350)
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Legend dãn dòng rộng rãi (labelspacing=1.35, handletextpad=1.2, borderpad=1.0)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=2, 
               framealpha=0.95, fontsize=9.0, 
               labelspacing=1.35, 
               handletextpad=1.2, 
               borderpad=1.0, 
               columnspacing=2.2, 
               edgecolor="#cccccc", fancybox=True, 
               title="Models", title_fontsize=9.5)

    # Subplot 2: Test Acc vs FLOPs
    for r in results:
        m = r["model_name"]
        style = model_styles[m]
        fl = r["flops_m"]
        acc = r["test_acc"]
        
        ax2.scatter(fl, acc, color=style["color"], marker=style["marker"], 
                    s=160 if style["marker"] != "*" else 290, 
                    edgecolors="black", linewidth=1.2, zorder=4, label=style["label"])
        
        ox, oy = offsets_flops[m]
        ax2.annotate(
            f"{m[:2]}: {acc:.1f}%",
            (fl, acc),
            xytext=(ox, oy),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=style["color"],
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#ffffff", alpha=0.95, edgecolor=style["color"], linewidth=1.1),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.08", color=style["color"], lw=1.1)
        )

    ax2.set_title("(b) Trade-off: Test Accuracy vs. Computation (FLOPs)", fontsize=12, fontweight="bold", pad=14)
    ax2.set_xlabel("Computational Complexity (Mega-FLOPs)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Test Accuracy (%)", fontsize=11, fontweight="bold")
    ax2.set_ylim(96.5, 103.5)
    ax2.set_xlim(2.0, 17.5)
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    # Legend dãn dòng rộng rãi (labelspacing=1.35, handletextpad=1.2, borderpad=1.0)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=2, 
               framealpha=0.95, fontsize=9.0, 
               labelspacing=1.35, 
               handletextpad=1.2, 
               borderpad=1.0, 
               columnspacing=2.2, 
               edgecolor="#cccccc", fancybox=True, 
               title="Models", title_fontsize=9.5)

    plt.tight_layout()
    os.makedirs("MT/results", exist_ok=True)
    plt.savefig("MT/results/pareto_tradeoff.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("[SUCCESS] Regenerated pareto_tradeoff.png with further expanded legend line spacing!")

if __name__ == "__main__":
    main()