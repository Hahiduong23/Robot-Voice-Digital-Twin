import os
import matplotlib.pyplot as plt

os.makedirs("ADL/results/equations", exist_ok=True)

# Matplotlib configuration for crisp, professional mathematical typography
plt.rcParams['mathtext.fontset'] = 'cm' # Computer Modern (classic LaTeX font)
plt.rcParams['font.family'] = 'STIXGeneral'

equations = [
    ("eq1_center_pad.png", 
     r"$x_{\mathrm{pad}}[n] = [\; 0_{\mathrm{left}} \;,\; x[n] \;,\; 0_{\mathrm{right}} \;], \quad \mathrm{pad}_{\mathrm{left}} = \left\lfloor \frac{N_{\mathrm{target}} - L}{2} \right\rfloor, \quad \mathrm{pad}_{\mathrm{right}} = N_{\mathrm{target}} - L - \mathrm{pad}_{\mathrm{left}}$",
     14, (7.5, 0.9)),
    
    ("eq2_stft.png",
     r"$X(m, \omega) = \sum_{n=-\infty}^{\infty} x[n] \cdot w[n - m \cdot H] \cdot e^{-j \cdot \omega \cdot n}$",
     15, (5.5, 0.9)),
    
    ("eq3_mel.png",
     r"$S_{\mathrm{mel}}(m, f_{\mathrm{mel}}) = 10 \cdot \log_{10} \left( \sum_{k=0}^{N_{\mathrm{fft}}/2} |X(m, k)|^2 \cdot H_{\mathrm{mel}}(k, f_{\mathrm{mel}}) + \varepsilon \right)$",
     14, (6.8, 0.95)),
    
    ("eq4_batchnorm.png",
     r"$\hat{x}^{(k)} = \frac{x^{(k)} - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}}, \qquad y^{(k)} = \gamma^{(k)} \cdot \hat{x}^{(k)} + \beta^{(k)}$",
     14, (6.2, 0.95)),
    
    ("eq5_residual.png",
     r"$y = \mathcal{F}(x, \{W_i\}) + x, \qquad \mathrm{where} \quad \mathrm{dim}(\mathcal{F}) = \mathrm{dim}(x)$",
     14, (5.8, 0.85)),
    
    ("eq6_squeeze.png",
     r"$z_c = \mathcal{F}_{\mathrm{sq}}(u_c) = \frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W u_c(i, j)$",
     14, (5.5, 0.95)),
    
    ("eq7_excitation.png",
     r"$s = \mathcal{F}_{\mathrm{ex}}(z, W) = \sigma\left( W_2 \cdot \mathrm{ReLU}(W_1 \cdot z) \right), \qquad \tilde{x}_c = s_c \cdot u_c$",
     14, (6.8, 0.9)),
    
    ("eq8_cross_entropy.png",
     r"$\mathcal{L}_{\mathrm{CE}} = - \sum_{k=1}^K y_k \cdot \log(p_k), \qquad \mathrm{where} \quad p_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$",
     14, (6.5, 0.95)),
    
    ("eq9_label_smoothing.png",
     r"$y_k^{\mathrm{LS}} = (1 - \alpha) \cdot y_k + \frac{\alpha}{K} = 0.9 \cdot y_k + \frac{0.1}{7}$",
     14, (5.2, 0.9)),
    
    ("eq10_cosine_annealing.png",
     r"$\eta_t = \eta_{\min} + \frac{1}{2} (\eta_{\max} - \eta_{\min}) \cdot \left[ 1 + \cos\left( \frac{T_{\mathrm{cur}}}{T_{\max}} \cdot \pi \right) \right]$",
     14, (6.2, 0.9)),
    
    ("eq11_gradcam_alpha.png",
     r"$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^H \sum_{j=1}^W \frac{\partial Y^c}{\partial A_{i,j}^k}$",
     15, (4.2, 0.95)),
    
    ("eq12_gradcam_map.png",
     r"$L_{\mathrm{Grad-CAM}}^c = \mathrm{ReLU}\left( \sum_k \alpha_k^c \cdot A^k \right)$",
     15, (4.5, 0.85))
]

for filename, latex_str, fontsize, figsize in equations:
    fig = plt.figure(figsize=figsize, dpi=300)
    # Pure white background
    fig.patch.set_facecolor('#FFFFFF')
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.text(0.5, 0.5, latex_str, 
            ha='center', va='center', 
            fontsize=fontsize, 
            color='#102C57') # Elegant Phenikaa Navy
    
    out_path = os.path.join("ADL/results/equations", filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.06, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"[RENDERED] {out_path}")

print("[SUCCESS] All 12 equation images generated with 300 DPI mathematical typography!")
