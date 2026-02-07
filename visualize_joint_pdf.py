import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def plot_joint_pdf_snapshot(z1_val, sigma1, z2_val, sigma2, fused_val, sigma_fused, time_step):

    x_min = min(z1_val, z2_val, fused_val) - (3 * max(sigma1, sigma2))
    x_max = max(z1_val, z2_val, fused_val) + (3 * max(sigma1, sigma2))
    x = np.linspace(x_min, x_max, 1000)

    pdf_1 = stats.norm.pdf(x, z1_val, sigma1)
    pdf_2 = stats.norm.pdf(x, z2_val, sigma2)
    pdf_fused = stats.norm.pdf(x, fused_val, sigma_fused)

    plt.figure(figsize=(10, 6))
    
    plt.plot(x, pdf_1, 'r--', label=f'Sensor 1 (GPS): $\mu$={z1_val:.2f}, $\sigma$={sigma1}')
    plt.fill_between(x, pdf_1, color='red', alpha=0.1)
    
    plt.plot(x, pdf_2, 'b--', label=f'Sensor 2 (WiFi): $\mu$={z2_val:.2f}, $\sigma$={sigma2}')
    plt.fill_between(x, pdf_2, color='blue', alpha=0.1)
    
    plt.plot(x, pdf_fused, 'g-', linewidth=3, label=f'Fused Result: $\mu$={fused_val:.2f}, $\sigma$={sigma_fused:.2f}')
    plt.fill_between(x, pdf_fused, color='green', alpha=0.2)

    plt.title(f"Visualizing Fusion Math at Time Step k={time_step}\n(Multiplication of Probabilities)")
    plt.xlabel("X Position (m)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("result_2_joint_pdf.png")
    print(f"   -> PDF Snapshot saved to 'result_2_joint_pdf.png'")
    plt.close()