import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def plot_joint_pdf():
    # 1. Define two sensor distributions (1D example for clarity)
    # Sensor 1: Measurement = 10, Sigma = 2.0 (Noisy)
    mu1, sigma1 = 10.0, 2.0
    
    # Sensor 2: Measurement = 12, Sigma = 1.0 (Less Noisy)
    mu2, sigma2 = 12.0, 1.0

    # 2. Calculate Fused Parameters (The Math)
    # Variance = (s1^2 * s2^2) / (s1^2 + s2^2)
    var_fused = (sigma1**2 * sigma2**2) / (sigma1**2 + sigma2**2)
    sigma_fused = np.sqrt(var_fused)
    
    # Mean = (m1*v2 + m2*v1) / (v1 + v2)
    mu_fused = (mu1 * sigma2**2 + mu2 * sigma1**2) / (sigma1**2 + sigma2**2)

    # 3. Generate PDF Curves
    x = np.linspace(5, 17, 1000)
    pdf1 = stats.norm.pdf(x, mu1, sigma1)
    pdf2 = stats.norm.pdf(x, mu2, sigma2)
    pdf_fused = stats.norm.pdf(x, mu_fused, sigma_fused)

    # 4. Plot
    plt.figure(figsize=(10, 6))
    
    plt.plot(x, pdf1, 'r--', linewidth=2, label=f'Sensor 1 (Meas={mu1}, $\sigma$={sigma1})')
    plt.fill_between(x, pdf1, color='red', alpha=0.1)
    
    plt.plot(x, pdf2, 'b--', linewidth=2, label=f'Sensor 2 (Meas={mu2}, $\sigma$={sigma2})')
    plt.fill_between(x, pdf2, color='blue', alpha=0.1)
    
    plt.plot(x, pdf_fused, 'g-', linewidth=3, label=f'Joint PDF (Fused: $\mu$={mu_fused:.2f}, $\sigma$={sigma_fused:.2f})')
    plt.fill_between(x, pdf_fused, color='green', alpha=0.2)
    
    plt.title("Visualizing Sensor Fusion as Joint Probability Multiplication")
    plt.xlabel("Position Estimate")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("joint_pdf_visualization.png")
    print("Joint PDF plot saved to 'joint_pdf_visualization.png'")

if __name__ == "__main__":
    plot_joint_pdf()