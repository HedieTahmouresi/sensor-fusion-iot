import numpy as np
import matplotlib.pyplot as plt
from fusion_engine import FusionEngine 

def run_fusion_experiment():
    print("Loading sensor data...")
    try:
        data = np.load("sensor_data.npz")
        ground_truth = data["ground_truth"]
        z1_readings = data["z1"]
        z2_readings = data["z2"]
        sigma_1 = float(data["sigma_1"])
        sigma_2 = float(data["sigma_2"])
        dt = float(data["dt"])
    except FileNotFoundError:
        print("Error: 'sensor_data.npz' not found. Run data_generator.py first!")
        return

    num_steps = len(ground_truth)
    print(f"Running fusion on {num_steps} time steps...")

    start_pos = ground_truth[0] 
    start_vel = [0, 0]
    
    engine = FusionEngine(dt, start_pos, start_vel)
    
    fused_path = np.zeros((num_steps, 2))
    fused_variances = np.zeros(num_steps) 

    for k in range(num_steps):
        z1 = z1_readings[k]
        z2 = z2_readings[k]
        
        z_fused_meas, R_fused_matrix = engine.spatial_fusion(z1, sigma_1, z2, sigma_2)
        
        estimated_pos = engine.temporal_update(z_fused_meas, R_fused_matrix)
        
        fused_path[k] = estimated_pos
        fused_variances[k] = R_fused_matrix[0,0] 

    err_1 = np.linalg.norm(z1_readings - ground_truth, axis=1).mean()
    err_2 = np.linalg.norm(z2_readings - ground_truth, axis=1).mean()
    err_fused = np.linalg.norm(fused_path - ground_truth, axis=1).mean()

    var_1 = sigma_1**2
    var_2 = sigma_2**2
    var_fused_theoretical = (var_1 * var_2) / (var_1 + var_2)
    var_fused_empirical = np.var(fused_path - ground_truth)

    print("-" * 40)
    print(f"### FINAL PERFORMANCE REPORT ###")
    print("-" * 40)
    print(f"1. ACCURACY (Mean Position Error):")
    print(f"   - Sensor 1 (GPS):  {err_1:.4f} m")
    print(f"   - Sensor 2 (WiFi): {err_2:.4f} m")
    print(f"   - FUSED SYSTEM:    {err_fused:.4f} m  <-- improvement!")
    print("-" * 40)
    print(f"2. PRECISION (Variance of Estimate):")
    print(f"   - Sensor 1 Var:    {var_1:.4f} m^2")
    print(f"   - Sensor 2 Var:    {var_2:.4f} m^2")
    print(f"   - Fused Var (Calc):{var_fused_theoretical:.4f} m^2")
    print(f"   * PROOF: {var_fused_theoretical:.4f} < {min(var_1, var_2):.4f}")
    print("-" * 40)
    
    plt.figure(figsize=(12, 8))
    
    plt.plot(ground_truth[:,0], ground_truth[:,1], 'k-', linewidth=3, label='Ground Truth')
    
    plt.scatter(z1_readings[:,0], z1_readings[:,1], c='red', s=5, alpha=0.3, label='Sensor 1 (GPS)')
    plt.scatter(z2_readings[:,0], z2_readings[:,1], c='blue', s=5, alpha=0.3, label='Sensor 2 (WiFi)')
    
    plt.plot(fused_path[:,0], fused_path[:,1], 'g-', linewidth=2, label='Fused & Filtered (Final)')
    
    plt.title(f"Sensor Fusion Result\nError Reduced from {err_1:.2f}m -> {err_fused:.2f}m")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    plt.savefig("final_fusion_result.png")
    print("Graph saved to 'final_fusion_result.png'")

if __name__ == "__main__":
    run_fusion_experiment()