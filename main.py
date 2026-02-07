import numpy as np
import matplotlib.pyplot as plt
import data_generator
from fusion_engine import FusionEngine 
from visualize_joint_pdf import plot_joint_pdf_snapshot

def run_project():
    print("1. GENERATING SIMULATION DATA...")
    data = data_generator.generate_data()
    
    ground_truth = data["ground_truth"]
    z1_readings = data["z1"]
    z2_readings = data["z2"]
    sigma_1 = data["sigma_1"]
    sigma_2 = data["sigma_2"]
    dt = data["dt"]
    
    num_steps = len(ground_truth)
    print(f"   -> Generated {num_steps} time steps.")

    print("2. INITIALIZING FUSION ENGINE...")
    engine = FusionEngine(dt, start_pos=ground_truth[0], start_vel=[0,0])
    
    fused_path = np.zeros((num_steps, 2))
    fused_variances = np.zeros(num_steps)

    snapshot_step = 50 
    snapshot_data = {}

    print("3. RUNNING FUSION LOOP...")
    for k in range(num_steps):
        z_fused, R_fused = engine.spatial_fusion(z1_readings[k], sigma_1, z2_readings[k], sigma_2)
        
        est_pos = engine.temporal_update(z_fused, R_fused)
        
        fused_path[k] = est_pos
        fused_variances[k] = R_fused[0,0] 
        
        if k == snapshot_step:
            snapshot_data = {
                "z1": z1_readings[k, 0],   
                "z2": z2_readings[k, 0],
                "fused": z_fused[0],       
                "sigma_fused": np.sqrt(R_fused[0,0])
            }

    print("4. CALCULATING METRICS...")
    err_1 = np.linalg.norm(z1_readings - ground_truth, axis=1).mean()
    err_2 = np.linalg.norm(z2_readings - ground_truth, axis=1).mean()
    err_fused = np.linalg.norm(fused_path - ground_truth, axis=1).mean()
    
    var_fused_mean = np.mean(fused_variances)
    
    print("-" * 40)
    print(f"Final Performance Report:")
    print(f"ACCURACY (Mean Error):")
    print(f"  Sensor 1 (GPS):  {err_1:.4f} m")
    print(f"  Sensor 2 (WiFi): {err_2:.4f} m")
    print(f"  FUSED SYSTEM:    {err_fused:.4f} m")
    print("-" * 40)
    print(f"PRECISION (Variance):")
    print(f"  Sensor 1 Var:    {sigma_1**2:.4f} m^2")
    print(f"  Sensor 2 Var:    {sigma_2**2:.4f} m^2")
    print(f"  FUSED Var:       {var_fused_mean:.4f} m^2")
    print("-" * 40)

    print("5. GENERATING PLOTS...")
    
    plt.figure(figsize=(10, 8))
    plt.plot(ground_truth[:,0], ground_truth[:,1], 'k-', linewidth=3, label='Ground Truth')
    plt.scatter(z1_readings[:,0], z1_readings[:,1], c='red', s=5, alpha=0.3, label='Sensor 1')
    plt.scatter(z2_readings[:,0], z2_readings[:,1], c='blue', s=5, alpha=0.3, label='Sensor 2')
    plt.plot(fused_path[:,0], fused_path[:,1], 'g-', linewidth=2, label='Fused Result')
    plt.title(f"Trajectory Tracking (Error Reduced: {err_2:.2f}m -> {err_fused:.2f}m)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig("result_1_trajectory.png")
    print(f"   -> Trajectory Map saved to 'result_1_trajectory.png'")
    plt.close()

    plot_joint_pdf_snapshot(
        snapshot_data["z1"], sigma_1,
        snapshot_data["z2"], sigma_2,
        snapshot_data["fused"], snapshot_data["sigma_fused"],
        snapshot_step
    )
    
    print("Done! Project execution complete.")

if __name__ == "__main__":
    run_project()