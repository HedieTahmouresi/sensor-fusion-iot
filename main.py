import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
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
    engine_std = FusionEngine(dt, start_pos=ground_truth[0], start_vel=[0,0])
    engine_adpt = FusionEngine(dt, start_pos=ground_truth[0], start_vel=[0,0])
    
    path_std = np.zeros((num_steps, 2))
    nis_std = np.zeros(num_steps)
    var_std = np.zeros(num_steps) 

    path_adpt = np.zeros((num_steps, 2))
    nis_adpt = np.zeros(num_steps)
    
    snapshot_step = 50 
    snapshot_data = {}

    print("3. RUNNING FUSION LOOP...")
    for k in range(num_steps):
        z_fused, R_fused = engine_std.spatial_fusion(z1_readings[k], sigma_1, z2_readings[k], sigma_2)
        
        path_std[k], nis_std[k] = engine_std.temporal_update(z_fused, R_fused, adaptive=False)
        var_std[k] = R_fused[0,0] 
        
        path_adpt[k], nis_adpt[k] = engine_adpt.temporal_update(z_fused, R_fused, adaptive=True)

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
    err_std = np.linalg.norm(path_std - ground_truth, axis=1).mean()
    err_adpt = np.linalg.norm(path_adpt - ground_truth, axis=1).mean()
    
    var_fused_mean = np.mean(var_std)
    
    print("-" * 50)
    print(f"FINAL PERFORMANCE REPORT")
    print("-" * 50)
    print(f"BASELINE SENSORS (Accuracy):")
    print(f"  Sensor 1 (GPS):      {err_1:.4f} m")
    print(f"  Sensor 2 (WiFi):     {err_2:.4f} m")
    print("-" * 50)
    print(f"PRECISION (Variance - Step 5 Req):")
    print(f"  Sensor 1 Var:        {sigma_1**2:.4f} m^2")
    print(f"  Sensor 2 Var:        {sigma_2**2:.4f} m^2")
    print(f"  FUSED Var:           {var_fused_mean:.4f} m^2")
    print(f"  -> Fusion is {(sigma_2**2)/var_fused_mean:.2f}x more precise than best sensor.")
    print("-" * 50)
    print(f"FILTER COMPARISON (Accuracy):")
    print(f"  STANDARD Filter:     {err_std:.4f} m")
    print(f"  ADAPTIVE Filter:     {err_adpt:.4f} m")
    print(f"  -> Improvement:      {(err_std - err_adpt):.4f} m")
    print("-" * 50)
    print("-" * 40)

    print("5. GENERATING PLOTS...")
    
    plt.figure(figsize=(10, 8))
    plt.plot(ground_truth[:,0], ground_truth[:,1], 'k-', linewidth=3, label='Ground Truth')
    plt.scatter(z1_readings[:,0], z1_readings[:,1], c='red', s=5, alpha=0.1) 
    plt.scatter(z2_readings[:,0], z2_readings[:,1], c='blue', s=5, alpha=0.1)
    plt.plot(path_std[:,0], path_std[:,1], 'm--', linewidth=2, label=f'Standard (Err={err_std:.2f}m)')
    plt.plot(path_adpt[:,0], path_adpt[:,1], 'g-', linewidth=2, label=f'Adaptive (Err={err_adpt:.2f}m)')
    plt.title(f"Trajectory Tracking: Standard vs. Adaptive")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig("result_1_trajectory.png")
    print("   -> Map saved to 'result_1_trajectory.png'")
    plt.close()

    plot_joint_pdf_snapshot(
        snapshot_data["z1"], sigma_1,
        snapshot_data["z2"], sigma_2,
        snapshot_data["fused"], snapshot_data["sigma_fused"],
        snapshot_step
    )

    plt.figure(figsize=(10, 6))
    plt.plot(nis_std, 'm-', linewidth=1, label='NIS (Standard)')
    plt.axhline(y=5.991, color='r', linestyle='--', linewidth=2, label='95% Threshold')
    plt.title("Filter Consistency Check (NIS)\nFull Duration")
    plt.xlabel("Time Step (k)")
    plt.ylabel("NIS Value")
    plt.legend()
    plt.grid(True)
    plt.savefig("result_3_nis_analysis.png")
    print("   -> NIS Plot saved to 'result_3_nis_analysis.png'")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(nis_std, 'm--', label='Standard NIS (High Spike)')
    plt.plot(nis_adpt, 'g-', label='Adaptive NIS (Corrected)')
    plt.axhline(y=5.991, color='r', linestyle='--', label='95% Threshold')
    mid_point = num_steps // 2
    plt.xlim(mid_point - 20, mid_point + 20)
    plt.title("NIS Response at the Turn (Zoomed)")
    plt.xlabel("Time Step")
    plt.ylabel("NIS Value")
    plt.legend()
    plt.grid(True)
    plt.savefig("result_4_nis_zoomed.png")
    print("   -> NIS Zoom plot saved to 'result_4_nis_zoomed.png'")
    plt.close()
    
    print("Done! Project execution complete.")

if __name__ == "__main__":
    run_project()