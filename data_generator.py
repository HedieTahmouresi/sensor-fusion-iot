import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

dt = 0.1             
total_time = 20.0    
steps = int(total_time / dt)

true_velocity = np.array([1.5, 1.0])
start_pos = np.array([0.0, 0.0])

sigma_1 = 2.0 

sigma_2 = 1.0 

def simulate_ground_truth(steps, dt, start_pos, mode = "L"):

    truth_data = np.zeros((steps, 2))
    current_pos = start_pos.copy()
    
    vel_east = np.array([1.5, 0.0]) 
    vel_north = np.array([0.0, 1.5]) 

    if mode == "I":
        vel_east = true_velocity
        vel_north = true_velocity  
    
    turn_step = steps // 2 
    
    print(f"Simulating Turn at step {turn_step} ({turn_step * dt} seconds)")

    for k in range(steps):
        truth_data[k] = current_pos
        
        if k < turn_step:
            velocity = vel_east
        else:
            velocity = vel_north
            
        current_pos = current_pos + velocity * dt
        
    return truth_data

def add_noise(truth_data, sigma):

    noise = np.random.normal(0, sigma, truth_data.shape)
    return truth_data + noise

def generate_data():
    print(f"Simulating {steps} time steps...")
    
    ground_truth = simulate_ground_truth(steps, dt, start_pos, "L")
    
    z1_readings = add_noise(ground_truth, sigma_1) 
    z2_readings = add_noise(ground_truth, sigma_2) 
    
    np.savez("sensor_data.npz", 
             ground_truth=ground_truth, 
             z1=z1_readings, 
             z2=z2_readings,
             sigma_1=sigma_1,
             sigma_2=sigma_2,
             dt=dt)
    print("Data saved to 'sensor_data.npz'")

    plt.figure(figsize=(10, 6))
    
    plt.scatter(z1_readings[:,0], z1_readings[:,1], 
                c='red', label=f'Sensor 1 (Sigma={sigma_1})', alpha=0.3, s=10)
    
    plt.scatter(z2_readings[:,0], z2_readings[:,1], 
                c='blue', label=f'Sensor 2 (Sigma={sigma_2})', alpha=0.3, s=10)
    
    plt.plot(ground_truth[:,0], ground_truth[:,1], 
             'k--', linewidth=2, label='Ground Truth')
    
    plt.title("Phase 2: Synthetic Data Generation")
    plt.xlabel("Position X (m)")
    plt.ylabel("Position Y (m)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig("phase2_data_visualization.png")
    print("Plot saved to 'phase2_data_visualization.png'")

    return {
        "ground_truth": ground_truth,
        "z1": z1_readings,
        "z2": z2_readings,
        "sigma_1": sigma_1,
        "sigma_2": sigma_2,
        "dt": dt
    }


if __name__ == "__main__":
    data = generate_data()