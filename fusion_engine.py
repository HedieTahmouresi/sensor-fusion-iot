import numpy as np

class FusionEngine:
    def __init__(self, dt, start_pos, start_vel, process_noise_std = 0.5):
        
        # State Vector x = [px, py, vx, vy]
        self.x = np.zeros(4)
        self.x[0] = start_pos[0]
        self.x[1] = start_pos[1]
        self.x[2] = start_vel[0]
        self.x[3] = start_vel[1]
        
        # Covariance Matrix P (Initial Uncertainty)
        self.P = np.eye(4) * 10.0
        
        # State Transition Matrix F (4x4)
        self.F = np.eye(4)
        self.F[0, 2] = dt # px += vx * dt
        self.F[1, 3] = dt # py += vy * dt
        
        # Measurement Matrix H (2x4) - We only measure Position
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process Noise Q (Uncertainty in Physics)
        q = process_noise_std**2
        dt2 = dt**2
        dt3 = dt**3
        dt4 = dt**4
        
        # The analytical solution for CV model process noise
        # This relates position variance to velocity variance
        self.Q = np.array([
            [dt4/4, 0,     dt3/2, 0    ],
            [0,     dt4/4, 0,     dt3/2],
            [dt3/2, 0,     dt2,   0    ],
            [0,     dt3/2, 0,     dt2  ]
        ]) * q

    def spatial_fusion(self, z1, sigma1, z2, sigma2):
        var1 = sigma1**2
        var2 = sigma2**2
        
        weight1 = var2 / (var1 + var2)
        weight2 = var1 / (var1 + var2)
        
        z_fused = weight1 * z1 + weight2 * z2
        
        var_fused = (var1 * var2) / (var1 + var2)
        
        R_fused = np.diag([var_fused, var_fused])
        
        return z_fused, R_fused
        

    def temporal_update(self, z_fused, R_fused):

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        y = z_fused - (self.H @ self.x)
        
        S = self.H @ self.P @ self.H.T + R_fused
        
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.x = self.x + (K @ y)
        
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        
        return self.x[:2]