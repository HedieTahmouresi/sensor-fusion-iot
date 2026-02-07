# Sensor Fusion Project: Mathematical Specification

## 1. Problem Definition
**Objective:** Estimate the 2D position and velocity of a mobile robot by fusing data from two noisy sensors.
**Method:** Two-Stage Fusion Architecture (Spatial Fusion + Temporal Kalman Filtering).

---

## 2. System Variables

### A. The State Vector ($x$)
The system tracks Position ($p$) and Velocity ($v$) in 2D space:
$$
x_k = \begin{bmatrix} p_x \\ p_y \\ v_x \\ v_y \end{bmatrix}_k
$$

### B. The Sensors
We possess two independent sensors measuring Position ($p_x, p_y$) with Gaussian noise:
1.  **Sensor 1 ($z_1$):** Variance $\sigma_1^2$ (e.g., Noisy GPS)
2.  **Sensor 2 ($z_2$):** Variance $\sigma_2^2$ (e.g., WiFi Positioning)

---

## 3. Stage 1: Spatial Fusion (Inverse Variance Weighting)
*Goal: Combine two noisy sensor readings into a single "Virtual Measurement" ($z_{fused}$) to minimize immediate variance.*

**The Fused Measurement:**
$$
z_{fused} = z_1 \left( \frac{\sigma_2^2}{\sigma_1^2 + \sigma_2^2} \right) + z_2 \left( \frac{\sigma_1^2}{\sigma_1^2 + \sigma_2^2} \right)
$$

**The Fused Variance ($R_{fused}$):**
$$
\sigma_{fused}^2 = \frac{\sigma_1^2 \cdot \sigma_2^2}{\sigma_1^2 + \sigma_2^2}
$$
*Proof:* $\sigma_{fused}^2 \leq \min(\sigma_1^2, \sigma_2^2)$.

---

## 4. Stage 2: Temporal Fusion (Linear Kalman Filter)
*Goal: Smooth the fused measurement over time using a Constant Velocity (CV) motion model.*

### A. Matrices
**1. State Transition Matrix ($F$):**
Defines the physics: $Position_{new} = Position_{old} + Velocity \times \Delta t$
$$
F = \begin{bmatrix}
1 & 0 & \Delta t & 0 \\
0 & 1 & 0 & \Delta t \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

**2. Measurement Matrix ($H$):**
Maps state to measurement (we observe Position only):
$$
H = \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0
\end{bmatrix}
$$

**3. Measurement Noise Covariance ($R$):**
Derived from Stage 1:
$$
R = \begin{bmatrix}
\sigma_{fused}^2 & 0 \\
0 & \sigma_{fused}^2
\end{bmatrix}
$$

**4. Process Noise Covariance ($Q$):**
Represents uncertainty in the physics model (e.g., wheel slip, wind).
*(To be tuned during simulation)*.

### B. The Algorithm (Recursive Loop)

**1. Predict Step (Time Update):**
Project the state ahead:
$$\hat{x}_{k|k-1} = F \hat{x}_{k-1|k-1}$$
Project the error covariance:
$$P_{k|k-1} = F P_{k-1|k-1} F^T + Q$$

**2. Update Step (Measurement Update):**
Compute Kalman Gain:
$$K_k = P_{k|k-1} H^T (H P_{k|k-1} H^T + R)^{-1}$$
Update Estimate with $z_{fused}$:
$$\hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k (z_{fused} - H \hat{x}_{k|k-1})$$
Update Error Covariance:
$$P_{k|k} = (I - K_k H) P_{k|k-1}$$


---

## 5. Validation Metrics: Normalized Innovation Squared (NIS)
*Goal: Mathematically validate the consistency of the filter and detect model mismatches (e.g., unmodeled maneuvers).*

### A. The Metric ($\epsilon_{\nu}$)
The NIS measures the statistical magnitude of the **Innovation** ($y_k = z_k - H\hat{x}_{k|k-1}$) relative to its expected covariance ($S_k$).
$$
\epsilon_{\nu, k} = y_k^T S_k^{-1} y_k
$$

### B. The Hypothesis Test ($\chi^2$)
Under the assumption of Gaussian white noise, the NIS follows a Chi-Square ($\chi^2$) distribution with $n_z$ degrees of freedom (where $n_z$ is the dimension of the measurement vector).

* **Degrees of Freedom ($n_z$):** 2 (We measure Position X and Position Y).
* **Confidence Interval:** 95%.
* **Critical Threshold:** $\chi^2_{0.95, 2} = 5.991$.

### C. Interpretation
* If $\epsilon_{\nu, k} \le 5.991$: The filter is **Consistent**. The errors are within the expected Gaussian bounds.
* If $\epsilon_{\nu, k} > 5.991$: The filter is **Inconsistent** or Divergent. This indicates an unmodeled maneuver (e.g., sudden acceleration) or incorrect noise tuning ($Q$ or $R$ is too small).