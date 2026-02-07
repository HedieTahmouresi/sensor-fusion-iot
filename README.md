# 📡 Sensor Fusion in IoT Systems

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Scientific_Computing-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c?style=for-the-badge&logo=matplotlib&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)

> **A Two-Stage Adaptive Sensor Fusion architecture** designed to track an autonomous robot's trajectory by combining noisy GPS and WiFi data using Inverse Variance Weighting and an Adaptive Kalman Filter (AKF).

---

## 🔭 Project Overview
This project implements a robust state estimation system for an autonomous robot following an "L-shaped" trajectory. By fusing data from noisy sensors (GPS & WiFi), the system achieves higher precision than any single sensor could provide.

**Key Innovation:** The system features an **"Adaptive Panic Switch"** that detects sudden maneuvers (like 90° turns) using Chi-Square statistics and dynamically adjusts the filter to prevent lag.

---

## 🏗 System Architecture

### 1️⃣ Stage 1: Spatial Fusion
* **Method:** Inverse Variance Weighting.
* **Function:** Combines redundant sensor readings ($z_1, z_2$) into a single, lower-variance "Virtual Measurement".

### 2️⃣ Stage 2: Temporal Fusion
* **Method:** Adaptive Kalman Filter (AKF).
* **Function:** Smooths data over time using a Constant Velocity model.
* **Adaptation:** Uses **Normalized Innovation Squared (NIS)** to inflate process noise covariance ($Q$) during sharp turns.

---

## 📊 Performance Results

The Adaptive Filter successfully outperforms both individual sensors and the Standard Kalman Filter.

| Source | Mean Error (m) | Variance ($m^2$) | Improvement vs GPS |
| :--- | :---: | :---: | :---: |
| **Sensor 1 (GPS)** | 2.40 | 4.00 | — |
| **Sensor 2 (WiFi)** | 1.26 | 1.00 | 47.5% |
| **Adaptive Filter** | **0.34** | **0.80** | **85.8%** |

> **Conclusion:** The system achieved an **85.8% reduction in tracking error** compared to raw GPS data.

---

## ⚙️ Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/sensor-fusion-iot.git](https://github.com/yourusername/sensor-fusion-iot.git)
cd sensor-fusion-iot
```

### 2. Install Dependencies
Ensure you have Python 3.8+ installed.

```bash
pip install numpy matplotlib scipy
```

### 3. Run the Simulation

Execute the main script to generate data, run the fusion engine, and save results.

```bash
python main.py
```

The script will print a performance report to the console and save 4 visualization images to your results folder.

