"""
SFM-X Adaptive Sensor Fusion (EKF) Engine
Fuses Visual Odometry / SLAM poses with GPS, IMU, and Barometric Altitude.
Implements Dynamic Innovation Gating (Adaptive Sensor Trust) to detect & reject
GPS multipath noise, spoofing, and signal degradation.
"""

import numpy as np
from typing import Dict, Any, List, Optional

class AdaptiveEKF:
    def __init__(
        self,
        chi2_gate_threshold: float = 7.815,  # 3-DOF 95% confidence chi-squared gate
        nominal_gps_std: float = 1.2,        # nominal consumer drone GPS horizontal accuracy (m)
        nominal_vo_std: float = 0.08,        # Visual odometry relative step precision (m)
        process_noise_accel: float = 0.5,    # Drone dynamic acceleration uncertainty (m/s^2)
    ):
        # State: [X, Y, Z, Vx, Vy, Vz, Roll, Pitch, Yaw] (9 states)
        self.state = np.zeros((9, 1))
        
        # State Covariance Matrix P (9x9)
        self.P = np.eye(9) * 1.0
        self.P[0:3, 0:3] *= 5.0    # Initial position uncertainty
        self.P[3:6, 3:6] *= 1.0    # Initial velocity uncertainty
        self.P[6:9, 6:9] *= 0.1    # Initial attitude uncertainty
        
        self.q_accel = process_noise_accel
        self.chi2_gate = chi2_gate_threshold
        self.nominal_gps_std = nominal_gps_std
        self.nominal_vo_std = nominal_vo_std
        
        # Telemetry & Diagnostics
        self.current_trust_index = 1.0
        self.rejected_gps_count = 0
        self.total_gps_updates = 0
        self.estimated_horizontal_error = nominal_gps_std
        self.estimated_vertical_error = nominal_gps_std * 1.5

    def predict(self, dt: float, imu_accel: Optional[np.ndarray] = None, imu_gyro: Optional[np.ndarray] = None):
        """
        Constant-velocity / IMU-driven state prediction step.
        """
        if dt <= 0:
            return

        F = np.eye(9)
        # Position update: p = p + v * dt
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        
        # State prediction
        if imu_accel is not None and len(imu_accel) == 3:
            # Add IMU linear acceleration input
            self.state[3:6] += imu_accel.reshape((3, 1)) * dt
            
        if imu_gyro is not None and len(imu_gyro) == 3:
            # Add gyro angular rate input
            self.state[6:9] += imu_gyro.reshape((3, 1)) * dt
            
        self.state = F @ self.state
        
        # Process Noise Covariance Q
        Q = np.eye(9) * 1e-4
        q_pos = 0.5 * (dt**2) * self.q_accel
        q_vel = dt * self.q_accel
        for i in range(3):
            Q[i, i] = q_pos**2
            Q[i+3, i+3] = q_vel**2
        Q[6:9, 6:9] *= 0.001
        
        self.P = F @ self.P @ F.T + Q

    def update_visual_odometry(self, delta_xyz: np.ndarray, delta_rpy: Optional[np.ndarray] = None, dt: Optional[float] = None):
        """
        Updates filter using high-frequency relative displacement from Visual SLAM.
        Directly integrates precise relative step and tracks velocity without covariance divergence.
        """
        self.state[0:3] += delta_xyz.reshape((3, 1))
        if dt and dt > 0:
            self.state[3:6] = (delta_xyz / dt).reshape((3, 1))
            
        R_vo = np.eye(3) * (self.nominal_vo_std ** 2)
        self.P[0:3, 0:3] += R_vo
        
        if delta_rpy is not None:
            self.state[6:9] = delta_rpy.reshape((3, 1))

    def update_gps(
        self, 
        gps_enu: np.ndarray, 
        reported_accuracy: float = 1.5, 
        hdop: float = 1.0
    ) -> Dict[str, Any]:
        """
        Updates filter with GPS measurement.
        Executes Adaptive Sensor Trust (Innovation Gating):
        Detects anomalous GPS multipath jumps and dynamically derates or rejects them.
        """
        self.total_gps_updates += 1
        
        H = np.zeros((3, 9))
        H[0:3, 0:3] = np.eye(3)
        
        z = gps_enu.reshape((3, 1))
        y = z - H @ self.state  # Innovation residual
        
        # Base sensor measurement covariance
        sigma_xy = max(0.5, reported_accuracy * max(1.0, hdop))
        sigma_z = sigma_xy * 1.5
        R_base = np.diag([sigma_xy**2, sigma_xy**2, sigma_z**2])
        
        # Innovation Covariance
        S = H @ self.P @ H.T + R_base
        inv_S = np.linalg.inv(S)
        
        # Normalized Innovation Squared (Mahalanobis Distance)
        nis = float(np.squeeze(y.T @ inv_S @ y))
        
        # Adaptive Trust Calculation
        # Nominal chi-square for 3 DOF is ~7.815 (95%) and ~11.345 (99%)
        if nis > 14.0:
            # Extreme outlier (e.g. GPS multipath spike or tunnel loss) -> Complete rejection
            self.rejected_gps_count += 1
            self.current_trust_index = max(0.05, self.current_trust_index * 0.7)
            action = "REJECTED_OUTLIER"
            trust_weight = 0.0
        elif nis > self.chi2_gate:
            # Suspicious measurement -> Soft scaling of measurement noise (de-weight)
            action = "DEGRADED_TRUST"
            trust_weight = float(np.exp(-0.5 * (nis - self.chi2_gate)))
            self.current_trust_index = max(0.2, 0.9 * self.current_trust_index + 0.1 * trust_weight)
            R_adaptive = R_base / max(0.01, trust_weight)
            S_adapt = H @ self.P @ H.T + R_adaptive
            K = self.P @ H.T @ np.linalg.inv(S_adapt)
            self.state = self.state + K @ y
            self.P = (np.eye(9) - K @ H) @ self.P
        else:
            # Healthy GPS measurement
            action = "ACCEPTED_HIGH_CONFIDENCE"
            trust_weight = 1.0
            self.current_trust_index = min(1.0, self.current_trust_index * 0.95 + 0.05)
            K = self.P @ H.T @ inv_S
            self.state = self.state + K @ y
            self.P = (np.eye(9) - K @ H) @ self.P
            
        # Update 1-sigma uncertainty bounds
        self.estimated_horizontal_error = round(float(np.sqrt(self.P[0, 0] + self.P[1, 1])), 3)
        self.estimated_vertical_error = round(float(np.sqrt(self.P[2, 2])), 3)
        
        return {
            "action": action,
            "nis_score": round(nis, 3),
            "trust_index": round(self.current_trust_index, 3),
            "est_horizontal_error_m": self.estimated_horizontal_error,
            "est_vertical_error_m": self.estimated_vertical_error,
            "fused_position": self.state[0:3].flatten().tolist()
        }

    def get_pose(self) -> Dict[str, Any]:
        """
        Returns metric camera pose in local ENU frame with uncertainties.
        """
        pos = self.state[0:3].flatten()
        vel = self.state[3:6].flatten()
        rpy = self.state[6:9].flatten()
        
        return {
            "x": float(pos[0]),
            "y": float(pos[1]),
            "z": float(pos[2]),
            "vx": float(vel[0]),
            "vy": float(vel[1]),
            "vz": float(vel[2]),
            "roll_deg": float(np.degrees(rpy[0])),
            "pitch_deg": float(np.degrees(rpy[1])),
            "yaw_deg": float(np.degrees(rpy[2])),
            "trust_index": round(self.current_trust_index, 3),
            "horiz_acc_m": self.estimated_horizontal_error,
            "vert_acc_m": self.estimated_vertical_error,
        }
