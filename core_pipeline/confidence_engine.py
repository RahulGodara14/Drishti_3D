"""
SFM-X Quality Engine & AI Reconstruction Confidence
Computes multi-factor spatial confidence for every 3D point and voxel surface.
Combines observation counts, depth variance, sensor trust, and camera ray angle.
"""

import numpy as np
from typing import Dict, Any, Tuple

class ConfidenceEngine:
    def __init__(
        self,
        w_observations: float = 0.30,
        w_depth_variance: float = 0.25,
        w_pose_trust: float = 0.25,
        w_ray_angle: float = 0.20,
    ):
        self.w_obs = w_observations
        self.w_var = w_depth_variance
        self.w_trust = w_pose_trust
        self.w_ray = w_ray_angle

    def compute_point_confidence(
        self,
        observation_counts: np.ndarray,
        depth_std_dev: np.ndarray,
        sensor_trust_scores: np.ndarray,
        ray_cosines: np.ndarray,
    ) -> np.ndarray:
        """
        Computes normalized confidence [0.0, 1.0] for an array of 3D points.
        """
        # 1. Observation saturation curve (3+ views give high photogrammetric redundancy)
        obs_score = np.clip(observation_counts / 5.0, 0.1, 1.0)
        
        # 2. Depth variance penalty (low std dev = high multi-view consistency)
        depth_consistency = np.clip(1.0 - (depth_std_dev / 1.5), 0.1, 1.0)
        
        # 3. Sensor trust factor from Adaptive EKF
        trust_factor = np.clip(sensor_trust_scores, 0.0, 1.0)
        
        # 4. Ray incident angle cosine (penalize extreme glancing/grazing angles)
        ray_factor = np.clip(ray_cosines, 0.1, 1.0)
        
        confidence = (
            self.w_obs * obs_score +
            self.w_var * depth_consistency +
            self.w_trust * trust_factor +
            self.w_ray * ray_factor
        )
        
        return np.clip(confidence, 0.0, 1.0)

    @staticmethod
    def confidence_to_jet_colors(confidence_array: np.ndarray) -> np.ndarray:
        """
        Converts confidence scores [0..1] into RGB Heatmap colors [0..255].
        Low (0.0): Red/Orange -> Medium (0.5): Yellow -> High (1.0): Cyan/Blue.
        """
        c = np.clip(confidence_array, 0.0, 1.0)
        r = np.clip(1.5 - np.abs(c * 4.0 - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(c * 4.0 - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(c * 4.0 - 1.0), 0.0, 1.0)
        
        # Reverse map so Red is Low, Cyan/Blue is High (standard engineering inspection)
        heatmap = np.stack([r, g, b], axis=-1)
        # Flip r and b for Red-to-Blue
        heatmap_inverted = np.stack([1.0 - b, g, 1.0 - r], axis=-1)
        return (heatmap_inverted * 255.0).astype(np.uint8)

    def evaluate_reconstruction_quality(self, confidence_array: np.ndarray) -> Dict[str, Any]:
        """
        Summarizes global reconstruction quality and structural completeness.
        """
        avg_conf = float(np.mean(confidence_array))
        high_conf_pct = float(np.mean(confidence_array >= 0.80) * 100.0)
        med_conf_pct = float(np.mean((confidence_array >= 0.50) & (confidence_array < 0.80)) * 100.0)
        low_conf_pct = float(np.mean(confidence_array < 0.50) * 100.0)
        
        grade = "ENGINEERING_GRADE_A" if avg_conf >= 0.85 else ("SURVEY_GRADE_B" if avg_conf >= 0.70 else "PRELIMINARY_GRADE_C")
        
        return {
            "mean_confidence": round(avg_conf, 3),
            "high_confidence_pct": round(high_conf_pct, 1),
            "medium_confidence_pct": round(med_conf_pct, 1),
            "uncertain_occluded_pct": round(low_conf_pct, 1),
            "quality_grade": grade,
            "total_points_evaluated": len(confidence_array)
        }
