"""
SFM-X Neural Metric Depth Pipeline
Handles monocular depth prediction and metric scale recovery.
Aligns relative disparity with metric flight parameters (camera height, SLAM baseline, and terrain bounds).
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional

class NeuralDepthEngine:
    def __init__(
        self,
        default_fov_deg: float = 84.0,   # Standard wide drone lens (e.g. DJI Mavic 3 / Mini 4)
        min_depth_m: float = 2.0,
        max_depth_m: float = 300.0,
    ):
        self.fov_deg = default_fov_deg
        self.min_depth = min_depth_m
        self.max_depth = max_depth_m

    def get_camera_intrinsics(self, img_w: int, img_h: int) -> np.ndarray:
        """
        Constructs pinhole camera intrinsic matrix K from image dimensions and FOV.
        """
        focal_px = (img_w / 2.0) / np.tan(np.radians(self.fov_deg / 2.0))
        cx = img_w / 2.0
        cy = img_h / 2.0
        
        K = np.array([
            [focal_px, 0.0,      cx],
            [0.0,      focal_px, cy],
            [0.0,      0.0,      1.0]
        ], dtype=float)
        return K

    def recover_metric_scale(
        self,
        relative_depth_map: np.ndarray,
        drone_altitude_m: float,
        camera_pitch_deg: float = -45.0,  # typical oblique drone angle
    ) -> np.ndarray:
        """
        Aligns relative depth map to absolute physical meters.
        Uses nadir/oblique ray projection against ground plane altitude.
        """
        # Ground distance along principal ray
        pitch_rad = np.radians(abs(camera_pitch_deg))
        expected_center_range = drone_altitude_m / max(0.1, np.sin(pitch_rad))
        
        # Relative depth median at central patch
        h, w = relative_depth_map.shape
        center_patch = relative_depth_map[h//4:3*h//4, w//4:3*w//4]
        rel_median = float(np.median(center_patch))
        
        if rel_median <= 1e-4:
            rel_median = 1.0
            
        scale_factor = expected_center_range / rel_median
        metric_depth = relative_depth_map * scale_factor
        
        return np.clip(metric_depth, self.min_depth, self.max_depth)

    def depth_to_point_cloud(
        self,
        rgb_img: np.ndarray,
        metric_depth: np.ndarray,
        K: np.ndarray,
        camera_pose_enu: Dict[str, float],
        stride: int = 2,
        valid_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Unprojects metric depth pixels (u, v, z) into 3D local ENU world space.
        Transforms from camera optical frame to vehicle ENU coordinates.
        """
        h, w = metric_depth.shape
        u_grid, v_grid = np.meshgrid(np.arange(0, w, stride), np.arange(0, h, stride))
        
        z = metric_depth[v_grid, u_grid]
        
        if valid_mask is not None:
            mask = valid_mask[v_grid, u_grid] & (z > self.min_depth) & (z < self.max_depth)
        else:
            mask = (z > self.min_depth) & (z < self.max_depth)
            
        u_val = u_grid[mask]
        v_val = v_grid[mask]
        z_val = z[mask]
        
        # Pinhole unprojection
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        x_cam = (u_val - cx) * z_val / fx
        y_cam = (v_val - cy) * z_val / fy
        z_cam = z_val
        
        # Camera optical frame (Z forward, X right, Y down)
        P_cam = np.vstack([x_cam, y_cam, z_cam])  # (3, N)
        
        # Camera to ENU Rotation matrix based on pitch, roll, yaw
        # Drone optical mount: forward = North/heading, down = ground
        yaw = np.radians(camera_pose_enu.get("yaw_deg", 0.0))
        pitch = np.radians(camera_pose_enu.get("pitch_deg", -55.0))
        roll = np.radians(camera_pose_enu.get("roll_deg", 0.0))
        
        # In ENU body frame (X=East, Y=North, Z=Up):
        # Camera optical frame: X=Right, Y=Down, Z=Forward (depth)
        # Base alignment (pitch=0, yaw=0): X_cam -> X_body (East), Y_cam -> -Z_body (Down), Z_cam -> Y_body (North)
        R_base = np.array([
            [1.0,  0.0,  0.0],
            [0.0,  0.0,  1.0],
            [0.0, -1.0,  0.0]
        ])
        
        # Pitch down rotation around East/X axis (tilts optical axis down toward ground)
        Rx_pitch = np.array([
            [1.0, 0.0,            0.0          ],
            [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch),  np.cos(pitch)]
        ])
        
        # Roll rotation around North/Y axis
        Ry_roll = np.array([
            [ np.cos(roll), 0.0, np.sin(roll)],
            [ 0.0,          1.0, 0.0         ],
            [-np.sin(roll), 0.0, np.cos(roll)]
        ])
        
        # Yaw heading rotation around Up/Z axis
        Rz_yaw = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw),  np.cos(yaw), 0.0],
            [0.0,          0.0,         1.0]
        ])
        
        R_world = Rz_yaw @ Ry_roll @ Rx_pitch @ R_base
        
        # Transform points to World ENU
        cam_pos = np.array([
            [camera_pose_enu.get("x", 0.0)],
            [camera_pose_enu.get("y", 0.0)],
            [camera_pose_enu.get("z", 50.0)]
        ])
        
        P_world = (R_world @ P_cam) + cam_pos  # (3, N)
        
        # Extract RGB colors
        rgb_sampled = rgb_img[v_grid, u_grid][mask]
        
        return P_world.T, rgb_sampled  # Points (N, 3), Colors (N, 3)
