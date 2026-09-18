"""
SFM-X Video Intelligence & Keyframe Selection Engine
Filters raw 30/60 FPS drone footage down to optimal 2-4 FPS keyframes
based on sharpness, exposure quality, feature density, and motion overlap.
"""

import numpy as np
from typing import List, Dict, Tuple, Any

class VideoFrameEngine:
    def __init__(
        self,
        target_fps: float = 3.0,
        min_sharpness_threshold: float = 45.0,
        weight_sharpness: float = 0.35,
        weight_exposure: float = 0.20,
        weight_features: float = 0.25,
        weight_overlap: float = 0.20,
    ):
        self.target_fps = target_fps
        self.min_sharpness = min_sharpness_threshold
        self.w_sharpness = weight_sharpness
        self.w_exposure = weight_exposure
        self.w_features = weight_features
        self.w_overlap = weight_overlap

    @staticmethod
    def compute_sharpness(gray_img: np.ndarray) -> float:
        """
        Calculates Laplacian variance as high-frequency focus metric.
        Higher = crisper details, Lower = motion or focal blur.
        """
        if gray_img.ndim > 2:
            gray_img = np.mean(gray_img, axis=2)
        
        # Discrete Laplacian kernel
        lap = (
            np.roll(gray_img, 1, axis=0) +
            np.roll(gray_img, -1, axis=0) +
            np.roll(gray_img, 1, axis=1) +
            np.roll(gray_img, -1, axis=1) -
            4.0 * gray_img
        )
        # Variance of Laplacian
        var = float(np.var(lap[1:-1, 1:-1]))
        return var

    @staticmethod
    def compute_exposure_score(gray_img: np.ndarray) -> float:
        """
        Scores luminance distribution between 0.0 (severely clipped) and 1.0 (balanced).
        Penalizes underexposed shadows and washed out highlights.
        """
        if gray_img.ndim > 2:
            gray_img = np.mean(gray_img, axis=2)
        
        under_exposed = np.mean(gray_img < 25)
        over_exposed = np.mean(gray_img > 230)
        mean_lum = np.mean(gray_img)
        
        # Ideal drone imagery luminance center around 110-140
        lum_centrality = max(0.0, 1.0 - abs(mean_lum - 128.0) / 128.0)
        penalty = (under_exposed * 1.5 + over_exposed * 1.5)
        return float(np.clip(lum_centrality - penalty, 0.0, 1.0))

    @staticmethod
    def compute_feature_density(gray_img: np.ndarray, grid_size: int = 8) -> float:
        """
        Evaluates spatial distribution of gradients across an 8x8 spatial grid.
        High score indicates rich texture suitable for feature tracking & SLAM.
        """
        if gray_img.ndim > 2:
            gray_img = np.mean(gray_img, axis=2)
            
        gy, gx = np.gradient(gray_img.astype(float))
        grad_mag = np.sqrt(gx**2 + gy**2)
        
        h, w = gray_img.shape
        gh, gw = h // grid_size, w // grid_size
        active_cells = 0
        
        for r in range(grid_size):
            for c in range(grid_size):
                cell = grad_mag[r*gh:(r+1)*gh, c*gw:(c+1)*gw]
                if np.percentile(cell, 85) > 15.0:
                    active_cells += 1
                    
        return float(active_cells / (grid_size * grid_size))

    @staticmethod
    def compute_overlap_score(prev_img: np.ndarray, curr_img: np.ndarray) -> float:
        """
        Computes motion overlap. 
        Too high overlap (1.0) = drone hovering/redundant frame.
        Too low overlap (<0.3) = camera jump or occluded frame.
        Optimal target overlap for multi-view photogrammetry is 0.70 - 0.85.
        """
        if prev_img is None:
            return 1.0
        
        p_gray = np.mean(prev_img, axis=2) if prev_img.ndim > 2 else prev_img
        c_gray = np.mean(curr_img, axis=2) if curr_img.ndim > 2 else curr_img
        
        diff = np.mean(np.abs(p_gray.astype(float) - c_gray.astype(float))) / 255.0
        # diff 0.05 - 0.20 corresponds to ~70-85% overlap
        overlap = max(0.0, 1.0 - diff * 3.0)
        
        # Penalize zero motion (redundant) and massive sudden motion (blur)
        if overlap > 0.96:
            score = 0.3  # hovering, redundant
        elif overlap < 0.40:
            score = 0.2  # too abrupt
        else:
            score = 1.0 - abs(overlap - 0.78) / 0.35
            
        return float(np.clip(score, 0.0, 1.0))

    def score_frame(
        self, 
        frame: np.ndarray, 
        prev_frame: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Produces composite quality score for a candidate frame.
        """
        gray = np.mean(frame, axis=2) if frame.ndim > 2 else frame
        
        raw_sharp = self.compute_sharpness(gray)
        # Normalize sharpness to 0..1 scale (100+ is excellent)
        norm_sharp = min(1.0, raw_sharp / 150.0)
        
        exp_score = self.compute_exposure_score(gray)
        feat_score = self.compute_feature_density(gray)
        over_score = self.compute_overlap_score(prev_frame, frame)
        
        composite = (
            self.w_sharpness * norm_sharp +
            self.w_exposure * exp_score +
            self.w_features * feat_score +
            self.w_overlap * over_score
        )
        
        is_keyframe = (raw_sharp >= self.min_sharpness) and (composite >= 0.55)
        
        return {
            "sharpness": round(raw_sharp, 2),
            "exposure": round(exp_score, 3),
            "feature_density": round(feat_score, 3),
            "overlap_score": round(over_score, 3),
            "composite_score": round(composite, 3),
            "is_keyframe": bool(is_keyframe),
        }

    def process_frame_stream(
        self, 
        frames: List[np.ndarray], 
        source_fps: float = 30.0
    ) -> List[Dict[str, Any]]:
        """
        Decimates raw video frame stream down to selected keyframes with metrics.
        """
        stride = max(1, int(round(source_fps / self.target_fps)))
        selected_keyframes = []
        prev_keyframe = None
        
        for idx in range(0, len(frames), stride):
            frame = frames[idx]
            metrics = self.score_frame(frame, prev_keyframe)
            metrics["frame_index"] = idx
            metrics["timestamp_sec"] = round(idx / source_fps, 3)
            
            if metrics["is_keyframe"] or len(selected_keyframes) == 0:
                selected_keyframes.append(metrics)
                prev_keyframe = frame
                
        return selected_keyframes
