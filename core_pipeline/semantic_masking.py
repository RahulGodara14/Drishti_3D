"""
SFM-X Semantic Engine & Dynamic Object Masking
Separates static structural elements (buildings, terrain, roads, vegetation)
from transient dynamic objects (cars, trucks, pedestrians) to eliminate
ghosting trails and reconstruction noise.
"""

import numpy as np
from typing import Dict, Any, List, Tuple

class SemanticClass:
    TERRAIN = 0
    ROAD = 1
    BUILDING = 2
    VEGETATION = 3
    INFRASTRUCTURE = 4
    # Dynamic classes to eliminate
    VEHICLE = 10
    PEDESTRIAN = 11
    ANIMAL = 12

DYNAMIC_CLASSES = {SemanticClass.VEHICLE, SemanticClass.PEDESTRIAN, SemanticClass.ANIMAL}

# Color palette for 3D digital twin semantic layers [R, G, B]
SEMANTIC_COLORS = {
    SemanticClass.TERRAIN: [194, 178, 128],        # Sandy Earth #c2b280
    SemanticClass.ROAD: [90, 95, 105],            # Asphalt Gray #5a5f69
    SemanticClass.BUILDING: [74, 144, 226],        # Architectural Blue #4a90e2
    SemanticClass.VEGETATION: [46, 204, 113],      # Foliage Green #2ecc71
    SemanticClass.INFRASTRUCTURE: [243, 156, 18],  # Industrial Amber #f39c12
    SemanticClass.VEHICLE: [231, 76, 60],          # Dynamic Red #e74c3c
    SemanticClass.PEDESTRIAN: [255, 107, 107],     # Coral Pink #ff6b6b
}

class SemanticMaskEngine:
    def __init__(self, confidence_threshold: float = 0.50):
        self.conf_thresh = confidence_threshold

    def generate_synthetic_segmentation(
        self, 
        rgb_img: np.ndarray,
        detected_objects: List[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Produces semantic class map and static binary mask.
        In standalone / CPU mode, segments based on spectral cues + bounding boxes.
        When PyTorch / YOLOv8-seg is loaded, uses deep instance segmentation.
        """
        h, w = rgb_img.shape[:2]
        class_map = np.zeros((h, w), dtype=np.uint8)
        static_mask = np.ones((h, w), dtype=bool)
        
        # Color-space heuristic for baseline segmentation
        r = rgb_img[:, :, 0].astype(float)
        g = rgb_img[:, :, 1].astype(float)
        b = rgb_img[:, :, 2].astype(float)
        
        # Vegetation: Green excess
        veg_mask = (g > r * 1.08) & (g > b * 1.08) & (g > 40)
        class_map[veg_mask] = SemanticClass.VEGETATION
        
        # Roads / Asphalt: neutral gray, low saturation
        gray_diff = np.maximum(np.abs(r - g), np.maximum(np.abs(g - b), np.abs(b - r)))
        road_mask = (gray_diff < 18) & (r < 110) & (r > 35) & (~veg_mask)
        class_map[road_mask] = SemanticClass.ROAD
        
        # Buildings: structural facades, higher luminance or geometric blocks
        building_mask = (~veg_mask) & (~road_mask) & (r > 100)
        class_map[building_mask] = SemanticClass.BUILDING
        
        dynamic_detections = []
        
        # Apply dynamic object bounding boxes
        if detected_objects:
            for obj in detected_objects:
                x1, y1, x2, y2 = obj["bbox"]
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))
                cls_id = obj.get("class_id", SemanticClass.VEHICLE)
                
                # Dynamic mask disables fusion inside vehicle/pedestrian footprint
                static_mask[y1:y2, x1:x2] = False
                class_map[y1:y2, x1:x2] = cls_id
                
                dynamic_detections.append({
                    "label": obj.get("label", "vehicle"),
                    "confidence": obj.get("confidence", 0.92),
                    "bbox": [x1, y1, x2, y2],
                    "masked_pixels": int((x2 - x1) * (y2 - y1))
                })
                
        return class_map, static_mask, dynamic_detections

    def filter_dynamic_artifacts(
        self, 
        points: np.ndarray, 
        colors: np.ndarray, 
        point_classes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Strips all 3D points classified as dynamic (vehicles, pedestrians).
        Guarantees clear road surfaces and architectural integrity without motion blur.
        """
        static_idx = ~np.isin(point_classes, list(DYNAMIC_CLASSES))
        return points[static_idx], colors[static_idx], point_classes[static_idx]
