"""
SFM-X Georeferencing & Spatial Accuracy Engine
Transforms local 3D Cartesian coordinates (ENU) into global geospatial coordinates
(WGS84 Latitude/Longitude/Altitude and UTM).
Validates metric accuracy target (<= 1.0m spatial error).
"""

import numpy as np
import math
from typing import Dict, Any, Tuple

# WGS84 Ellipsoid Constants
WGS84_A = 6378137.0         # semi-major axis (meters)
WGS84_F = 1.0 / 298.257223563  # flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_E2 = 2.0 * WGS84_F - WGS84_F ** 2  # first eccentricity squared

class GeoreferencingEngine:
    def __init__(
        self, 
        origin_lat: float = 26.9124, 
        origin_lon: float = 75.7873, 
        origin_alt: float = 431.0
    ):
        """
        Initializes geodetic origin for Local East-North-Up (ENU) tangent plane.
        Default origin: Jaipur, India (SIH Smart City benchmark).
        """
        self.lat0 = origin_lat
        self.lon0 = origin_lon
        self.alt0 = origin_alt
        
        # Precompute trigonometric constants
        self.phi0 = math.radians(origin_lat)
        self.lam0 = math.radians(origin_lon)
        self.sin_phi0 = math.sin(self.phi0)
        self.cos_phi0 = math.cos(self.phi0)
        self.sin_lam0 = math.sin(self.lam0)
        self.cos_lam0 = math.cos(self.lam0)
        
        # Origin in ECEF
        self.ecef0 = self.wgs84_to_ecef(self.lat0, self.lon0, self.alt0)

    @staticmethod
    def wgs84_to_ecef(lat: float, lon: float, alt: float) -> Tuple[float, float, float]:
        """Converts WGS84 geodetic coordinates to Earth-Centered, Earth-Fixed (ECEF) Cartesian coordinates."""
        phi = math.radians(lat)
        lam = math.radians(lon)
        s_phi = math.sin(phi)
        c_phi = math.cos(phi)
        
        N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * (s_phi ** 2))
        x = (N + alt) * c_phi * math.cos(lam)
        y = (N + alt) * c_phi * math.sin(lam)
        z = (N * (1.0 - WGS84_E2) + alt) * s_phi
        return x, y, z

    def ecef_to_enu(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """Converts ECEF Cartesian point to local East-North-Up (ENU) offset in meters."""
        dx = x - self.ecef0[0]
        dy = y - self.ecef0[1]
        dz = z - self.ecef0[2]
        
        e = -self.sin_lam0 * dx + self.cos_lam0 * dy
        n = -self.sin_phi0 * self.cos_lam0 * dx - self.sin_phi0 * self.sin_lam0 * dy + self.cos_phi0 * dz
        u = self.cos_phi0 * self.cos_lam0 * dx + self.cos_phi0 * self.sin_lam0 * dy + self.sin_phi0 * dz
        return e, n, u

    def wgs84_to_enu(self, lat: float, lon: float, alt: float) -> Tuple[float, float, float]:
        """Direct transformation from WGS84 GPS to local ENU meters."""
        x, y, z = self.wgs84_to_ecef(lat, lon, alt)
        return self.ecef_to_enu(x, y, z)

    def enu_to_wgs84(self, east: float, north: float, up: float) -> Tuple[float, float, float]:
        """
        Converts local 3D model point (East, North, Up meters) back to WGS84 (Lat, Lon, Alt).
        Used for real-time cursor GPS inspection in 3D viewer.
        """
        # ENU to ECEF delta
        dx = -self.sin_lam0 * east - self.sin_phi0 * self.cos_lam0 * north + self.cos_phi0 * self.cos_lam0 * up
        dy =  self.cos_lam0 * east - self.sin_phi0 * self.sin_lam0 * north + self.cos_phi0 * self.sin_lam0 * up
        dz =  self.cos_phi0 * north + self.sin_phi0 * up
        
        x = self.ecef0[0] + dx
        y = self.ecef0[1] + dy
        z = self.ecef0[2] + dz
        
        # ECEF to Geodetic (Bowring's iterative method)
        p = math.sqrt(x**2 + y**2)
        theta = math.atan2(z * WGS84_A, p * WGS84_B)
        
        lat = math.atan2(
            z + (WGS84_E2 / (1.0 - WGS84_E2)) * WGS84_B * (math.sin(theta)**3),
            p - WGS84_E2 * WGS84_A * (math.cos(theta)**3)
        )
        lon = math.atan2(y, x)
        
        s_lat = math.sin(lat)
        N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * (s_lat**2))
        alt = (p / math.cos(lat)) - N
        
        return math.degrees(lat), math.degrees(lon), alt

    def compute_accuracy_report(
        self, 
        ekf_horiz_err: float, 
        ekf_vert_err: float, 
        drone_alt_m: float = 50.0,
        focal_length_mm: float = 24.0,
        sensor_width_mm: float = 17.3,
        image_width_px: int = 3840
    ) -> Dict[str, Any]:
        """
        Computes Ground Sample Distance (GSD) and total spatial RMSE.
        Validates the <= 1.0 m spatial accuracy hackathon requirement.
        """
        # GSD formula: (Altitude * SensorWidth) / (FocalLength * ImageWidth)
        gsd_cm = ((drone_alt_m * 1000.0) * sensor_width_mm) / (focal_length_mm * image_width_px) * 0.1
        
        # 3D Spatial RMSE
        rmse_3d = math.sqrt(ekf_horiz_err**2 + ekf_vert_err**2)
        passed_target = rmse_3d <= 1.0
        
        return {
            "origin": {
                "latitude": self.lat0,
                "longitude": self.lon0,
                "altitude_m": self.alt0
            },
            "gsd_cm_per_pixel": round(gsd_cm, 2),
            "est_horizontal_rmse_m": round(ekf_horiz_err, 3),
            "est_vertical_rmse_m": round(ekf_vert_err, 3),
            "est_3d_spatial_rmse_m": round(rmse_3d, 3),
            "target_threshold_m": 1.0,
            "target_achieved": passed_target,
            "crs": "EPSG:4326 (WGS84) / Local ENU Tangent Plane",
            "utm_zone": f"{int((self.lon0 + 180) / 6) + 1}N"
        }
