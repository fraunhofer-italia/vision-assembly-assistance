import numpy as np
import json
import os

class CalibrationLoader:

    def __init__(self, calibration_path):
        self.calibration_path = calibration_path
        self.H = self.load_homography(calibration_path)

    @staticmethod
    def load_homography(calibration_path):
        """
        Supports:
        - New format: calibration folder containing homography.npy
        - Legacy format: JSON file containing {"homography": ...}
        """
        if os.path.isdir(calibration_path):
            homography_path = os.path.join(calibration_path, "homography.npy")
            if not os.path.exists(homography_path):
                raise FileNotFoundError(f"Missing homography.npy in {calibration_path}")
            return np.load(homography_path)

        if calibration_path.endswith(".json"):
            if not os.path.exists(calibration_path):
                raise FileNotFoundError(calibration_path)
            with open(calibration_path, "r") as f:
                data = json.load(f)
            if "homography" not in data:
                raise ValueError("JSON calibration missing 'homography'")
            return np.array(data["homography"], dtype=float)

        raise FileNotFoundError(f"Unsupported calibration path: {calibration_path}")

    def transform_point(self, x, y):

        p = np.array([x, y, 1.0])
        p_t = self.H @ p

        return p_t[0] / p_t[2], p_t[1] / p_t[2]
    
    def transform_trajectory(self, trajectory):

        transformed = []

        for pt in trajectory:

            x_new, y_new = self.transform_point(pt["x"], pt["y"])

            new_pt = dict(pt)
            new_pt["x"] = x_new
            new_pt["y"] = y_new

            transformed.append(new_pt)

        return transformed
    
    
