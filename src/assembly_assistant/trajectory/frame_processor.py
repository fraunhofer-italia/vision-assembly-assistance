import numpy as np

import json
import os
 
 
class FramePreprocessor:

    def __init__(self, calibration_path):
        if os.path.isdir(calibration_path):
            homography_path = os.path.join(calibration_path, "homography.npy")
            if not os.path.exists(homography_path):
                raise FileNotFoundError(f"Missing homography.npy in {calibration_path}")
            self.H = np.load(homography_path).astype(float)
        else:
            with open(calibration_path, "r") as f:
                data = json.load(f)
            self.H = np.array(data["homography"], dtype=float)
 
    # -------------------------------------------------

    # Warp single point

    # -------------------------------------------------

    def warp_point(self, pt):

        p = np.array([pt[0], pt[1], 1.0])

        p_warp = self.H @ p

        p_warp /= p_warp[2]

        return p_warp[:2]
 
    # -------------------------------------------------

    # Warp all keypoints

    # -------------------------------------------------

    def warp_keypoints(self, keypoints):

        return np.array([self.warp_point(kp) for kp in keypoints])
 
    # -------------------------------------------------

    # Warp wrist

    # -------------------------------------------------

    def warp_wrist(self, wrist):

        return self.warp_point(wrist)
 
    # -------------------------------------------------

    # Warp tool bbox center

    # -------------------------------------------------

    def warp_tool_bbox(self, bbox):

        x1, y1, x2, y2 = bbox

        center = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]

        return self.warp_point(center)

 
