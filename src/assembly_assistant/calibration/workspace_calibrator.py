import cv2
import numpy as np
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from assembly_assistant.utils.logger import get_logger

logger = get_logger(__name__)

class WorkspaceCalibrator:

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = Path(__file__).resolve().parent / "marker_config.json"
        else:
            config_path = Path(config_path)

        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.aruco_dict = self._get_aruco_dict(
            self.config["aruco_dictionary"]
        )

        self.detector = self._build_detector(self.aruco_dict)

    def _get_aruco_dict(self, name):
        return cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, name)
        )

    def _build_detector(self, aruco_dict):
        # OpenCV >=4.7: ArucoDetector class
        if hasattr(cv2.aruco, "ArucoDetector"):
            try:
                return cv2.aruco.ArucoDetector(aruco_dict)
            except TypeError:
                params = cv2.aruco.DetectorParameters()
                return cv2.aruco.ArucoDetector(aruco_dict, params)
        # OpenCV <4.7: functional API
        return None

    def _detect_markers(self, frame):
        if self.detector is not None:
            return self.detector.detectMarkers(frame)
        if hasattr(cv2.aruco, "DetectorParameters_create"):
            params = cv2.aruco.DetectorParameters_create()
        else:
            params = cv2.aruco.DetectorParameters()
        return cv2.aruco.detectMarkers(frame, self.aruco_dict, parameters=params)
    
    def calibrate_from_video(
        self,
        video_path,
        output_dir="calibration_data",
        debug=False,
        preview=False,
    ):

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError("Cannot open video")
        
        collected_corners = []
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_indices = np.linspace(0, frame_count - 1, 20).astype(int)

        expected_ids = set(self.config["workspace_corners"].keys())
        frames_with_any_marker = 0
        frames_with_all_markers = 0

        for idx in sample_indices:

            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if not ret:
                continue

            corners, ids, _ = self._detect_markers(frame)

            if ids is None:
                if debug:
                    logger.info("[Calibration][Frame %d] No markers detected", idx)
                continue

            ids = ids.flatten()
            detected_ids = {str(marker_id) for marker_id in ids}
            frames_with_any_marker += 1

            frame_map = {}

            for i, marker_id in enumerate(ids):
                if str(marker_id) in self.config["workspace_corners"]:
                    frame_map[str(marker_id)] = corners[i][0]

            if debug:
                missing = sorted(list(expected_ids - detected_ids))
                logger.info(
                    "[Calibration][Frame %d] detected=%s, usable=%s, missing_expected=%s",
                    idx,
                    sorted(list(detected_ids)),
                    sorted(list(frame_map.keys())),
                    missing,
                )

            if preview:
                vis = frame.copy()
                cv2.aruco.drawDetectedMarkers(vis, corners, ids.reshape(-1, 1))
                cv2.imshow("Calibration Marker Detection", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    preview = False

            if len(frame_map) == 4:
                collected_corners.append(frame_map)
                frames_with_all_markers += 1

        cap.release()
        if preview:
            cv2.destroyAllWindows()

        if len(collected_corners) == 0:
            raise RuntimeError(
                "Markers not detected in video. "
                f"dictionary={self.config['aruco_dictionary']}, "
                f"frames_with_any_marker={frames_with_any_marker}, "
                f"frames_with_all_4_markers={frames_with_all_markers}"
            )
        
        H = self._compute_homography(collected_corners)

        calibration_id = str(uuid.uuid4())[:8]
        save_path = os.path.join(output_dir, f"calibration_{calibration_id}")
        os.makedirs(save_path, exist_ok=True)

        np.save(os.path.join(save_path,  "homography.npy"), H)

        meta = {
            "calibration_id": calibration_id,
            "created_at": datetime.utcnow().isoformat(),
            "video_source": video_path,
            "markers_used": list(self.config["workspace_corners"].keys()),
            "frame_size": [frame_width, frame_height]
        }

        with open(os.path.join(save_path, "calibration_meta.json"), "w") as f:
            json.dump(meta, f, indent=4)

        logger.info("[Calibration] Saved at %s", save_path)

        return save_path
    
    def _compute_homography(self, collected_frames):

        image_pts = []
        workspace_pts = []

        marker_corner_index = self.config["marker_corner_index"]

        for marker_id, workspace_coord in self.config["workspace_corners"].items():

            avg_point = []

            for frame in collected_frames:
                marker_corners = frame[marker_id]
                avg_point.append(marker_corners[marker_corner_index])

            avg_point = np.mean(avg_point, axis=0)

            image_pts.append(avg_point)
            workspace_pts.append(workspace_coord)

        image_pts = np.array(image_pts, dtype=np.float32)
        workspace_pts = np.array(workspace_pts, dtype=np.float32)

        H, _ = cv2.findHomography(image_pts, workspace_pts)

        return H
