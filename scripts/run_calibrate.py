import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from assembly_assistant.utils.logger import get_logger

from assembly_assistant.calibration.workspace_calibrator import WorkspaceCalibrator

BASE_DIR = SRC / "assembly_assistant" / "calibration"
VIDEO_DIR = BASE_DIR / "calibration"
CALIB_OUTPUT_DIR = BASE_DIR / "calibration_data"
CALIB_VIDEO = VIDEO_DIR / "calibration_video.mp4"
DEBUG_DETECTION = True
PREVIEW_DETECTION = False
logger = get_logger(__name__) 

def main():

    if not CALIB_VIDEO.exists():
        raise FileNotFoundError(f"Calibration video not found: {CALIB_VIDEO}")
    
    calibrator = WorkspaceCalibrator()

    logger.info("Running Workspace Calibration")
    calibration_path = calibrator.calibrate_from_video(
        str(CALIB_VIDEO),
        output_dir=str(CALIB_OUTPUT_DIR),
        debug=DEBUG_DETECTION,
        preview=PREVIEW_DETECTION,
    )

    logger.info("Calibration saved at: %s", calibration_path)

if __name__=="__main__":
    main()

    
