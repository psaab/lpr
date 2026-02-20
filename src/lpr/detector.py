"""License plate detector using YOLOv8."""

import logging
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger("lpr.detector")


@dataclass
class Detection:
    """A detected license plate region."""

    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    plate_image: np.ndarray


class PlateDetector:
    """Detects license plates in video frames using YOLOv8."""

    # COCO class indices that might contain vehicles/plates
    # We'll use a license-plate-specific model if available,
    # otherwise fall back to general object detection
    VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck

    def __init__(self, device: str = "cpu", confidence: float = 0.5):
        self.device = device
        self.confidence = confidence
        self._model: YOLO | None = None
        self._is_plate_model = False

    def load_model(self) -> None:
        """Load the YOLO model."""
        # Try to load a license plate detection model first
        # Fall back to general YOLOv8 for vehicle detection
        try:
            self._model = YOLO("yolov8n.pt")
            self._is_plate_model = False
            logger.info(
                "Loaded YOLOv8n model on device=%s (vehicle detection mode)",
                self.device,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}") from e

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect license plates (or vehicles) in a frame."""
        if self._model is None:
            self.load_model()

        results = self._model(
            frame,
            device=self.device,
            conf=self.confidence,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                cls = int(boxes.cls[i])
                conf = float(boxes.conf[i])

                # If using general model, only look at vehicles
                if not self._is_plate_model and cls not in self.VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())

                # Ensure valid crop region
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                if self._is_plate_model:
                    plate_img = frame[y1:y2, x1:x2]
                else:
                    # For vehicle detections, try to find the plate region
                    # within the lower portion of the vehicle bounding box
                    plate_img = self._extract_plate_region(frame, x1, y1, x2, y2)

                if plate_img is not None and plate_img.size > 0:
                    detections.append(Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        plate_image=plate_img,
                    ))

        return detections

    def _extract_plate_region(
        self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> np.ndarray:
        """Extract the likely plate region from a vehicle bounding box.

        License plates are typically in the lower third of a vehicle.
        """
        vehicle_h = y2 - y1
        vehicle_w = x2 - x1

        # Focus on lower 40% of vehicle, middle 80% width
        plate_y1 = y1 + int(vehicle_h * 0.6)
        plate_y2 = y2
        plate_x1 = x1 + int(vehicle_w * 0.1)
        plate_x2 = x2 - int(vehicle_w * 0.1)

        return frame[plate_y1:plate_y2, plate_x1:plate_x2]
