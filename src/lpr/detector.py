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
    ) -> np.ndarray | None:
        """Extract the license plate rectangle from a vehicle bounding box.

        Uses contour detection to find bright rectangular regions that
        match plate aspect ratios (typically 2:1 to 5:1 width:height).
        Falls back to the lower-center crop if no contour is found.
        """
        import cv2

        vehicle_crop = frame[y1:y2, x1:x2]
        vh, vw = vehicle_crop.shape[:2]
        if vh < 10 or vw < 10:
            return None

        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)

        # Look for plate-like rectangles via edges + contours
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 200)

        # Dilate to close gaps in plate border
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
        )

        candidates = []
        min_plate_area = vh * vw * 0.005  # plate is at least 0.5% of vehicle
        max_plate_area = vh * vw * 0.25   # and at most 25%

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_plate_area or area > max_plate_area:
                continue

            # Approximate contour to polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

            # Plates are roughly rectangular (4 vertices)
            if len(approx) < 4 or len(approx) > 6:
                continue

            rx, ry, rw, rh = cv2.boundingRect(approx)
            if rh == 0:
                continue

            aspect = rw / rh
            # US plates are roughly 12"x6" = 2:1, but with perspective
            # distortion allow a wider range
            if not (1.5 <= aspect <= 6.0):
                continue

            # Prefer candidates in the lower half of the vehicle
            center_y = ry + rh / 2
            vertical_score = center_y / vh  # higher = lower in image = better

            # Check that the region has decent contrast (plate chars on plate bg)
            roi = gray[ry:ry + rh, rx:rx + rw]
            if roi.size == 0:
                continue
            std_dev = float(np.std(roi))
            if std_dev < 20:
                continue

            candidates.append((rx, ry, rw, rh, vertical_score, std_dev))

        if candidates:
            # Pick the best: prefer lower position + higher contrast
            candidates.sort(key=lambda c: (c[4] * 0.4 + c[5] / 100 * 0.6), reverse=True)
            rx, ry, rw, rh = candidates[0][:4]

            # Add a small margin
            margin_x = int(rw * 0.05)
            margin_y = int(rh * 0.1)
            rx = max(0, rx - margin_x)
            ry = max(0, ry - margin_y)
            rw = min(vw - rx, rw + 2 * margin_x)
            rh = min(vh - ry, rh + 2 * margin_y)

            plate_img = vehicle_crop[ry:ry + rh, rx:rx + rw]
            if plate_img.size > 0:
                logger.debug(
                    "Found plate contour at (%d,%d %dx%d) aspect=%.1f",
                    rx, ry, rw, rh, rw / max(rh, 1),
                )
                return plate_img

        # Fallback: lower-center crop of vehicle
        plate_y1 = int(vh * 0.6)
        plate_y2 = vh
        plate_x1 = int(vw * 0.1)
        plate_x2 = int(vw * 0.9)
        return vehicle_crop[plate_y1:plate_y2, plate_x1:plate_x2]
