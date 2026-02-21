"""Vehicle attribute detection (color, type, make/model) from plate region."""

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("lpr.vehicle")

_BARRIER_URL = (
    "https://storage.googleapis.com/ailia-models/"
    "vehicle-attributes-recognition-barrier/"
    "vehicle-attributes-recognition-barrier-0042.onnx"
)
_BARRIER_FILENAME = "vehicle-attributes-recognition-barrier-0042.onnx"


@dataclass
class VehicleAttributes:
    color: str
    color_confidence: float
    vehicle_type: str
    type_confidence: float
    make_model: str | None = None
    make_model_confidence: float = 0.0


class VehicleClassifier:
    """Classifies vehicle color and type from a frame region near a plate."""

    COLORS = ["white", "gray", "yellow", "red", "green", "blue", "black"]
    TYPES = ["car", "van", "truck", "bus"]

    def __init__(
        self,
        device: str = "cpu",
        cache_dir: Path | None = None,
        make_model_path: Path | None = None,
        make_model_labels_path: Path | None = None,
    ) -> None:
        self._device = device
        self._cache_dir = cache_dir or Path.home() / ".cache" / "lpr"
        self._make_model_path = make_model_path
        self._make_model_labels_path = make_model_labels_path
        self._session = None
        self._mm_session = None
        self._mm_labels: list[str] = []

    def _download_model(self) -> Path:
        """Download barrier-0042 ONNX model if not cached."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = self._cache_dir / _BARRIER_FILENAME
        if model_path.exists():
            logger.debug("Using cached model: %s", model_path)
            return model_path

        logger.info("Downloading vehicle attribute model to %s ...", model_path)
        urllib.request.urlretrieve(_BARRIER_URL, model_path)
        logger.info("Download complete: %s", model_path)
        return model_path

    def load_model(self) -> None:
        """Create ONNX Runtime sessions for vehicle attribute models."""
        import onnxruntime as ort

        model_path = self._download_model()

        providers = ["CPUExecutionProvider"]
        if self._device != "cpu":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self._session = ort.InferenceSession(str(model_path), providers=providers)
        active = self._session.get_providers()
        logger.info(
            "Vehicle attribute model loaded (providers: %s)", ", ".join(active)
        )

        if self._make_model_path and self._make_model_labels_path:
            self._mm_session = ort.InferenceSession(
                str(self._make_model_path), providers=providers
            )
            self._mm_labels = (
                self._make_model_labels_path.read_text().strip().splitlines()
            )
            logger.info(
                "Make/model classifier loaded (%d labels)", len(self._mm_labels)
            )

    @staticmethod
    def estimate_vehicle_roi(
        plate_bbox: tuple[int, int, int, int],
        frame_shape: tuple[int, ...],
    ) -> tuple[int, int, int, int]:
        """Estimate the vehicle bounding box from the plate location.

        The plate is typically at the bottom-center of the vehicle.
        Expand ~6x plate width horizontally (centered) and ~6x height
        upward / ~1x downward, clamped to frame bounds.
        """
        px1, py1, px2, py2 = plate_bbox
        pw = px2 - px1
        ph = py2 - py1
        pcx = (px1 + px2) // 2

        vw = pw * 6
        vh_up = ph * 6
        vh_down = ph

        h, w = frame_shape[:2]
        vx1 = max(0, pcx - vw // 2)
        vx2 = min(w, pcx + vw // 2)
        vy1 = max(0, py1 - vh_up)
        vy2 = min(h, py2 + vh_down)

        return (vx1, vy1, vx2, vy2)

    def classify(
        self,
        frame: np.ndarray,
        plate_bbox: tuple[int, int, int, int],
    ) -> VehicleAttributes | None:
        """Classify vehicle attributes from the frame region around a plate."""
        if self._session is None:
            return None

        vx1, vy1, vx2, vy2 = self.estimate_vehicle_roi(plate_bbox, frame.shape)
        crop = frame[vy1:vy2, vx1:vx2]
        if crop.size == 0:
            return None

        # Preprocess: resize to 72x72, normalize to [0,1], NCHW
        resized = cv2.resize(crop, (72, 72))
        blob = resized.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 72, 72)

        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: blob})

        # Map output heads by name rather than assuming index order
        output_names = [o.name for o in self._session.get_outputs()]
        output_map = dict(zip(output_names, outputs))

        color_probs = output_map["color"].flatten()
        type_probs = output_map["type"].flatten()

        color_idx = int(np.argmax(color_probs))
        type_idx = int(np.argmax(type_probs))

        attrs = VehicleAttributes(
            color=self.COLORS[color_idx],
            color_confidence=float(color_probs[color_idx]),
            vehicle_type=self.TYPES[type_idx],
            type_confidence=float(type_probs[type_idx]),
        )

        if self._mm_session is not None:
            self._classify_make_model(crop, attrs)

        return attrs

    def _classify_make_model(
        self, vehicle_crop: np.ndarray, attrs: VehicleAttributes
    ) -> None:
        """Run optional make/model classifier on the vehicle crop."""
        meta = self._mm_session.get_inputs()[0]
        _, _, h, w = meta.shape
        resized = cv2.resize(vehicle_crop, (w, h))
        blob = resized.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]

        outputs = self._mm_session.run(None, {meta.name: blob})
        probs = outputs[0].flatten()
        idx = int(np.argmax(probs))
        attrs.make_model = self._mm_labels[idx]
        attrs.make_model_confidence = float(probs[idx])
