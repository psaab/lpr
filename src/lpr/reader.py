"""License plate text reader using EasyOCR."""

import logging
import re
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger("lpr.reader")


@dataclass
class PlateText:
    """Recognized plate text."""

    text: str
    confidence: float
    raw_text: str


class PlateReader:
    """Reads text from license plate images using EasyOCR."""

    # Common plate text pattern: letters and digits
    PLATE_PATTERN = re.compile(r"[A-Z0-9]{2,}")

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._reader = None

    def load_model(self) -> None:
        """Load the EasyOCR model."""
        import easyocr

        gpu = self.device != "cpu"
        self._reader = easyocr.Reader(
            ["en"],
            gpu=gpu,
            verbose=False,
        )
        logger.info("Loaded EasyOCR model (gpu=%s)", gpu)

    def read(self, plate_image: np.ndarray) -> PlateText | None:
        """Read text from a plate image."""
        if self._reader is None:
            self.load_model()

        processed = self._preprocess(plate_image)

        try:
            results = self._reader.readtext(processed)
        except Exception as e:
            logger.debug("OCR failed: %s", e)
            return None

        if not results:
            return None

        # Combine all detected text regions
        texts = []
        confidences = []
        for _bbox, text, conf in results:
            texts.append(text)
            confidences.append(conf)

        raw_text = " ".join(texts)
        cleaned = self._clean_text(raw_text)

        if not cleaned:
            return None

        avg_confidence = sum(confidences) / len(confidences)

        return PlateText(
            text=cleaned,
            confidence=avg_confidence,
            raw_text=raw_text,
        )

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess plate image for better OCR accuracy."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Resize if too small
        h, w = gray.shape[:2]
        if w < 100:
            scale = 100 / w
            gray = cv2.resize(
                gray, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )

        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Denoise
        denoised = cv2.bilateralFilter(enhanced, 11, 17, 17)

        return denoised

    def _clean_text(self, text: str) -> str:
        """Clean and validate plate text."""
        # Uppercase and remove non-alphanumeric (except spaces/dashes)
        cleaned = text.upper().strip()
        cleaned = re.sub(r"[^A-Z0-9\s\-]", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Common OCR corrections for plates
        corrections = {
            "O": "0", "I": "1", "S": "5", "B": "8",
            "G": "6", "Z": "2", "T": "7",
        }
        # Only apply corrections in digit-heavy contexts
        # (don't blindly replace all O's with 0's)

        # Check if result looks like a plate (at least 2 alphanumeric chars)
        if self.PLATE_PATTERN.search(cleaned):
            return cleaned

        return ""
