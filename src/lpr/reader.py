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

    PLATE_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    PLATE_PATTERN = re.compile(r"[A-Z0-9]{2,}")

    # Characters that OCR commonly confuses with each other.
    # Each group contains visually similar chars; when we have
    # contextual evidence (surrounding chars) we pick the right one.
    CONFUSABLE_DIGIT_TO_LETTER = {
        "0": "O", "1": "I", "2": "Z", "5": "S",
        "6": "G", "8": "B",
    }
    CONFUSABLE_LETTER_TO_DIGIT = {v: k for k, v in CONFUSABLE_DIGIT_TO_LETTER.items()}

    # Common US plate formats: sequences of digit-runs and letter-runs.
    # We use these to decide whether a character "should" be a letter or digit.
    # Patterns are (type, length) tuples: D=digits, L=letters
    PLATE_FORMATS = [
        # 7-char formats
        [("D", 5), ("L", 1), ("D", 1)],  # 93508B3
        [("D", 4), ("L", 1), ("D", 1), ("L", 1)],
        [("D", 1), ("L", 3), ("D", 3)],
        [("D", 3), ("L", 3), ("D", 1)],
        [("L", 3), ("D", 4)],
        [("D", 4), ("L", 3)],
        [("L", 2), ("D", 5)],
        [("D", 5), ("L", 2)],
        [("L", 3), ("D", 3), ("L", 1)],
        [("L", 1), ("D", 3), ("L", 3)],
        [("L", 1), ("D", 5), ("L", 1)],
        [("D", 3), ("L", 1), ("D", 3)],
        # 6-char formats
        [("D", 3), ("L", 3)],
        [("L", 3), ("D", 3)],
        [("D", 2), ("L", 3), ("D", 1)],
        [("L", 2), ("D", 4)],
        [("D", 4), ("L", 2)],
        [("D", 1), ("L", 2), ("D", 3)],
        [("D", 1), ("L", 3), ("D", 2)],
    ]

    # Plate text must be within this length range
    MIN_PLATE_LEN = 4
    MAX_PLATE_LEN = 8

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
        """Read text from a plate image.

        Runs OCR with multiple preprocessing variants, extracts
        plate-length candidates from the results, and applies
        format-aware corrections.
        """
        if self._reader is None:
            self.load_model()

        variants = self._preprocess(plate_image)
        all_candidates: list[PlateText] = []

        for variant in variants:
            try:
                results = self._reader.readtext(
                    variant,
                    allowlist=self.PLATE_CHARS,
                )
            except Exception as e:
                logger.debug("OCR failed: %s", e)
                continue

            if not results:
                continue

            for _bbox, text, conf in results:
                cleaned = self._clean_text(text)
                if not cleaned:
                    continue

                # If the OCR text is already plate-length, use it directly
                if self.MIN_PLATE_LEN <= len(cleaned) <= self.MAX_PLATE_LEN:
                    all_candidates.append(PlateText(
                        text=cleaned, confidence=conf, raw_text=text,
                    ))
                elif len(cleaned) > self.MAX_PLATE_LEN:
                    # Extract all plate-length substrings — the real plate
                    # is often embedded in a longer noisy OCR string
                    for length in range(self.MAX_PLATE_LEN, self.MIN_PLATE_LEN - 1, -1):
                        for start in range(len(cleaned) - length + 1):
                            sub = cleaned[start:start + length]
                            all_candidates.append(PlateText(
                                text=sub, confidence=conf * 0.9,
                                raw_text=text,
                            ))

        if not all_candidates:
            return None

        # Score each candidate: format correction quality + confidence
        scored: list[tuple[float, PlateText, str]] = []
        for cand in all_candidates:
            corrected = self._apply_format_correction(cand.text)
            fmt_score = self._format_match_score(corrected)
            # Composite score: format match quality + OCR confidence
            score = fmt_score * 0.6 + cand.confidence * 0.4
            scored.append((score, cand, corrected))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_cand, corrected = scored[0]

        if corrected != best_cand.text:
            logger.debug(
                "Format correction: %s -> %s", best_cand.text, corrected
            )
        return PlateText(
            text=corrected,
            confidence=best_cand.confidence,
            raw_text=best_cand.raw_text,
        )

    def _preprocess(self, image: np.ndarray) -> list[np.ndarray]:
        """Preprocess plate image into multiple variants for OCR."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Upscale small images
        h, w = gray.shape[:2]
        if w < 200:
            scale = 200 / w
            gray = cv2.resize(
                gray, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )

        variants = []

        # Variant 1: CLAHE + bilateral filter
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.bilateralFilter(enhanced, 11, 17, 17)
        variants.append(denoised)

        # Variant 2: Adaptive threshold (good for high-contrast plates)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 10,
        )
        variants.append(thresh)

        # Variant 3: Otsu threshold
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(otsu)

        return variants

    def _clean_text(self, text: str) -> str:
        """Clean and validate plate text."""
        cleaned = text.upper().strip()
        cleaned = re.sub(r"[^A-Z0-9]", "", cleaned)

        if self.PLATE_PATTERN.search(cleaned):
            return cleaned
        return ""

    def _apply_format_correction(self, text: str) -> str:
        """Apply format-aware character correction.

        Tries to match the text against known plate formats.
        Where a format says "this position should be a letter" but
        we have a digit (or vice versa), swap confusable characters.
        """
        stripped = re.sub(r"[^A-Z0-9]", "", text)
        if not stripped:
            return text

        best_corrected = stripped
        best_score = -1

        for fmt in self.PLATE_FORMATS:
            fmt_len = sum(length for _, length in fmt)
            if fmt_len != len(stripped):
                continue

            corrected = list(stripped)
            score = 0
            pos = 0

            for char_type, length in fmt:
                for i in range(length):
                    ch = corrected[pos]
                    if char_type == "D":
                        if ch.isdigit():
                            score += 1
                        elif ch in self.CONFUSABLE_LETTER_TO_DIGIT:
                            corrected[pos] = self.CONFUSABLE_LETTER_TO_DIGIT[ch]
                            score += 1
                        # else: letter with no digit equivalent, no match
                    elif char_type == "L":
                        if ch.isalpha():
                            score += 1
                        elif ch in self.CONFUSABLE_DIGIT_TO_LETTER:
                            corrected[pos] = self.CONFUSABLE_DIGIT_TO_LETTER[ch]
                            score += 1
                    pos += 1

            if score > best_score:
                best_score = score
                best_corrected = "".join(corrected)

        return best_corrected

    def _format_match_score(self, text: str) -> float:
        """Score how well text matches any known plate format (0.0 to 1.0)."""
        stripped = re.sub(r"[^A-Z0-9]", "", text)
        if not stripped:
            return 0.0

        best = 0.0
        for fmt in self.PLATE_FORMATS:
            fmt_len = sum(length for _, length in fmt)
            if fmt_len != len(stripped):
                continue

            matches = 0
            pos = 0
            for char_type, length in fmt:
                for _ in range(length):
                    ch = stripped[pos]
                    if char_type == "D" and ch.isdigit():
                        matches += 1
                    elif char_type == "L" and ch.isalpha():
                        matches += 1
                    pos += 1

            score = matches / len(stripped)
            if score > best:
                best = score

        # If no format matched the length, give a small base score
        # for being a reasonable length
        if best == 0.0 and self.MIN_PLATE_LEN <= len(stripped) <= self.MAX_PLATE_LEN:
            best = 0.3

        return best
