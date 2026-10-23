"""OCR engines behind one call: PaddleOCR (PP-OCRv5 Korean) first, Tesseract when it is missing or fails.

One PaddleOCR instance and one lock serve the whole process (the ad reader in the Agent and the evidence
reader share them): the predictor is not thread safe and each instance holds about 1 GB.
OCR_ENGINE=tesseract keeps PaddleOCR out, and so does a missing paddleocr package.
"""
from dataclasses import dataclass
from functools import cache
import os
import threading

from PIL import Image

LOCK = threading.Lock()


@dataclass
class OcrOutput:
    text: str
    confidence: float | None  # 0-100, engine's own score; never legal certainty
    engine: str               # "paddle" | "tesseract"
    fell_back: bool = False   # PaddleOCR was wanted and failed


def paddle_enabled() -> bool:
    if os.getenv("OCR_ENGINE", "paddle").lower() != "paddle":
        return False
    try:
        import paddleocr  # noqa: F401
    except ImportError:
        return False
    return True


@cache
def engine():
    from paddleocr import PaddleOCR
    return PaddleOCR(text_detection_model_name="PP-OCRv5_mobile_det", text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
                     use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)


def paddle_read(image: Image.Image) -> OcrOutput:
    import numpy as np
    array = np.array(image.convert("RGB"))[:, :, ::-1]
    with LOCK:
        results = list(engine().predict(array))
    rows = []
    for result in results:
        for text, score, box in zip(result["rec_texts"], result["rec_scores"], result["rec_boxes"]):
            if str(text).strip():
                rows.append((float(box[1]), float(box[0]), str(text), float(score)))
    rows.sort()
    confidence = sum(row[3] for row in rows) / len(rows) * 100 if rows else None
    return OcrOutput("\n".join(row[2] for row in rows), confidence, "paddle")


def tesseract_read(image: Image.Image) -> OcrOutput:
    from .documents import Tile, local_ocr
    width, height = image.size
    lines = local_ocr(Tile("ocr", "ocr", "ocr", 1, (0, 0, width, height), (width, height), image.convert("RGB")))
    lines.sort(key=lambda line: (line["location"]["bbox"][1], line["location"]["bbox"][0]))
    confidences = [line["ocrConfidence"] for line in lines]
    return OcrOutput("\n".join(line["text"] for line in lines),
                     sum(confidences) / len(confidences) if confidences else None, "tesseract")


def read_image(image: Image.Image, prefer: str = "auto") -> OcrOutput:
    """prefer="tesseract" skips PaddleOCR (the cheap engine for short time limits); "auto" tries it first."""
    if prefer != "tesseract" and paddle_enabled():
        try:
            return paddle_read(image)
        except Exception:
            output = tesseract_read(image)
            output.fell_back = True
            return output
    return tesseract_read(image)
