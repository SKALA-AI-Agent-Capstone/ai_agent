"""OCR engines behind one call: PaddleOCR (PP-OCRv5 Korean) first, Tesseract when it is missing or fails.

A small pool of PaddleOCR instances serves the whole process (the ad reader in the Agent and the evidence
reader share it). A predictor is not thread safe, so each instance runs one image at a time. OCR_ENGINES (default 3)
sets the pool size: on the 17-section Pyunkang ad, 1 instance took 40s, 2 took 25s and 3 took 19s with the same text.
OCR_ENGINE=tesseract keeps PaddleOCR out, and so does a missing paddleocr package.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
import os
import queue
import threading

from PIL import Image

LOCK = threading.Lock()  # guards pool creation
_POOL: queue.Queue | None = None


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


def _create():
    from paddleocr import PaddleOCR
    return PaddleOCR(text_detection_model_name="PP-OCRv5_mobile_det", text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
                     use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
                     cpu_threads=max(1, (os.cpu_count() or 4) // pool_size()))


def pool_size() -> int:
    return max(1, int(os.getenv("OCR_ENGINES", "3")))


@cache
def engine():
    """The first instance, created on its own (warm-up and callers that need just one)."""
    return _create()


@contextmanager
def slot():
    """Borrow one instance for one prediction. Instances are created on first need, up to OCR_ENGINES."""
    global _POOL
    with LOCK:
        if _POOL is None:
            _POOL = queue.Queue()
            _POOL.put(engine())
            for _ in range(pool_size() - 1):
                _POOL.put(None)  # created lazily by the first borrower
    item = _POOL.get()
    try:
        if item is None:
            item = _create()
        yield item
    finally:
        _POOL.put(item)


def paddle_read(image: Image.Image) -> OcrOutput:
    import numpy as np
    array = np.array(image.convert("RGB"))[:, :, ::-1]
    with slot() as instance:
        results = list(instance.predict(array))
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
