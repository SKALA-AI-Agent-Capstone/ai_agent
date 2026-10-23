"""PaddleOCR (PP-OCRv5 Korean) lines for ad copy: where text is and what it literally says.

Measured on the Pyunkang ad (4 sections, 47 hand-checked lines): character error 2.3% against Tesseract's 61%,
and on five real ads it placed 168 of 172 copy blocks against Tesseract's 78 of 134
(2026-10-22 comparison).
Its text is used for position and for checking the vision read, never as the wording on a card.

The engines are shared with the evidence reader through capstone_shared.ocr_engine: a small pool, each instance
running one image at a time because a predictor is not thread safe.
OCR_ENGINE=tesseract, or a missing paddleocr package, falls back to Tesseract in ad_reading.
"""
from capstone_shared import ocr_engine
from capstone_shared.ocr_engine import engine, paddle_enabled as enabled  # noqa: F401  (re-exported)



def split_words(text: str, box: list[float]) -> list[dict]:
    """PaddleOCR boxes whole lines. Word boxes are cut from the line by character share, enough to mark a span."""
    words = text.split()
    total = sum(len(w) for w in words) + max(0, len(words) - 1)
    if not words or total == 0:
        return []
    x0, y0, x1, y1 = box
    out, used = [], 0
    for word in words:
        left = x0 + (x1 - x0) * used / total
        used += len(word)
        out.append({"text": word, "bbox": [left, y0, x0 + (x1 - x0) * used / total, y1]})
        used += 1
    return out


def paddle_lines(tile) -> list[dict]:
    """Same shape as capstone_shared.documents.local_ocr: page-ratio boxes, word boxes, engine confidence."""
    import numpy as np
    width, height = tile.source_size
    image = np.array(tile.image.convert("RGB"))[:, :, ::-1]
    with ocr_engine.slot() as instance:
        results = list(instance.predict(image))
    lines = []
    for result in results:
        for k, (text, score, box) in enumerate(zip(result["rec_texts"], result["rec_scores"], result["rec_boxes"])):
            if not str(text).strip():
                continue
            x0, y0, x1, y1 = [float(v) for v in box]
            bbox = [(x0 + tile.box[0]) / width, (y0 + tile.box[1]) / height,
                    (x1 + tile.box[0]) / width, (y1 + tile.box[1]) / height]
            lines.append({"id": f"{tile.id}:pocr:{k}", "text": str(text), "ocrConfidence": float(score) * 100,
                          "words": split_words(str(text), bbox), "location": tile.location(precise_box=bbox, method="PADDLE_OCR")})
    return lines
