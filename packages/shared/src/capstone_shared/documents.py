"""Bounded PDF/image preparation. No network, business judgment or persistence."""
from dataclasses import dataclass
from io import BytesIO
import base64
import warnings

from PIL import Image, ImageOps
import pdfplumber
import pypdfium2 as pdfium

Image.MAX_IMAGE_PIXELS = 40_000_000


@dataclass
class Tile:
    id: str
    asset_id: str
    name: str
    page: int
    box: tuple[int, int, int, int]
    source_size: tuple[int, int]
    image: Image.Image

    def location(self, *, precise_box=None, method="TILE") -> dict:
        width, height = self.source_size
        x0, y0, x1, y1 = self.box
        return {"kind": "image", "fileId": self.asset_id, "fileName": self.name,
                "page": self.page, "tileId": self.id, "coordinateSource": method,
                "bbox": precise_box, "region": [x0/width, y0/height, x1/width, y1/height]}

    def data_url(self) -> str:
        # Prepared transient pixels only. Never include this in Graph State or logs.
        image = self.image.copy()
        image.thumbnail((1600, 1600))
        output = BytesIO()
        image.convert("RGB").save(output, format="JPEG", quality=92)
        return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode()


def row_features(image: Image.Image, bins=8, probe=160):
    """Per-row color spread, horizontal edge energy and a coarse RGB profile, read from narrow copies."""
    width, height = image.size
    gray = image.convert("L").resize((probe, height), Image.BOX).tobytes()
    color = image.resize((bins, height), Image.BOX).tobytes()
    spread, edges, profile = [], [], []
    for y in range(height):
        row = gray[y*probe:(y+1)*probe]
        mean = sum(row)/probe
        spread.append((sum((p-mean)**2 for p in row)/probe)**.5)
        edges.append(sum(abs(row[i+1]-row[i]) for i in range(probe-1))/(probe-1))
        profile.append(color[y*bins*3:(y+1)*bins*3])
    return spread, edges, profile


def section_boundaries(spread, edges, profile, flat=3.0, band=12, window=24):
    """Rows where one section of a detail page ends and the next begins, with a strength score.

    A blank band is a strong boundary when the background differs on its two sides (the band before
    it and the band after it have different colors) or when it is clearly taller than the usual gap
    between text lines on this page. Gaps between lines of one paragraph share the background and the
    usual height, so they score low. A second signal catches a photo meeting a colored block without
    a blank band: the rows just above and just below differ strongly."""
    height = len(spread)
    bands = []
    y = 0
    while y < height:
        if spread[y] < flat:
            start = y
            while y < height and spread[y] < flat:
                y += 1
            rows = profile[start:y]
            color = [sum(r[i] for r in rows)/len(rows) for i in range(3)]
            bands.append((start, y, color))
        y += 1
    usual = sorted(e-s for s, e, _ in bands if e-s >= 4)
    usual = usual[len(usual)//2] if len(usual) >= 3 else 24
    shift = lambda a, b: sum(abs(a[i]-b[i]) for i in range(3))/3
    found = {}
    for i, (start, end, color) in enumerate(bands):
        if end-start < band or start == 0 or end >= height:
            continue
        before = bands[i-1][2] if i > 0 else color
        after = bands[i+1][2] if i+1 < len(bands) else color
        tall = min(1.0, max(0.0, (end-start)/usual-1))
        found[(start+end)//2] = 0.5+1.5*tall+2.0*min(1.0, shift(before, after)/30)
    size = len(profile[0])
    totals = [[0]*size]
    for row in profile:
        last = totals[-1]
        totals.append([last[i]+row[i] for i in range(size)])
    mean = lambda a, b: [(totals[b][i]-totals[a][i])/(b-a) for i in range(size)]
    differ = lambda a, b: sum(abs(a[i]-b[i]) for i in range(size))/size
    change = [0.0]*height
    for y in range(2*window, height-2*window, 2):
        change[y] = differ(mean(y-window, y), mean(y, y+window))
    for y in range(2*window, height-2*window, 2):
        if change[y] < 28 or change[y] != max(change[y-window:y+window]) or edges[y] >= 6 \
                or any(abs(y-other) < window for other in found):
            continue
        # Each side must look the same one window further out; a single text line or bar does not.
        steady = max(differ(mean(y-2*window, y-window), mean(y-window, y)),
                     differ(mean(y, y+window), mean(y+window, y+2*window)))
        if steady < change[y]*0.5:
            found[y] = min(2.5, change[y]/30)
    return found


def plan_sections(height, width, found, edges, min_height=None, max_height=None, cost=2.4):
    """Choose cuts that maximize boundary strength minus a cost per cut, inside height limits.
    Weak boundaries are used only when a section would otherwise grow past max_height. Where no
    boundary exists at all, the quietest row of each 80px window is the fallback, so a forced cut
    still avoids text lines."""
    min_height = min_height or max(320, int(.6*width))
    max_height = max_height or max(1600, int(1.5*width))
    if height <= max_height:
        return [0, height]
    scores = dict(found)
    for start in range(80, height-80, 80):
        quiet = min(range(start, min(height, start+80)), key=lambda r: edges[r])
        scores.setdefault(quiet, 0.2)
    points = sorted({0, height, *[y for y in scores if min_height <= y <= height-min_height]})
    best = {0: (0.0, None)}
    for j, point in enumerate(points[1:], 1):
        gain = 0.0 if point == height else scores[point]-cost
        options = [(best[prev][0]+gain, prev) for prev in points[:j]
                   if prev in best and min_height <= point-prev <= max_height]
        if options:
            best[point] = max(options)
    if height not in best:
        return list(range(0, height, max_height))+[height]
    cuts, point = [height], height
    while best[point][1] is not None:
        point = best[point][1]
        cuts.append(point)
    return sorted(cuts)


def tile_image(image: Image.Image, asset_id: str, name: str, page=1) -> list[Tile]:
    """Split a page into content sections (no overlap). A short page stays one section."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, total = image.size
    if width < 32 or total < 32 or width * total > 40_000_000:
        raise ValueError("IMAGE_DIMENSIONS_UNSUPPORTED")
    spread, edges, profile = row_features(image)
    cuts = plan_sections(total, width, section_boundaries(spread, edges, profile), edges)
    result = []
    for index, (top, bottom) in enumerate(zip(cuts, cuts[1:])):
        box = (0, top, width, bottom)
        result.append(Tile(f"{asset_id}:p{page}:t{index}", asset_id, name, page,
                           box, image.size, image.crop(box)))
    if len(result) > 40:
        raise ValueError("TOO_MANY_TILES")
    return result


def prepare_document(raw: bytes, asset: dict) -> tuple[list[dict], list[Tile]]:
    if len(raw) > 25*1024*1024:
        raise ValueError("FILE_TOO_LARGE")
    blocks, tiles = [], []
    if raw.startswith(b"%PDF-"):
        with pdfplumber.open(BytesIO(raw)) as text_pdf, pdfium.PdfDocument(raw) as visual_pdf:
            if not 1 <= len(text_pdf.pages) <= 20:
                raise ValueError("PDF_PAGE_LIMIT")
            for index, page in enumerate(text_pdf.pages):
                for line_no, line in enumerate(page.extract_text_lines()):
                    text = line["text"].strip()
                    if not text:
                        continue
                    blocks.append({"id":f"{asset['id']}:p{index+1}:l{line_no}", "text":text,
                        "location":{"kind":"image", "fileId":asset["id"], "fileName":asset["name"],
                            "page":index+1, "coordinateSource":"PDF_TEXT",
                            "bbox":[line["x0"]/page.width,line["top"]/page.height,line["x1"]/page.width,line["bottom"]/page.height]}})
                pdf_page = visual_pdf[index]
                if pdf_page.get_width()*pdf_page.get_height()*4 > 40_000_000:
                    raise ValueError("PDF_PAGE_PIXELS_LIMIT")
                bitmap = pdf_page.render(scale=2)
                image = bitmap.to_pil().copy()
                bitmap.close()
                pdf_page.close()
                tiles.extend(tile_image(image,asset["id"],asset["name"],index+1))
                if len(tiles) > 40:
                    raise ValueError("TOO_MANY_TILES")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(raw)) as image:
                image.load()
                tiles = tile_image(image,asset["id"],asset["name"])
    return blocks, tiles


EVIDENCE_MAX_PAGES = 80
TEXT_PAGE_MIN_CHARS = 40


def prepare_evidence(raw: bytes, asset: dict, max_pages=EVIDENCE_MAX_PAGES) -> tuple[list[dict], list[Tile], int]:
    """Evidence reading is text-first. Test reports are long native PDFs, so their text layer is read as is
    and only pages without text (scans) become OCR tiles, one per page. Ad limits (20 pages, 40 tiles) do not
    apply. Returns (text blocks, pages to OCR, page count). Errors are specific codes, never a blanket failure."""
    if len(raw) > 25*1024*1024:
        raise ValueError("FILE_TOO_LARGE")
    if not raw.startswith(b"%PDF-"):
        blocks, tiles = prepare_document(raw, asset)
        return blocks, tiles, 1
    blocks, scans = [], []
    with pdfplumber.open(BytesIO(raw)) as text_pdf, pdfium.PdfDocument(raw) as visual_pdf:
        count = len(text_pdf.pages)
        if not 1 <= count <= max_pages:
            raise ValueError("EVIDENCE_PAGE_LIMIT")
        for index, page in enumerate(text_pdf.pages):
            text = (page.extract_text() or "").strip()
            if len(text) >= TEXT_PAGE_MIN_CHARS:
                blocks.append({"id": f"{asset['id']}:p{index+1}", "text": text, "page": index+1})
                continue
            pdf_page = visual_pdf[index]
            bitmap = pdf_page.render(scale=2)
            image = bitmap.to_pil().convert("RGB")
            bitmap.close()
            pdf_page.close()
            scans.append(Tile(f"{asset['id']}:p{index+1}:scan", asset["id"], asset["name"], index+1,
                              (0, 0, *image.size), image.size, image))
    return blocks, scans, count


def local_ocr(tile: Tile, scale: float = 1.0) -> list[dict]:
    """Tesseract confidence belongs to this engine; it is not legal certainty.

    scale < 1 reads a shrunk copy: Tesseract misses very large headline letters at full size. Boxes are mapped
    back to the original pixels either way."""
    import pytesseract
    image = tile.image if scale == 1.0 else tile.image.resize(
        (max(1, round(tile.image.width * scale)), max(1, round(tile.image.height * scale))), Image.LANCZOS)
    data = pytesseract.image_to_data(image, lang="kor+eng", timeout=20,
                                     output_type=pytesseract.Output.DICT)
    if scale != 1.0:
        for name in ("left", "top", "width", "height"):
            data[name] = [round(v / scale) for v in data[name]]
    lines = {}
    for i, text in enumerate(data["text"]):
        if not text.strip():
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(i)
    result = []
    width, height = tile.source_size
    for key, indices in lines.items():
        text = " ".join(data["text"][i] for i in indices)
        left = min(data["left"][i] for i in indices) + tile.box[0]
        top = min(data["top"][i] for i in indices) + tile.box[1]
        right = max(data["left"][i]+data["width"][i] for i in indices)+tile.box[0]
        bottom = max(data["top"][i]+data["height"][i] for i in indices)+tile.box[1]
        # Word boxes let a later step mark the exact problem words inside a line.
        words = [{"text": data["text"][i], "bbox": [(data["left"][i]+tile.box[0])/width, (data["top"][i]+tile.box[1])/height,
                  (data["left"][i]+data["width"][i]+tile.box[0])/width, (data["top"][i]+data["height"][i]+tile.box[1])/height]}
                 for i in indices]
        suffix = "" if scale == 1.0 else f":x{scale:g}"
        result.append({"id":f"{tile.id}:ocr:{key}{suffix}", "text":text, "words":words,
            "ocrConfidence":sum(float(data["conf"][i]) for i in indices)/len(indices),
            "location":tile.location(precise_box=[left/width,top/height,right/width,bottom/height],method="TESSERACT")})
    return result
