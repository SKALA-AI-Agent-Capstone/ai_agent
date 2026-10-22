"""Preserve the S01 annex tables as rows, including notes and cross references."""
from pathlib import Path
import re


ANNEX_HEADING = re.compile(r"^\[별표\s*([12])\]")
ANNEX_REFERENCE = re.compile(r"\[별표\s*2\]\s*(\d+)\.")


def _clean(value: str | None) -> str:
    return re.sub(r"[ \t]+", " ", (value or "").replace("\r", "")).strip()


def _continues_previous(first_row: bool, group: str, expression: str) -> bool:
    """A row cut at a page break resumes with an empty group cell and no '·' item marker."""
    return first_row and not group and not expression.startswith("·")


def s01_annex_rows(path: Path) -> list[dict]:
    """Extract one searchable chunk per table row; keep a row split across pages as one chunk."""
    import pdfplumber

    rows: list[dict] = []
    annex = category = None
    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, 1):
            heading = ANNEX_HEADING.search((page.extract_text() or "")[:180])
            if heading:
                annex, category = heading.group(1), None
            if annex is None:
                continue
            tables = page.extract_tables()
            if not tables:
                annex = category = None
                continue
            first_row = True
            for table in tables:
                if not table or len(table[0]) != 3 or _clean(table[0][0]) != "구 분":
                    continue
                for raw in table[1:]:
                    group, expression, note = (_clean(cell) for cell in raw)
                    previous = rows[-1] if rows and rows[-1]["annex"] == annex else None
                    if previous and (not expression or _continues_previous(first_row, group, expression)):
                        previous["expressions"] += [expression] if expression else []
                        previous["notes"] += [note] if note else []
                        previous["page_until"] = page_number
                        first_row = False
                        continue
                    first_row = False
                    if group:
                        category = group.replace("\n", " ")
                    if not expression or not category:
                        continue
                    rows.append({"annex": annex, "section": f"[별표 {annex}] > {category}"[:90],
                                 "expressions": [expression], "notes": [note] if note else [],
                                 "page_from": page_number, "page_until": page_number})
    chunks = []
    for row in rows:
        label = "금지 표현" if row["annex"] == "1" else "실증 대상"
        body = f"{label}: " + "\n".join(row["expressions"])
        if row["notes"]:
            body += "\n비고: " + "\n".join(row["notes"])
        chunks.append({"section": row["section"], "body": body,
                       "page_from": row["page_from"], "page_until": row["page_until"]})
    return chunks


def annex_number(section: str | None) -> tuple[str, str] | None:
    match = re.match(r"^\[별표\s*([12])\] > (\d+)\.", section or "")
    return match.groups() if match else None
