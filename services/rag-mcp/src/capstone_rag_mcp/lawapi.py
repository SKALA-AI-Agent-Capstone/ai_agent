"""Statutes, administrative rules and annexes from the 국가법령정보 공동활용 OpenAPI (LAW_OC).

Only the documents listed in reference-sources.json are fetched. Articles become
one chunk each; annex text is split at its numbered items so a citation can name
"별표 5 제2호 가목" instead of the whole annex.
"""
from dataclasses import dataclass, field
from datetime import date
import hashlib
import re
import xml.etree.ElementTree as ET

import httpx

BASE = "https://www.law.go.kr/DRF"
MAX_ANNEX_CHUNK = 1500
ARTICLE_HEAD = re.compile(r"^\s*(제\d+조(?:의\d+)?)\s*\(([^)]*)\)")
ITEM_TOP = re.compile(r"^\s{0,3}(\d{1,2})\.\s+(\S.*)$")
ITEM_SUB = re.compile(r"^\s{0,5}([가-하])\.\s+(\S.*)$")
ITEM_ROW = re.compile(r"^\s*○\s*(\S.*)$")  # 표 형식 별표의 행 (실증 규정 별표)


@dataclass
class LawChunk:
    section: str
    context: str
    body: str


@dataclass
class LawApiDocument:
    target: str
    law_id: str
    title: str
    kind: str
    version_label: str
    issuer: str
    published_on: date | None
    effective_from: date | None
    raw: bytes
    chunks: list[LawChunk] = field(default_factory=list)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.raw).hexdigest()


def _date(value: str | None) -> date | None:
    value = (value or "").strip()
    return date(int(value[:4]), int(value[4:6]), int(value[6:8])) if re.fullmatch(r"\d{8}", value) else None


def _text(node, tag: str) -> str:
    return (node.findtext(tag) or "").strip()


def clean(text: str) -> str:
    text = text.replace("&#9656;", "▸").replace("\r", "")
    lines = [line.rstrip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def split_annex(number: str, title: str, text: str) -> list[LawChunk]:
    """Split annex text at "1." and "가." items. Very long lists (원료 목록) fall back to size windows."""
    label = "별표" if number in ("0", "") else f"별표 {number}"
    lines = clean(text).split("\n")
    chunks, top, sub, row, buffer = [], None, None, None, []

    def flush():
        body = "\n".join(buffer).strip()
        if not body:
            return
        section = label + (f" 제{top}호" if top else "") + (f" {sub}목" if sub else "") + (f" 「{row}」" if row else "")
        for start in range(0, len(body), MAX_ANNEX_CHUNK):
            part = body[start:start + MAX_ANNEX_CHUNK]
            suffix = "" if start == 0 else f" (계속 {start // MAX_ANNEX_CHUNK + 1})"
            chunks.append(LawChunk(section=section + suffix, context=f"{label} {title}", body=part))

    for line in lines:
        top_match, sub_match, row_match = ITEM_TOP.match(line), ITEM_SUB.match(line), ITEM_ROW.match(line)
        if top_match or (sub_match and top) or row_match:
            flush()
            buffer = []
            if top_match:
                top, sub, row = top_match.group(1), None, None
            elif sub_match:
                sub, row = sub_match.group(1), None
            else:
                row = row_match.group(1)[:30]
        buffer.append(line.strip())
    flush()
    return chunks


def law_articles(root, law_title: str) -> list[LawChunk]:
    chunks = []
    for unit in root.iter("조문단위"):
        if _text(unit, "조문여부") not in ("", "조문"):
            continue  # 장, 절 제목 행
        number, branch = _text(unit, "조문번호"), _text(unit, "조문가지번호")
        head = f"제{number}조" + (f"의{branch}" if branch and branch != "0" else "")
        title = _text(unit, "조문제목")
        parts = [_text(unit, "조문내용")]
        for node in unit.iter():
            if node.tag in ("항내용", "호내용", "목내용") and (node.text or "").strip():
                parts.append(node.text.strip())
        body = clean("\n".join(p for p in parts if p))
        if body and not re.search(r"^\s*제\d+조(?:의\d+)?\s*삭제", body):
            chunks.append(LawChunk(section=f"{head} ({title})" if title else head, context=law_title, body=body))
    return chunks


def admrul_articles(root, rule_title: str) -> list[LawChunk]:
    chunks = []
    for node in root.iter("조문내용"):
        body = clean(node.text or "")
        match = ARTICLE_HEAD.match(body)
        if not body or (match and "삭제" in body[:len(match.group(0)) + 6]):
            continue
        if not match and (len(body) < 60 or re.match(r"^제\d+[장절관]", body)):
            continue  # 장, 절 제목
        section = f"{match.group(1)} ({match.group(2)})" if match else "본문"
        chunks.append(LawChunk(section=section, context=rule_title, body=body))
    return chunks


def annexes(root, wanted: set[str] | None) -> list[LawChunk]:
    chunks = []
    for unit in root.iter("별표단위"):
        number = _text(unit, "별표번호").lstrip("0") or "0"
        if (_text(unit, "별표가지번호") or "00") != "00" or _text(unit, "별표구분") not in ("", "별표"):
            continue  # 가지 별표와 서식은 제외
        if wanted is not None and number not in wanted:
            continue
        chunks += split_annex(number, _text(unit, "별표제목"), unit.findtext("별표내용") or "")
    return chunks


class LawApi:
    def __init__(self, oc: str, client: httpx.AsyncClient):
        self.oc, self.client = oc, client

    async def _get(self, path: str, **params) -> tuple[bytes, ET.Element]:
        response = await self.client.get(f"{BASE}/{path}", params={"OC": self.oc, "type": "XML", **params})
        response.raise_for_status()
        root = ET.fromstring(response.content)
        if root.tag in ("Law", "error") or (root.findtext(".//resultCode") or "00") not in ("00", "0", ""):
            raise RuntimeError("LAW_API_REJECTED")
        return response.content, root

    async def law(self, name: str, *, annex_numbers: set[str] | None = None, articles: bool = True) -> LawApiDocument:
        _, found = await self._get("lawSearch.do", target="eflaw", query=name, display=50, nw=3)
        # Several rows may share one 법령ID (현행과 시행 예정 버전). The ID itself must be unique.
        ids = {_text(n, "법령ID") for n in found.iter("law") if _text(n, "법령명한글") == name}
        if len(ids) != 1:
            raise RuntimeError(f"LAW_ID_NOT_UNIQUE:{name}:{len(ids)}")
        law_id = ids.pop()
        raw, root = await self._get("lawService.do", target="eflaw", ID=law_id)
        info = root.find("기본정보") if root.find("기본정보") is not None else root
        published = _text(info, "공포일자")
        chunks = (law_articles(root, name) if articles else []) + annexes(root, annex_numbers or set())
        return LawApiDocument(target="eflaw", law_id=law_id, title=name, kind=_text(info, "법종구분") or "법령",
                              version_label=f"{published[:4]}-{published[4:6]}-{published[6:8]} 제{_text(info, '공포번호')}호",
                              issuer=_text(info, "소관부처") or "법제처", published_on=_date(published),
                              effective_from=_date(_text(info, "시행일자")), raw=raw, chunks=chunks)

    async def admrul(self, name: str, *, annex_numbers: set[str] | None = None) -> LawApiDocument:
        _, found = await self._get("lawSearch.do", target="admrul", query=name, display=50)
        rows = [n for n in found.iter("admrul") if _text(n, "행정규칙명") == name and _text(n, "현행연혁구분") in ("", "현행")]
        if len(rows) != 1:
            raise RuntimeError(f"ADMRUL_NOT_UNIQUE:{name}:{len(rows)}")
        serial = _text(rows[0], "행정규칙일련번호")
        raw, root = await self._get("lawService.do", target="admrul", ID=serial)
        info = root.find("행정규칙기본정보")
        issued = _text(info, "발령일자")
        chunks = admrul_articles(root, name) + annexes(root, annex_numbers)
        return LawApiDocument(target="admrul", law_id=_text(info, "행정규칙ID") or serial, title=name,
                              kind=_text(info, "행정규칙종류") or "고시",
                              version_label=f"{issued[:4]}-{issued[4:6]}-{issued[6:8]} 제{_text(info, '발령번호')}호",
                              issuer=_text(info, "소관부처명") or "식품의약품안전처", published_on=_date(issued),
                              effective_from=_date(_text(info, "시행일자")), raw=raw, chunks=chunks)
