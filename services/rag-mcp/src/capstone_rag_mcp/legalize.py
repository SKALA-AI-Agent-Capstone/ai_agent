"""Statute text from legalize-kr (https://github.com/legalize-kr/legalize-kr).

Each law is a Markdown file and each amendment is a commit dated on its
promulgation day. We keep the version in force on the review date and any
promulgated version that is not yet in force, so search can filter by date.
"""
from dataclasses import dataclass, field
from datetime import date, timedelta
import hashlib
import os
import re
from urllib.parse import quote

import httpx

REPOSITORY = "legalize-kr/legalize-kr"
API = f"https://api.github.com/repos/{REPOSITORY}"
RAW = f"https://raw.githubusercontent.com/{REPOSITORY}"
MAX_SECTION_CHARS = 1800

HEADING = re.compile(r"^(#{1,5}) (.+)$")


@dataclass
class LawChunk:
    section: str
    context: str
    body: str


@dataclass
class LawVersion:
    path: str
    commit: str
    meta: dict
    text: str
    effective_from: date | None
    effective_until: date | None = None
    chunks: list[LawChunk] = field(default_factory=list)

    @property
    def title(self) -> str:
        return self.meta.get("제목", self.path)

    @property
    def version_label(self) -> str:
        return f"{self.meta.get('공포일자', 'unknown')} 제{self.meta.get('공포번호', '?')}호"

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def storage_key(self) -> str:
        return f"legalize-kr@{self.commit}:{self.path}"


def parse_front_matter(text: str) -> tuple[dict, str]:
    """Read top-level scalars and plain string lists (소관부처). Nested maps such as 첨부파일 are skipped."""
    if not text.startswith("---\n"):
        return {}, text
    end = text.index("\n---\n", 4)
    meta: dict = {}
    key = None
    for line in text[4:end].splitlines():
        match = re.match(r"^([^\s:-][^:]*): ?(.*)$", line)
        if match:
            key = match.group(1)
            if match.group(2):
                meta[key] = match.group(2).strip().strip("'\"")
            continue
        item = re.match(r"^- ([^:]+)$", line)
        if key and item:
            meta.setdefault(key, [])
            if isinstance(meta[key], list):
                meta[key].append(item.group(1).strip())
    return meta, text[end + 5:]


def parse_date(value: str | None) -> date | None:
    try:
        return date.fromisoformat(value) if value else None
    except ValueError:
        return None


def clean(line: str) -> str:
    return line.replace("**", "").replace("\\.", ".").rstrip()


def split_long(body: str) -> list[str]:
    if len(body) <= MAX_SECTION_CHARS:
        return [body]
    parts, current = [], ""
    for paragraph in body.split("\n\n"):
        if current and len(current) + len(paragraph) > MAX_SECTION_CHARS:
            parts.append(current)
            current = ""
        current = f"{current}\n\n{paragraph}" if current else paragraph
    return parts + [current] if current else parts


def chunk_law(markdown: str) -> list[LawChunk]:
    """One chunk per article (조). Supplementary provisions (부칙) and deleted articles are skipped."""
    _, body = parse_front_matter(markdown)
    path: dict[int, str] = {}
    chunks: list[LawChunk] = []
    article: str | None = None
    lines: list[str] = []

    def flush():
        text = "\n".join(lines).strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        if article and text and not text.startswith("삭제"):
            context = " > ".join(path[level] for level in sorted(path))
            for index, part in enumerate(split_long(text)):
                section = article if index == 0 else f"{article} (계속 {index + 1})"
                chunks.append(LawChunk(section=section, context=context, body=part))

    for raw in body.splitlines():
        heading = HEADING.match(raw)
        if heading:
            flush()
            article, lines = None, []
            level, title = len(heading.group(1)), clean(heading.group(2))
            if level == 2 and title.startswith("부칙"):
                break
            if level == 5:
                article = title
            else:
                path = {k: v for k, v in path.items() if k < level}
                path[level] = title
            continue
        if article is not None:
            lines.append(clean(raw))
    flush()
    return chunks


def assign_effective_ranges(versions: list[LawVersion]) -> list[LawVersion]:
    """Return newest first. Each older version ends the day before the next one takes effect."""
    ordered = sorted(versions, key=lambda v: v.effective_from or date.min)
    for older, newer in zip(ordered, ordered[1:]):
        if newer.effective_from:
            older.effective_until = newer.effective_from - timedelta(days=1)
    return list(reversed(ordered))


def github_headers() -> dict:
    token = os.getenv("GITHUB_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


async def fetch_versions(path: str, *, as_of: date, max_versions: int = 10) -> list[LawVersion]:
    """Fetch versions newest first until one in force on as_of is found."""
    versions: list[LawVersion] = []
    async with httpx.AsyncClient(timeout=30, headers=github_headers()) as client:
        response = await client.get(f"{API}/commits", params={"path": path, "per_page": max_versions})
        if response.status_code in (403, 429) and "Authorization" not in client.headers:
            raise RuntimeError("GitHub API rate limit. Set GITHUB_TOKEN (read-only public access is enough).")
        response.raise_for_status()
        for commit in response.json():
            sha = commit["sha"]
            raw = await client.get(f"{RAW}/{sha}/{quote(path)}")
            raw.raise_for_status()
            meta, _ = parse_front_matter(raw.text)
            version = LawVersion(path=path, commit=sha, meta=meta, text=raw.text,
                                 effective_from=parse_date(meta.get("시행일자")))
            version.chunks = chunk_law(raw.text)
            versions.append(version)
            if version.effective_from and version.effective_from <= as_of:
                break
    return assign_effective_ranges(versions)


def blob_url(version: LawVersion) -> str:
    return f"https://github.com/{REPOSITORY}/blob/{version.commit}/{quote(version.path)}"
