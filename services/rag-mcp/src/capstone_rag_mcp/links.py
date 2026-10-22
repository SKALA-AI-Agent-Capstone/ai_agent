"""Citation and delegation edges between loaded articles (knowledge.article_links).

Statute text cites other provisions all the time: 시행규칙 제22조 says "법 제13조제2항에 따른 ...
별표 5와 같다". Search finds one chunk; these edges let retrieval follow the chain to the
provision that actually lists the prohibited expressions. Edges are stored by law name and
article label so they survive re-ingestion of new versions.
"""
import re

import psycopg

# Until 2026-10-22 the 화장품법 statutes came from legalize-kr (GitHub); the Law API copy replaces them.
# Stored legacy rows stay readable until a reload ends their effective range (ingest.retire_replaced).
LEGACY_KEYS = {f"lawapi:화장품법/{kind}": f"legalize-kr:화장품법/{kind}" for kind in ("법률", "시행령", "시행규칙")}

# document_key prefix -> official law name used in citations
DOCUMENT_LAWS = {
    "legalize-kr:화장품법/법률": "화장품법",
    "legalize-kr:화장품법/시행령": "화장품법 시행령",
    "legalize-kr:화장품법/시행규칙": "화장품법 시행규칙",
    "lawapi:화장품법/법률": "화장품법",
    "lawapi:화장품법/시행령": "화장품법 시행령",
    "lawapi:화장품법/시행규칙": "화장품법 시행규칙",
    "lawapi:S06": "화장품법 시행규칙",
    "lawapi:S07": "화장품 표시\u00b7광고 실증에 관한 규정",  # 공식 명칭의 가운뎃점
    "lawapi:S08": "기능성화장품 심사에 관한 규정",
    "lawapi:S09": "표시ㆍ광고의 공정화에 관한 법률",
    "lawapi:S10": "화장품 안전기준 등에 관한 규정",
}
PARENT_LAW = {"화장품법 시행령": "화장품법", "화장품법 시행규칙": "화장품법"}

CITED_LAW = re.compile(r"「([^」]{2,40})」\s*(?:제(\d+)조(?:의(\d+))?)?")
SAME_ACT_RULE = re.compile(r"같은\s*법\s*(시행령|시행규칙)\s*제(\d+)조(?:의(\d+))?")
PARENT_ARTICLE = re.compile(r"(?<![가-힣])법\s*제(\d+)조(?:의(\d+))?")
ANNEX = re.compile(r"별표\s*(\d+)")
PLAIN_ARTICLE = re.compile(r"제(\d+)조(?:의(\d+))?")
HUB_FAN_OUT = 8
NORMALIZE = str.maketrans({"\u00b7": "", "\u318d": "", "\u30fb": "", " ": ""})

DDL = """
CREATE TABLE IF NOT EXISTS knowledge.article_links (
    source_law text NOT NULL,
    source_section text NOT NULL,
    target_law text NOT NULL,
    target_section text NOT NULL,
    relation text NOT NULL CHECK (relation IN ('cites', 'implements', 'annex')),
    PRIMARY KEY (source_law, source_section, target_law, target_section, relation)
)
"""


def law_of(document_key: str) -> str | None:
    return next((law for prefix, law in DOCUMENT_LAWS.items() if document_key.startswith(prefix)), None)


def article_label(section: str | None) -> str | None:
    match = re.match(r"^(제\d+조(?:의\d+)?|별표(?: \d+)?)", section or "")
    return match.group(1) if match else None


def canonical_law(name: str) -> str | None:
    key = name.translate(NORMALIZE)
    return next((law for law in set(DOCUMENT_LAWS.values()) if law.translate(NORMALIZE) == key), None)


def extract(law: str, section: str, body: str) -> set[tuple[str, str, str]]:
    """Return (target_law, target_label, relation) edges found in one article body."""
    edges: set[tuple[str, str, str]] = set()
    taken: list[tuple[int, int]] = []
    last_cited = None
    for pattern in (CITED_LAW, SAME_ACT_RULE, PARENT_ARTICLE):
        taken += [m.span() for m in pattern.finditer(body)]
    for match in PLAIN_ARTICLE.finditer(body):
        if not any(a <= match.start() < b for a, b in taken):
            edges.add((law, f"제{match.group(1)}조" + (f"의{match.group(2)}" if match.group(2) else ""), "cites"))
    for match in CITED_LAW.finditer(body):
        target = canonical_law(match.group(1))
        if target:
            last_cited = target
            if match.group(2):
                label = f"제{match.group(2)}조" + (f"의{match.group(3)}" if match.group(3) else "")
                edges.add((target, label, "cites"))
    for match in SAME_ACT_RULE.finditer(body):
        base = last_cited or PARENT_LAW.get(law) or law
        target = f"{PARENT_LAW.get(base, base)} {match.group(1)}"
        if target in DOCUMENT_LAWS.values():
            edges.add((target, f"제{match.group(2)}조" + (f"의{match.group(3)}" if match.group(3) else ""), "cites"))
    if law in PARENT_LAW:
        for match in PARENT_ARTICLE.finditer(body):
            label = f"제{match.group(1)}조" + (f"의{match.group(2)}" if match.group(2) else "")
            edges.add((PARENT_LAW[law], label, "implements"))
    if not section.startswith("별표"):
        for match in ANNEX.finditer(body):
            edges.add((law, f"별표 {match.group(1)}", "annex"))
    edges.discard((law, section, "cites"))
    return edges


def rebuild_links(conn: psycopg.Connection) -> int:
    with conn.cursor() as cur:
        cur.execute(DDL)
        cur.execute("TRUNCATE knowledge.article_links")
        cur.execute("""SELECT DISTINCT d.document_key, c.section, c.body FROM knowledge.chunks c
                       JOIN knowledge.document_versions d USING (document_version_id)
                       WHERE d.collection_key = 'rules' AND c.section IS NOT NULL AND c.parent_chunk_id IS NULL""")
        rows = cur.fetchall()
        edges = set()
        for document_key, section, body in rows:
            law, label = law_of(document_key), article_label(section)
            if not law or not label:
                continue
            for target_law, target_label, relation in extract(law, label, body):
                if (target_law, target_label) != (law, label):
                    edges.add((law, label, target_law, target_label, relation))
        # Articles that cite many others (권한의 위임, 벌칙, 과태료) are hubs; following them adds noise, not basis.
        fan_out: dict[tuple[str, str], int] = {}
        for edge in edges:
            fan_out[edge[:2]] = fan_out.get(edge[:2], 0) + 1
        edges = {e for e in edges if fan_out[e[:2]] <= HUB_FAN_OUT}
        cur.executemany("""INSERT INTO knowledge.article_links VALUES (%s,%s,%s,%s,%s)
                           ON CONFLICT DO NOTHING""", sorted(edges))
    conn.commit()
    return len(edges)
