"""Parent-child chunks for statute articles, linked by knowledge.chunks.parent_chunk_id.

A long article is stored as a parent row (the whole article, parent_chunk_id NULL) and one child row
per 항 or 호 pointing at it, e.g. section "제14조 (표시ㆍ광고 내용의 실증 등) 제3항". Search ranks only
leaf rows (children, and articles that were not split), so a claim that matches one 호 is not diluted
by the rest of the article, then answers with the parent article and the matched child's label.
Everything that reads an article as a unit (rule lookup, article links, document view, inventory)
reads top-level rows only, so it sees each article exactly once, as before the split.
"""
import re

MIN_SPLIT = 300  # shorter articles stay one chunk (화장품법 제13조 is 347 characters and must split)
MIN_LEAD = 60    # a lead shorter than this (usually just the title) joins the first child

CIRCLED = {chr(0x2460 + i): i + 1 for i in range(20)}  # ① .. ⑳
PARAGRAPH = re.compile(r"^\s*([①-⑳])")
ITEM = re.compile(r"^\s{0,3}(\d{1,2})\.\s")


def split_article(body: str) -> list[tuple[str, str]]:
    """[(child label, child body)] of one article; [] when the article is not worth splitting.

    The children's bodies joined with newlines are the article exactly.
    """
    if len(body) < MIN_SPLIT:
        return []
    parts: list[tuple[str, list[str]]] = [("", [])]
    paragraph = None
    for line in body.split("\n"):
        if match := PARAGRAPH.match(line):
            paragraph = CIRCLED[match.group(1)]
            parts.append((f"제{paragraph}항", [line]))
        elif match := ITEM.match(line):
            prefix = f"제{paragraph}항" if paragraph else ""
            parts.append((f"{prefix}제{match.group(1)}호", [line]))
        else:
            parts[-1][1].append(line)
    lead = parts[0][1]
    parts = parts[1:]
    if not parts:
        return []
    if len("\n".join(lead).strip()) < MIN_LEAD:
        parts[0] = (parts[0][0], lead + parts[0][1])
    else:
        parts.insert(0, ("본문", lead))
    if len(parts) < 2:
        return []
    return [(label, "\n".join(lines)) for label, lines in parts]


def child_section(parent_section: str, label: str) -> str:
    return f"{parent_section} {label}"


def collapse(ranked: list, ranked_type) -> list:
    """Replace ranked children with their parent article, best child first, one entry per article.

    Candidate rows carry parent_chunk_id, parent_section and parent_body (NULL for unsplit articles).
    The parent keeps the best child's scores, and the child's label in "matched_child".
    """
    out, seen = [], set()
    for item in ranked:
        row = item.row
        parent_id = row.get("parent_chunk_id")
        key = parent_id or row["chunk_id"]
        if key in seen:
            continue
        seen.add(key)
        if not parent_id:
            out.append(item)
            continue
        label = (row["section"] or "")[len(row["parent_section"] or ""):].strip() or None
        parent = {**row, "chunk_id": parent_id, "section": row["parent_section"], "body": row["parent_body"],
                  "matched_child": label}
        out.append(ranked_type(parent, item.similarity, keyword=item.keyword, vector=item.vector))
    return out
