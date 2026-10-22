"""Long statute articles are searched by 항/호 children and answered with the whole article (parent)."""
from datetime import date

from capstone_rag_mcp.lawapi import LawApiDocument, LawChunk, admrul_articles, law_articles
from capstone_rag_mcp.ingest import from_lawapi_statute
from capstone_rag_mcp.parents import child_section, collapse, split_article
from capstone_rag_mcp.retrieval import Ranked, keyword_rank

ARTICLE_14 = "\n".join([
    "제14조(표시ㆍ광고 내용의 실증 등)",
    "① 영업자 및 판매자는 자기가 행한 표시ㆍ광고 중 사실과 관련한 사항에 대하여는 이를 실증할 수 있어야 한다.",
    "② 식품의약품안전처장은 표시ㆍ광고가 제13조제1항제4호에 해당하는지를 판단하기 위하여 실증이 필요하다고 "
    "인정하는 경우에는 그 내용을 구체적으로 명시하여 관련 자료의 제출을 요청할 수 있다.",
    "③ 제2항에 따라 실증자료의 제출을 요청받은 영업자 또는 판매자는 요청받은 날부터 15일 이내에 그 실증자료를 "
    "식품의약품안전처장에게 제출하여야 한다. 다만, 정당한 사유가 있다고 인정하는 경우에는 그 제출기간을 연장할 수 있다.",
    "④ 식품의약품안전처장은 제출기간 내에 이를 제출하지 아니한 채 계속하여 표시ㆍ광고를 하는 때에는 "
    "실증자료를 제출할 때까지 그 표시ㆍ광고 행위의 중지를 명하여야 한다.",
])
SECTION_14 = "제14조 (표시ㆍ광고 내용의 실증 등)"

ARTICLE_13 = "\n".join([
    "제13조(부당한 표시ㆍ광고 행위 등의 금지)",
    "① 영업자 또는 판매자는 다음 각 호의 어느 하나에 해당하는 표시 또는 광고를 하여서는 아니 된다.",
    "  1. 의약품으로 잘못 인식할 우려가 있는 표시 또는 광고",
    "  2. 기능성화장품이 아닌 화장품을 기능성화장품으로 잘못 인식할 우려가 있거나 기능성화장품의 안전성ㆍ유효성에 "
    "관한 심사결과와 다른 내용의 표시 또는 광고",
    "  3. 천연화장품 또는 유기농화장품이 아닌 화장품을 천연화장품 또는 유기농화장품으로 잘못 인식할 우려가 있는 표시 또는 광고",
    "  4. 그 밖에 사실과 다르게 소비자를 속이거나 소비자가 잘못 인식하도록 할 우려가 있는 표시 또는 광고",
    "② 제1항에 따른 표시ㆍ광고의 구체적인 내용 및 범위는 총리령으로 정한다.",
])
SECTION_13 = "제13조 (부당한 표시ㆍ광고 행위 등의 금지)"


def test_long_article_splits_into_paragraph_children_that_rebuild_it_exactly():
    children = split_article(ARTICLE_14)
    assert [label for label, _ in children] == ["제1항", "제2항", "제3항", "제4항"]
    # The title line joins the first child; joined children are the article as written.
    assert children[0][1].startswith("제14조(표시ㆍ광고 내용의 실증 등)\n①")
    assert "\n".join(body for _, body in children) == ARTICLE_14


def test_items_are_children_under_their_paragraph():
    labels = [label for label, _ in split_article(ARTICLE_13)]
    assert labels == ["제1항", "제1항제1호", "제1항제2호", "제1항제3호", "제1항제4호", "제2항"]


def test_short_or_flat_articles_have_no_children():
    assert split_article("제1조(목적) 이 법은 화장품의 제조ㆍ수입ㆍ판매에 관한 사항을 규정한다.") == []
    assert split_article("제5조(규정) " + "이 고시는 화장품의 표시ㆍ광고 실증에 필요한 사항을 정한다. " * 12) == []


def leaf_rows():
    """Candidate rows as CANDIDATES_SQL returns them: children with their parent, plus one unsplit article."""
    rows = []
    for parent_id, section, body in (("p14", SECTION_14, ARTICLE_14), ("p13", SECTION_13, ARTICLE_13)):
        for index, (label, part) in enumerate(split_article(body)):
            rows.append({"chunk_id": f"{parent_id}-{index}", "parent_chunk_id": parent_id, "parent_section": section,
                         "parent_body": body, "section": child_section(section, label), "body": part,
                         "document_key": "lawapi:화장품법/법률"})
    rows.append({"chunk_id": "p1", "parent_chunk_id": None, "parent_section": None, "parent_body": None,
                 "section": "제1조 (목적)", "body": "제1조(목적) 이 법은 화장품에 관한 사항을 규정한다.",
                 "document_key": "lawapi:화장품법/법률"})
    return rows


def test_search_ranks_children_and_answers_with_one_whole_article():
    ranked = collapse(keyword_rank("실증자료 제출 요청 15일 이내 제출기간 연장", leaf_rows()), Ranked)
    top = ranked[0]
    assert (top.row["chunk_id"], top.row["section"], top.row["matched_child"]) == ("p14", SECTION_14, "제3항")
    assert top.row["body"] == ARTICLE_14
    assert [r.row["chunk_id"] for r in ranked].count("p14") == 1


def test_matched_item_names_the_exact_ho():
    top = collapse(keyword_rank("천연화장품 유기농화장품으로 잘못 인식", leaf_rows()), Ranked)[0]
    assert (top.row["section"], top.row["matched_child"]) == (SECTION_13, "제1항제3호")


def test_unsplit_rows_pass_through_unchanged():
    row = leaf_rows()[-1]
    item = Ranked(row, 0.5, keyword=0.5)
    assert collapse([item], Ranked) == [item]


class Unit:
    """Minimal stand-in for an XML node: findtext and iter as law_articles uses them."""

    def __init__(self, tag, text="", children=(), **fields):
        self.tag, self.text, self.children, self.fields = tag, text, list(children), fields

    def findtext(self, tag):
        return self.fields.get(tag)

    def iter(self, tag=None):
        nodes = [self]
        for child in self.children:
            nodes += child.iter()
        return [n for n in nodes if tag is None or n.tag == tag]


def article_13_unit():
    lines = ARTICLE_13.split("\n")
    return Unit("조문단위", children=[Unit("항내용", lines[1]), *[Unit("호내용", line) for line in lines[2:6]],
                                      Unit("항내용", lines[6])],
                **{"조문여부": "조문", "조문번호": "13", "조문가지번호": "", "조문제목": "부당한 표시ㆍ광고 행위 등의 금지",
                   "조문내용": lines[0]})


def test_law_api_articles_keep_children_under_the_article_and_version_by_layout():
    chunks = law_articles(Unit("법령", children=[article_13_unit()]), "화장품법")
    assert [c.section for c in chunks] == [SECTION_13]
    # The parent is the article as the Law API gives it; its children rebuild it exactly.
    assert chunks[0].body == "\n".join(child.body for child in chunks[0].children)
    assert [c.section for c in chunks[0].children][1] == SECTION_13 + " 제1항제1호"
    doc = LawApiDocument(target="eflaw", law_id="1", title="화장품법", kind="법률", version_label="v", issuer="식약처",
                         published_on=date(2026, 1, 1), effective_from=date(2026, 1, 1), raw=b"<xml/>", chunks=chunks)
    split = from_lawapi_statute("lawapi:화장품법/법률", doc, "https://www.law.go.kr")
    assert len(split.chunks) == 1 and len(split.chunks[0]["children"]) == 6
    whole = LawApiDocument(**{**doc.__dict__, "chunks": [LawChunk(SECTION_13, "화장품법", ARTICLE_13)]})
    before = from_lawapi_statute("lawapi:화장품법/법률", whole, "https://www.law.go.kr")
    assert split.content_sha256 != before.content_sha256  # a reload stores the split as a new version


def test_admrul_articles_split_the_same_way():
    node = Unit("조문내용", ARTICLE_14.replace("제14조(표시ㆍ광고 내용의 실증 등)", "제3조(실증자료)"))
    chunks = admrul_articles(Unit("행정규칙", children=[node]), "실증 규정")
    assert chunks[0].section == "제3조 (실증자료)"
    assert [c.section for c in chunks[0].children][:2] == ["제3조 (실증자료) 제1항", "제3조 (실증자료) 제2항"]


class InsertCursor:
    """Records chunk inserts and hands out ids, so write() can be checked without a database."""

    def __init__(self):
        self.rows, self.next_id = [], 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, sql, params=None):
        self.last = " ".join(sql.split())
        if self.last.startswith("INSERT INTO knowledge.chunks"):
            self.rows.append(params)

    def fetchone(self):
        if self.last.startswith("SELECT document_version_id, effective_from"):
            return None
        self.next_id += 1
        return (f"id{self.next_id}",)

    def fetchall(self):
        return []


class InsertConnection:
    def __init__(self):
        self.cursor_ = InsertCursor()

    def cursor(self):
        return self.cursor_

    def commit(self):
        pass


def test_write_stores_the_article_then_children_pointing_at_it():
    from capstone_rag_mcp.ingest import write

    chunks = law_articles(Unit("법령", children=[article_13_unit()]), "화장품법")
    doc = from_lawapi_statute("lawapi:화장품법/법률", LawApiDocument(
        target="eflaw", law_id="1", title="화장품법", kind="법률", version_label="v", issuer="식약처",
        published_on=date(2026, 1, 1), effective_from=date(2026, 1, 1), raw=b"<xml/>", chunks=chunks), "https://www.law.go.kr")
    connection = InsertConnection()
    assert write(connection, doc, corpus_version="test") == "inserted"
    rows = connection.cursor_.rows
    # (version, chunk_index, parent_chunk_id, section, ...): id1 is the version, id2 the parent article.
    assert [(r[1], r[2], r[3]) for r in rows[:3]] == [
        (0, None, SECTION_13), (1, "id2", SECTION_13 + " 제1항"), (2, "id2", SECTION_13 + " 제1항제1호")]
    assert len(rows) == 7 and all(r[2] == "id2" for r in rows[1:])
    assert "\n".join(r[6] for r in rows[1:]) == rows[0][6]


class RecordingCursor:
    def __init__(self, stale):
        self.stale, self.sql = stale, []

    def execute(self, sql, params):
        self.sql.append((" ".join(sql.split()), params))

    def fetchall(self):
        return [(version,) for version in self.stale]


def statute_version(effective):
    from capstone_rag_mcp.ingest import Document

    return Document(document_key="lawapi:화장품법/법률", version_label="2026-10-20 제22113호", title="화장품법", issuer="식약처",
                    collection="rules", source_type="law", source_url="https://www.law.go.kr", content_sha256="b" * 64,
                    storage_key="lawapi:eflaw:1", effective_from=effective)


def test_rechunked_statute_replaces_the_same_version_instead_of_ending_it_before_it_starts():
    """Regression: the reload failed on effective_until >= effective_from for the old, unsplit copy."""
    from capstone_rag_mcp.ingest import replace_rechunked

    cursor = RecordingCursor(stale=["old-version"])
    assert replace_rechunked(cursor, statute_version(date(2026, 10, 20))) == 1
    select, children, chunks, versions = cursor.sql
    assert "effective_from = %s AND content_sha256 <> %s" in select[0]
    assert select[1] == ("lawapi:화장품법/법률", "2026-10-20 제22113호", date(2026, 10, 20), "b" * 64)
    # Children go first: they reference their parent row in the same table.
    assert children[0].startswith("DELETE FROM knowledge.chunks") and "parent_chunk_id IS NOT NULL" in children[0]
    assert chunks[0].startswith("DELETE FROM knowledge.chunks") and versions[0].startswith("DELETE FROM knowledge.document_versions")


def test_documents_without_an_effective_date_keep_older_versions():
    from capstone_rag_mcp.ingest import replace_rechunked

    cursor = RecordingCursor(stale=["old"])
    assert replace_rechunked(cursor, statute_version(None)) == 0 and cursor.sql == []
