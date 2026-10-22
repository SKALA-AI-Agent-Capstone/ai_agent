"""화장품법, 시행령, 시행규칙 come from the 법제처 API only (legalize-kr / GitHub removed 2026-10-22)."""
from datetime import date
import importlib.util
import json
from pathlib import Path

import pytest

from capstone_rag_mcp import batch
from capstone_rag_mcp.claims import pick
from capstone_rag_mcp.ingest import Document, retire_replaced
from capstone_rag_mcp.lawapi import LawApiDocument, LawChunk
from capstone_rag_mcp.links import DOCUMENT_LAWS, LEGACY_KEYS

RULES = Path(batch.__file__).with_name("claim_rules.json")


def law_document(name: str, effective: date) -> LawApiDocument:
    return LawApiDocument(target="eflaw", law_id=name, title=name, kind="법률" if name == "화장품법" else "대통령령",
                          version_label="2026-03-01 제1호", issuer="식품의약품안전처", published_on=effective,
                          effective_from=effective, raw=name.encode(),
                          chunks=[LawChunk(section="제13조 (부당한 표시ㆍ광고 행위 등의 금지)", context=name, body="본문")])


class FakeLawApi:
    calls: list = []

    def __init__(self, oc, client):
        pass

    async def law(self, name, *, annex_numbers=None, articles=True):
        FakeLawApi.calls.append(("law", name, articles, annex_numbers))
        return law_document(name, date(2026, 3, 1))

    async def admrul(self, name, *, annex_numbers=None):
        FakeLawApi.calls.append(("admrul", name, True, annex_numbers))
        return law_document(name, date(2026, 3, 1))


async def test_statutes_are_loaded_from_the_law_api_with_their_own_keys(monkeypatch):
    FakeLawApi.calls = []
    monkeypatch.setenv("LAW_OC", "test-oc")
    monkeypatch.setattr(batch, "LawApi", FakeLawApi)
    documents, skipped = await batch.collect("lawapi")
    keys = [d.document_key for d in documents]
    assert keys[:3] == ["lawapi:화장품법/법률", "lawapi:화장품법/시행령", "lawapi:화장품법/시행규칙"]
    assert {"lawapi:S06", "lawapi:S07", "lawapi:S08", "lawapi:S09", "lawapi:S10"} <= set(keys)
    assert skipped == []
    statute = documents[0]
    assert statute.source_type == "law" and statute.collection == "rules"
    assert statute.source_url == "https://www.law.go.kr/법령/화장품법"
    assert documents[1].source_type == "regulation"
    # S06 still asks only for annex 5; the 시행규칙 articles come with the S05 statute.
    assert ("law", "화장품법 시행규칙", False, {"5"}) in FakeLawApi.calls
    assert ("law", "화장품법 시행규칙", True, set()) in FakeLawApi.calls


async def test_without_law_oc_every_law_api_source_is_reported_not_configured(monkeypatch):
    monkeypatch.delenv("LAW_OC", raising=False)
    documents, skipped = await batch.collect("lawapi")
    assert documents == []
    assert {s["id"] for s in skipped} == {"S05", "S06", "S07", "S08", "S09", "S10"}
    assert {s["reason"] for s in skipped} == {"LAW_OC_NOT_CONFIGURED"}


def test_legalize_kr_route_and_github_client_are_gone():
    assert importlib.util.find_spec("capstone_rag_mcp.legalize") is None
    with pytest.raises(SystemExit):
        batch.main(["legalize", "--dry-run"])
    assert "legalize-kr:" not in RULES.read_text(encoding="utf-8")


def test_every_statute_key_in_the_rule_table_is_a_known_law_document():
    keys = set()

    def walk(node):
        if isinstance(node, dict):
            if isinstance(node.get("document_key"), str):
                keys.add(node["document_key"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(json.loads(RULES.read_text(encoding="utf-8")))
    assert {"lawapi:화장품법/법률", "lawapi:화장품법/시행규칙"} <= keys
    assert all(key in DOCUMENT_LAWS or key.startswith("pdf:") for key in keys)


def test_rule_keys_still_find_a_statute_loaded_before_the_switch():
    legacy = {"document_key": "legalize-kr:화장품법/법률"}
    current = {"document_key": "lawapi:화장품법/법률"}
    assert pick({"legalize-kr:화장품법/법률": legacy}, "lawapi:화장품법/법률") is legacy
    assert pick({"legalize-kr:화장품법/법률": legacy, "lawapi:화장품법/법률": current}, "lawapi:화장품법/법률") is current
    assert pick({}, "lawapi:화장품법/법률") is None


class FakeConnection:
    def __init__(self, rowcount=4):
        self.executed, self.commits, self.rowcount = [], 0, rowcount

    def cursor(self):
        connection = self

        class Cursor:
            rowcount = connection.rowcount

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def execute(self, sql, params):
                connection.executed.append((" ".join(sql.split()), params))

        return Cursor()

    def commit(self):
        self.commits += 1


def statute(key: str, effective: date | None) -> Document:
    return Document(document_key=key, version_label="v", title="화장품법", issuer="식약처", collection="rules",
                    source_type="law", source_url="https://www.law.go.kr", content_sha256="0", storage_key="lawapi:x",
                    effective_from=effective)


def test_reload_ends_legacy_versions_the_day_before_the_law_api_version():
    connection = FakeConnection(rowcount=4)
    assert retire_replaced(connection, statute("lawapi:화장품법/법률", date(2026, 3, 1))) == 4
    sql, params = connection.executed[0]
    assert sql.startswith("UPDATE knowledge.document_versions SET effective_until")
    assert params == (date(2026, 2, 28), "legalize-kr:화장품법/법률", date(2026, 2, 28))
    assert connection.commits == 1


def test_documents_without_a_legacy_copy_are_left_alone():
    connection = FakeConnection()
    assert retire_replaced(connection, statute("lawapi:S07", date(2026, 3, 1))) == 0
    assert connection.executed == [] and connection.commits == 0
    assert set(LEGACY_KEYS) == {"lawapi:화장품법/법률", "lawapi:화장품법/시행령", "lawapi:화장품법/시행규칙"}


class FakeAsyncConnection:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def execute(self, *args, **kwargs):
        return self


def legacy_statute(key: str, section: str, body: str) -> dict:
    return {"document_version_id": key, "document_key": key, "title": "화장품법", "version_label": "2026-10-20 제22113호",
            "source_url": "https://github.com/legalize-kr/legalize-kr", "storage_key": f"legalize-kr@abc:{key}",
            "source_type": "law", "effective_from": date(2026, 10, 20), "effective_until": None, "corpus_version": "1",
            "chunks": [{"chunk_id": f"{key}#1", "section": section, "context": "화장품법", "body": body, "page_from": None}]}


@pytest.mark.parametrize("phrase, claim_type, legacy_key, section", [
    ("필러 침투 재생", "drug", "legalize-kr:화장품법/법률", "제13조 (부당한 표시ㆍ광고 행위 등의 금지)"),
    ("볼륨 개선도 92.7%", "clinical", "legalize-kr:화장품법/시행규칙", "제23조 (표시ㆍ광고 실증의 대상 등)"),
])
async def test_rule_lookup_runs_against_a_corpus_loaded_before_the_switch(monkeypatch, phrase, claim_type, legacy_key, section):
    """Regression: a leftover name made every rule lookup fail once legalize-kr was removed."""
    from capstone_rag_mcp import claims

    loaded = {legacy_key: legacy_statute(legacy_key, section, "1. 의약품으로 잘못 인식할 우려가 있는 표시 또는 광고")}
    requested = []

    async def load_documents(conn, keys, review_date):
        requested.extend(keys)
        return loaded

    async def connect(dsn):
        return FakeAsyncConnection()

    async def related_rows(*args, **kwargs):
        return []

    monkeypatch.setattr(claims, "connect", connect)
    monkeypatch.setattr(claims, "load_documents", load_documents)
    monkeypatch.setattr(claims, "related_rows", related_rows)
    result = await claims.resolve_claim_rules("postgresql://unused", phrase, claim_type, date(2026, 10, 22))
    assert legacy_key in requested
    hits = [p for match in result.matches for p in match.provisions]
    assert any(p.document_key == legacy_key and p.section == section for p in hits)
