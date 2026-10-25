"""The rule sweep adds lines the model skipped. A line is a duplicate only when it reads like a claim already listed,
not merely because it shares the matched word with another sentence."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services/rag-mcp/src"))
from capstone_rag_mcp.rules import scan_lines  # noqa: E402
from capstone_agent.regulations import sweep_findings  # noqa: E402


async def scan(_service, _tool, arguments):
    return {"matches": scan_lines(arguments["lines"])}


def block(i, text):
    return {"id": f"t{i}", "text": text, "location": {"page": 1}}


def test_brightening_names_the_whitening_function():
    assert scan_lines(["더 퍼스트 제너추어 브라이트닝 듀얼 세럼"])[0]["rules"] == ["functional"]


def test_another_sentence_with_the_same_word_is_swept_but_a_noisy_duplicate_is_not():
    known = [{"phrase": "진정과 동시에 투명한 피부로 가꾸어주는 브라이트닝 세럼"}]
    texts = [block(0, "더 퍼스트 제너추어 브라이트닝 듀얼 세럼"), block(1, "진정과 동시에 투명한 피부로 가꾸어 주는 브라이트님 세럼 브라이트닝")]
    added = asyncio.run(sweep_findings(scan, texts, known))
    assert [a["phrase"] for a in added] == ["더 퍼스트 제너추어 브라이트닝 듀얼 세럼"]
    assert added[0]["claimType"] == "whitening"
