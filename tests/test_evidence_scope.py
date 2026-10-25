"""A report that tests several products must not support one product's claim with another product's result."""
import asyncio

from capstone_agent.evidence import EvidenceSummary, summarize

TEXT = ("#1: 디퍼앤디퍼 스카워터링 크라이오 크림 마스크\n#2: 디퍼앤디퍼 핑크 리프팅 업 토닝 크림 마스크\n"
        "#3: 디퍼앤디퍼 세라루로닉 워터 코어 크림\n#4: 디퍼앤디퍼 엑토자일렌 퍼밍 밴드 크림\n"
        "1. 즉각적인 속 수분량 평가 스카워터링 10.10% 2. 피부 투명도(맑기) 평가 핑크 9.71% 4. 탄력 평가 퍼밍 10.53%")
FULL = EvidenceSummary(kind="efficacy_test", title="디퍼앤디퍼 외 3", purpose="수분량, 투명도, 17겹 수분량, 탄력 평가",
                       measures=["속 수분량", "피부 투명도", "17겹 수분량", "탄력"], period="2주", subjects="22명",
                       institution="KSRC", result="수분량 10.10%, 투명도 9.71%, 탄력 10.53%")


class Provider:
    def __init__(self, summary):
        self.summary = summary

    async def parse(self, *_args, **_kwargs):
        return self.summary


def doc(text=TEXT, results=None):
    return {"id": "d1", "name": "d.pdf", "docType": "clinical_test", "text": text, "results": results or []}


def run(document, summary=FULL):
    return asyncio.run(summarize(Provider(summary), [document]))[0]


def test_only_the_registered_products_rows_stay_in_a_multi_product_report():
    rows = [{"item": "즉각적인 피부 투명도(맑기) 평가", "changePct": 9.71, "significant": True, "page": 6},
            {"item": "즉각적인 피부 투명도(맑기) 평가", "changePct": 9.71, "significant": True, "page": 17}]
    summary = run(doc(results=rows))
    assert summary["measures"] == ["즉각적인 피부 투명도(맑기) 평가"]
    assert "9.71% 변화율 유의" in summary["result"] and "수분량" not in summary["purpose"] + summary["result"]
    assert "탄력" not in summary["purpose"] + summary["result"]


def test_a_registered_product_without_rows_has_nothing_measured():
    summary = run(doc(results=[]))
    assert summary["measures"] == [] and "찾지 못했습니다" in summary["result"]


def test_a_single_product_report_is_summarised_as_read():
    text = "인체적용시험 결과보고서 '투집 포어본드'의 피부 일차자극 평가"
    summary = run(doc(text=text), FULL.model_copy(update={"measures": ["피부 일차자극"], "result": "무자극"}))
    assert summary["measures"] == ["피부 일차자극"] and summary["result"] == "무자극"
