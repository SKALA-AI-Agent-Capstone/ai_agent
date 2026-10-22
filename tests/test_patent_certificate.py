from capstone_agent.patent_certificate import group_patent_certificate, patent_numbers
from capstone_agent.regulations import sweep_findings


def block(text, index, *, y=None, tile="top", method="VLM_OCR_REGION"):
    location = {"fileId": "ad", "page": 1, "tileId": tile, "coordinateSource": method,
                "region": [0, 0, 1, 0.6], "bbox": [0.1, y, 0.9, y + 0.02] if y is not None else None}
    return {"id": f"line-{index}", "text": text, "location": location}


def test_certificate_fields_group_without_swallowing_promotional_copy():
    lines = [block("특 허 증", 0), block("특허 제 0440357 호 “OCR noise", 1, y=0.15, method="TESSERACT"),
             block("출 원 번 호 제 2001-0065893 호", 2), block("출 원 일 2001년 10월 25일", 3),
             block("공 개 일 2004년 07월 05일", 4),
             block("2004 년 07 월 05%", 5, y=0.18, method="TESSERACT"),
             block("탄력섬유세포 활성화→콜라겐볼륨!", 6)]
    grouped = group_patent_certificate(lines)
    assert len(grouped) == 2
    assert grouped[0]["documentContext"] == "patent_certificate"
    assert "0440357" in grouped[0]["text"] and "출 원 일" in grouped[0]["text"]
    assert "05%" not in grouped[0]["text"]
    assert grouped[1]["text"] == lines[-1]["text"]
    assert patent_numbers(grouped[0]["text"]) == ["10-0440357"]


def test_without_certificate_header_keeps_separate_lines():
    lines = [block("특허 제 0440357 호", 0), block("볼륨 개선도 92.7%", 1)]
    assert group_patent_certificate(lines) == lines


async def test_certificate_metadata_is_not_sent_to_rule_sweep():
    seen = []
    async def call(server, tool, arguments):
        seen.extend(arguments["lines"])
        return {"matches": []}
    await sweep_findings(call, [block("2004년 07월 05일", 0) | {"documentContext": "patent_certificate"},
                                block("볼륨 개선도 92.7%", 1)], [])
    assert seen == ["볼륨 개선도 92.7%"]
