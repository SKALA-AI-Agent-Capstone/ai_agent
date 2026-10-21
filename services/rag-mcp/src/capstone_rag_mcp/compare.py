"""Keyword-only versus hybrid retrieval on the same ad claims (Harness evaluation: before and after).

  docker compose run --rm -T rag-ingest uv run --no-sync python -m capstone_rag_mcp.compare

Each claim is searched with the query the Agent sends (phrase plus statute wording for its claim type).
Two measures:
- judged relevance: a model reads each of the top 3 hits and says whether it is directly useful for reviewing
  the claim (LLM as judge). This is the main measure, because search exists to add guideline and association
  text that the rule table does not name.
- rule agreement: whether a top 3 hit is a provision the claim rule table maps to the claim. Rules already attach
  those provisions, so this only shows that search does not drift away from them.
The control phrases are not advertising claims; their best vector score shows how high an unrelated match goes.
"""
import asyncio
from datetime import date
import json
import os

from capstone_agent.regulations import build_queries, relevant, SEARCH_MIN_SIMILARITY, SEARCH_MIN_VECTOR

from .claims import resolve_claim_rules
from .embedding import load_embedder
from .retrieval import search

CLAIMS = [
    ("세월을 거스르는 피부 탄력", "exaggeration"),
    ("10년 전 피부로 돌아가는 기적", "exaggeration"),
    ("탄력 광채 안티에이징", "exaggeration"),
    ("아토피 피부도 진정시키는 크림", "drug"),
    ("피부 세포를 재생시키는 줄기세포 배양액", "drug"),
    ("피부과 전문의가 추천하는 크림", "exaggeration"),
    ("식약처 심사를 완료하였습니다", "exaggeration"),
    ("2주 만에 모공 크기 감소", "clinical"),
    ("처지고 늘어진 모공 2주 안에 개선", "clinical"),
    ("탄성 복원 63.91% 개선", "clinical"),
    ("인체적용시험으로 입증된 주름 개선 효과", "wrinkle"),
    ("미백 기능성 인증 완료", "whitening"),
    ("업계 최초, 유일한 특허 성분", "absolute"),
    ("무첨가 저자극 천연 유래 성분", "ingredient"),
]
CONTROLS = ["배송은 결제 후 2~3일 이내 출고됩니다", "고객센터 운영 시간은 평일 오전 10시부터입니다", "다 쓴 용기는 분리배출해 주세요"]
MODES = [("키워드만", None), ("하이브리드 RRF", "rrf"), ("하이브리드 계단식", "cascade"), ("하이브리드 키워드순+벡터확인", "confirm")]


def article(section: str | None) -> str:
    """'제13조 (부당한…) 제1항제5호' -> '제13조', '별표 5 제2호 사목' -> '별표 5'."""
    words = (section or "").replace("[", "").replace("]", "").split()
    return " ".join(words[:2]) if words[:1] == ["별표"] else (words[0] if words else "")


JUDGE_PROMPT = ("화장품 광고 문구를 심사하는 담당자 입장에서, 아래 각 문서 조각이 이 광고 주장의 적법성을 검토하는 데 "
                "직접 근거가 되는지 판단한다. 같은 표현 유형을 다루거나 그 주장에 필요한 실증, 표시 기준을 정하면 관련 있음이다. "
                "주제가 다르거나 일반 절차만 다루면 관련 없음이다. JSON으로 {\"relevant\": [true/false, ...]}만 답한다.")


def judge(client, claim: str, ranked) -> list[bool]:
    if not ranked:
        return []
    chunks = "\n\n".join(f"[{i}] {r.row['title']} {r.row['section'] or ''}\n{r.row['body'][:1200]}" for i, r in enumerate(ranked))
    response = client.chat.completions.create(
        model=os.getenv("TEXT_MODEL", "gpt-5.4-mini"), response_format={"type": "json_object"},
        messages=[{"role": "system", "content": JUDGE_PROMPT},
                  {"role": "user", "content": f"광고 주장: {claim}\n\n문서 조각 {len(ranked)}개:\n{chunks}"}])
    verdicts = json.loads(response.choices[0].message.content).get("relevant", [])
    return [bool(v) for v in verdicts[:len(ranked)]] + [False] * (len(ranked) - len(verdicts))


def hit(ranked, reference) -> bool:
    return any((r.row["document_key"], article(r.row["section"])) in reference for r in ranked[:3])


async def compare() -> list[str]:
    dsn = os.environ.get("RAG_DATABASE_URL") or os.environ["RAG_INGEST_DATABASE_URL"]
    embedder = load_embedder()
    if embedder is None:
        raise SystemExit("Set RAG_EMBEDDING_MODEL (and OPENAI_API_KEY for OpenAI models) to compare.")
    day = date.today()
    lines = [f"모델: {embedder.model_name} ({embedder.revision})", "",
             "| 문구 | 규칙 조항 | " + " | ".join(name for name, _ in MODES) + " |",
             "| --- | --- | " + " | ".join("---" for _ in MODES) + " |"]
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    relevance = {name: [0, 0] for name, _ in MODES}  # [관련 있음, 판정한 조각 수]
    chosen_ok = {name: 0 for name, _ in MODES}
    totals = {name: 0 for name, _ in MODES}
    attached = {name: [0, 0] for name, _ in MODES}  # [근거를 붙인 주장 수, 그중 규칙 조항과 같은 수]
    scored = 0
    for phrase, claim_type in CLAIMS:
        rules = await resolve_claim_rules(dsn, phrase, claim_type, day)
        reference = {(p.document_key, article(p.section)) for m in rules.matches for p in m.provisions}
        query = build_queries({"phrase": phrase, "claimType": claim_type})[0]
        cells = []
        for name, weight in MODES:
            _, ranked, _ = await search(dsn, collection="rules", query=query, top_k=3, review_date=day,
                                        embedder=embedder if weight else None, order=weight or "rrf")
            ok = hit(ranked, reference) if reference else None
            totals[name] += bool(ok)
            # The Agent attaches the first hit that clears its thresholds (regulations.select_hits).
            chosen = next((r for r in ranked if relevant({"keyword_score": r.keyword, "vector_score": r.vector,
                                                          "similarity": r.similarity},
                                                         SEARCH_MIN_SIMILARITY, SEARCH_MIN_VECTOR)), None)
            verdicts = await asyncio.to_thread(judge, client, phrase, ranked)
            relevance[name][0] += sum(verdicts)
            relevance[name][1] += len(verdicts)
            if chosen:
                attached[name][0] += 1
                attached[name][1] += (chosen.row["document_key"], article(chosen.row["section"])) in reference
                chosen_ok[name] += bool(verdicts[ranked.index(chosen)]) if chosen in ranked else 0
            top = ranked[0].row if ranked else None
            label = f"{top['title'][:14]} {article(top['section'])}" if top else "-"
            cells.append(f"관련 {sum(verdicts)}/{len(verdicts)}, 규칙 {'O' if ok else 'X'}, 1위 {label}")
        scored += bool(reference)
        refs = ", ".join(sorted({a for _, a in reference})) or "(규칙 없음)"
        lines.append(f"| {phrase} | {refs} | " + " | ".join(cells) + " |")
    lines.append(f"| **상위 3개 안에 규칙 조항** | {scored}건 기준 | "
                 + " | ".join(f"**{totals[name]}/{scored}**" for name, _ in MODES) + " |")
    lines.append("| **상위 3개 관련도** (모델 판정) | | "
                 + " | ".join(f"**{relevance[name][0]}/{relevance[name][1]}**" for name, _ in MODES) + " |")
    lines.append("| **Agent가 붙이는 검색 근거** (붙인 수, 그중 관련 있음) | | "
                 + " | ".join(f"{attached[name][0]}건, 관련 {chosen_ok[name]}" for name, _ in MODES) + " |")
    lines += ["", "| 대조군 | 벡터 최고 코사인 |", "| --- | --- |"]
    for phrase in CONTROLS:
        _, ranked, _ = await search(dsn, collection="rules", query=phrase, top_k=5, review_date=day, embedder=embedder)
        best = max((r.vector or 0 for r in ranked), default=0)
        lines.append(f"| {phrase} | {best:.2f} |")
    claim_best = []
    for phrase, claim_type in CLAIMS:
        _, ranked, _ = await search(dsn, collection="rules", query=build_queries({"phrase": phrase, "claimType": claim_type})[0],
                                    top_k=5, review_date=day, embedder=embedder)
        claim_best.append(max((r.vector or 0 for r in ranked), default=0))
    lines.append(f"| (참고) 광고 주장의 벡터 최고 코사인 범위 | {min(claim_best):.2f}~{max(claim_best):.2f} |")
    return lines


def main() -> int:
    print("\n".join(asyncio.run(compare())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
