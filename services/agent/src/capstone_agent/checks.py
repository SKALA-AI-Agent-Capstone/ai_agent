"""Cross-checks that the claim rules ask for. Each check compares the ad against an official record.

functional_report  claimed functional effect vs the MFDS functional cosmetic report for the product
substantiation     substantiation the rules require vs evidence files submitted with the review
ingredient         ingredient named in the ad vs 화장품 안전기준 별표 1, 2 and the MFDS regulation list
gmp                "GMP" claim vs the MFDS GMP company list, using the manufacturer from the report
patent             "특허" claim vs patent numbers found and the KIPRIS result

A check result is something the reviewer verifies, never an automatic violation.
"""
import asyncio
from datetime import datetime, timezone
import re

from pydantic import BaseModel

from .instructions import instruction
from .tools import McpTools

ATTENTION = {"mismatch", "not_found", "ambiguous", "evidence_missing", "prohibited", "restricted",
             "gmp_not_listed", "needs_input", "unavailable", "evidence_mismatch", "evidence_conditions",
             "evidence_not_covering", "evidence_unread"}
# evidence.match verdict -> check status. supports is the only one the reviewer does not need to chase.
MATCH_STATUS = {"supports": "evidence_supports", "different_test": "evidence_mismatch",
                "condition_mismatch": "evidence_conditions", "not_covered": "evidence_not_covering"}
CLAIMED_FUNCTIONS = [(r"주름|탄력 ?개선|링클", "주름개선"), (r"미백|브라이트닝|기미|주근깨|잡티|화이트닝", "미백"),
                     (r"자외선|SPF|PA\+|선 ?차단", "자외선차단"), (r"탈모", "탈모 증상 완화"),
                     (r"여드름", "여드름성 피부 완화"), (r"튼살", "튼살 완화")]
# Fallback when no model is configured: words that look like ingredient names after removing particles.
INGREDIENT_ENDINGS = re.compile(r"(추출물|오일|펩타이드|이드|산|드|놀|논|린|민|올|롤|졸|렌|틴|페놀|세라마이드)$")
PARTICLES = re.compile(r"(으로|과|와|을|를|이|가|의|은|는|로|에|도|만)$")
NOT_INGREDIENTS = {"함유", "성분", "원료", "추출", "추출물", "무첨가", "강력한", "보습감", "프리", "고농축", "특허 성분",
                   "천연 성분", "유효 성분", "핵심 성분", "보습 성분", "자연 유래 성분"}


GENERIC_INGREDIENT_WORDS = re.compile(r"성분|원료|함유|특허|무첨가")
FUNCTIONAL_CONTEXT = re.compile(r"식약처|식품의약품안전처|심사|보고|기능성")


def generic_ingredient(name: str) -> bool:
    """'특허 받은 성분' or '천연 성분' describe ingredients without naming one."""
    return name in NOT_INGREDIENTS or bool(GENERIC_INGREDIENT_WORDS.search(name))


def fallback_ingredients(phrase: str) -> list[str]:
    words = [PARTICLES.sub("", w) for w in re.findall(r"[가-힣A-Za-z0-9\-]{2,}", phrase)]
    return [w for w in dict.fromkeys(words) if len(w) >= 2 and not generic_ingredient(w) and INGREDIENT_ENDINGS.search(w)]


class IngredientMention(BaseModel):
    finding_id: str
    names: list[str]


class IngredientMentions(BaseModel):
    items: list[IngredientMention]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def check(kind: str, status: str, summary: str, *, source: str | None = None, detail: dict | None = None) -> dict:
    return {"type": kind, "status": status, "attention": status in ATTENTION, "summary": summary,
            "source": source, "detail": detail or {}, "checkedAt": now()}


def claimed_functions(phrase: str) -> list[str]:
    return [label for pattern, label in CLAIMED_FUNCTIONS if re.search(pattern, phrase)]


def functional_check(finding: dict, reports: dict | None, product_name: str) -> dict:
    claimed = claimed_functions(finding["phrase"])
    source = "식약처 기능성화장품 보고품목정보"
    if reports is None or reports.get("status") == "error":
        reason = (reports or {}).get("error") or "TOOL_UNAVAILABLE"
        return check("functional_report", "unavailable", f"보고품목 조회를 하지 못했습니다 ({reason}). 담당자가 심사결과통지서로 확인해야 합니다.", source=source)
    exact = [r for r in reports.get("exact", []) if not r.get("cancelled")]
    if exact:
        reported = sorted({f for r in exact for f in r.get("functions", [])})
        unreported = [f for f in claimed if f not in reported]
        detail = {"product": exact[0]["productName"], "reportSequence": exact[0].get("reportSequence"),
                  "reportedFunctions": reported, "claimedFunctions": claimed, "efficacy": exact[0].get("efficacyName")}
        if unreported:
            return check("functional_report", "mismatch",
                         f"광고는 {', '.join(unreported)} 효능을 말하지만 보고품목의 기능은 {', '.join(reported) or '없음'}입니다.",
                         source=source, detail=detail)
        return check("functional_report", "consistent",
                     f"보고품목 '{exact[0]['productName']}'의 기능({', '.join(reported) or '기능 표시 없음'}) 범위 안의 표현입니다.",
                     source=source, detail=detail)
    if reports.get("candidates"):
        names = [c["productName"] for c in reports["candidates"][:3]]
        return check("functional_report", "ambiguous",
                     f"'{product_name}'과 이름이 정확히 같은 보고품목이 없고 비슷한 품목만 있습니다: {', '.join(names)}. 보고번호를 입력하거나 통지서를 확인하세요.",
                     source=source, detail={"candidates": reports["candidates"][:5], "claimedFunctions": claimed})
    return check("functional_report", "not_found",
                 f"'{product_name}' 이름으로 기능성 보고품목을 찾지 못했습니다. 기능성 효능 표현이라면 보고번호나 심사결과통지서로 확인해야 합니다.",
                 source=source, detail={"claimedFunctions": claimed})


def substantiation_check(finding: dict, evidence: dict, evidence_count: int, matched: dict | None = None) -> dict:
    rows = finding.get("evidenceRequirements", [])
    required = "; ".join(f"{r['expression']}: {r['requirement']}" for r in rows) or \
        "사실과 관련한 표시광고는 실증할 수 있어야 합니다 (화장품법 제14조, 실증 규정 제3조)."
    source = rows[0]["source"] if rows else "화장품법 제14조"
    if evidence_count == 0:
        return check("substantiation", "evidence_missing",
                     f"수치나 시험 결과를 말하는 표현인데 제출된 증빙이 없습니다. 필요한 자료: {required}",
                     source=source, detail={"requirements": rows})
    if matched:
        needed = f" 추가로 받을 자료: {matched['needed']}." if matched.get("needed") and matched["verdict"] != "supports" else ""
        return check("substantiation", MATCH_STATUS[matched["verdict"]],
                     f"{matched.get('documentName') or '제출 증빙'}: {matched['reason']}{needed}",
                     source=source, detail={"requirements": rows, "match": matched})
    if not evidence.get("documents") and evidence.get("failures"):
        # 증빙을 읽으려 했지만 실패했다. 광고의 문제가 아니라 파일을 다시 확인할 일이다.
        # (도구 선택 loop가 증빙 도구를 부르지 않았으면 failures도 없으니 아래 evidence_submitted로 간다.)
        codes = ", ".join(sorted({f.get("code", "") for f in evidence.get("failures", [])})) or evidence.get("error") or "UNKNOWN"
        return check("substantiation", "evidence_unread",
                     f"제출한 증빙 {evidence_count}건을 읽지 못해 이 표현과 대조하지 못했습니다 ({codes}).",
                     source=source, detail={"requirements": rows, "evidenceStatus": evidence.get("status")})
    return check("substantiation", "evidence_submitted",
                 f"증빙 {evidence_count}건이 제출됐습니다. 이 표현을 뒷받침하는지 담당자가 대조해야 합니다. 필요한 자료: {required}",
                 source=source, detail={"requirements": rows, "evidenceStatus": evidence.get("status")})


async def ingredient_names(provider, findings: list[dict]) -> dict[str, list[str]]:
    targets = [f for f in findings if "ingredient" in f.get("requiredChecks", [])]
    if not targets:
        return {}
    if getattr(provider, "client", None):
        try:
            output = await provider.parse(
                instruction("ingredient_extraction"), IngredientMentions,
                text="\n".join(f"{f['id']}: {f['phrase']}" for f in targets))
            found = {item.finding_id: [n.strip() for n in item.names
                                       if 2 <= len(n.strip()) <= 60 and not generic_ingredient(n.strip())] for item in output.items}
            return {f["id"]: found.get(f["id"], []) for f in targets}
        except Exception:
            pass
    return {f["id"]: fallback_ingredients(f["phrase"]) for f in targets}


async def ingredient_check(call, name: str, day: str) -> dict:
    domestic, foreign = await asyncio.gather(
        call("rag", "check_ingredient_standard", {"ingredient": name, "review_date": day}),
        call("business", "lookup_ingredient_regulation", {"ingredient_name": name}), return_exceptions=True)
    domestic = domestic if isinstance(domestic, dict) else {"status": "error"}
    foreign = foreign if isinstance(foreign, dict) else {"status": "error"}
    # Only an exact name match counts; a substring of a longer chemical name is a different substance.
    exact_items = foreign.get("items", []) if foreign.get("exact") else []
    countries = sorted({c.strip() for item in exact_items for key in ("PROH_NATIONAL", "LIMIT_NATIONAL")
                        for c in str(item.get(key) or "").split(",") if c.strip()})
    detail = {"ingredient": name, "domestic": domestic, "foreignStatus": foreign.get("status"), "foreignCountries": countries[:20]}
    if domestic.get("status") == "prohibited":
        return check("ingredient", "prohibited", f"'{name}'은 화장품 안전기준 별표 1(사용할 수 없는 원료)에 있습니다.",
                     source="화장품 안전기준 등에 관한 규정", detail=detail)
    if domestic.get("status") == "restricted":
        return check("ingredient", "restricted", f"'{name}'은 별표 2(사용상의 제한이 필요한 원료)에 있습니다. 함량과 사용 범위를 확인하세요.",
                     source="화장품 안전기준 등에 관한 규정", detail=detail)
    if domestic.get("status") in ("error", "not_indexed") and foreign.get("status") in ("error", None):
        return check("ingredient", "unavailable", f"'{name}' 성분 기준을 조회하지 못했습니다.", detail=detail)
    note = f" 해외 금지 또는 제한 국가: {', '.join(countries[:5])}." if countries else ""
    return check("ingredient", "not_listed", f"'{name}'은 국내 사용 금지, 제한 원료 목록에서 찾지 못했습니다 (표기가 다르면 누락될 수 있음).{note}",
                 source="화장품 안전기준 등에 관한 규정", detail=detail)


async def cross_checks(findings: list[dict], *, product: dict, evidence: dict, evidence_count: int,
                       ad_text: str, provider, review_date: str, call=None, matches: dict | None = None) -> dict:
    needed = {c for f in findings for c in f.get("requiredChecks", [])}
    if not needed:
        return {"findings": findings, "summary": {}, "warnings": []}

    async def run(session):
        reports = None
        if needed & {"functional_report", "gmp"} and product.get("name"):
            try:
                reports = await session("business", "find_functional_reports",
                                        {"product_name": product["name"],
                                         "functional_approval_no": product.get("functionalReportNo") or None})
            except Exception:
                reports = None
        names = await ingredient_names(provider, findings)
        unique = sorted({n for ns in names.values() for n in ns})
        ingredient_results = dict(zip(unique, await asyncio.gather(*(ingredient_check(session, n, review_date) for n in unique))))
        gmp_result = None
        if "gmp" in needed:
            maker = next((r.get("manufacturer") or r.get("company") for r in (reports or {}).get("exact", [])), None)
            if maker:
                try:
                    gmp = await session("business", "verify_gmp_company", {"company_name": maker})
                    status = {"found": "gmp_listed", "not_found": "gmp_not_listed"}.get(gmp.get("status"), "unavailable")
                    gmp_result = check("gmp", status, f"보고품목의 업체 '{maker}': GMP 적합 업체 목록 {'에 있음' if status == 'gmp_listed' else '에서 찾지 못함' if status == 'gmp_not_listed' else ' 조회 실패'}.",
                                       source="식약처 화장품 GMP 적합 업체현황", detail={"company": maker, "items": gmp.get("items", [])[:3]})
                except Exception:
                    gmp_result = check("gmp", "unavailable", "GMP 업체 목록을 조회하지 못했습니다.")
            else:
                gmp_result = check("gmp", "needs_input", "제조업체를 알 수 없어 GMP 적합 여부를 대조하지 못했습니다. 보고번호나 제조사 정보를 확인하세요.")
        patent_numbers = re.findall(r"\b10[-– ]\d{7}\b", ad_text)
        out, summary = [], {}
        for finding in findings:
            results = []
            for kind in finding.get("requiredChecks", []):
                if kind == "functional_report":
                    # Only claims about a functional effect or the MFDS review itself are compared with the report.
                    if claimed_functions(finding["phrase"]) or FUNCTIONAL_CONTEXT.search(finding["phrase"]):
                        results.append(functional_check(finding, reports, product.get("name", "")))
                elif kind == "substantiation":
                    results.append(substantiation_check(finding, evidence, evidence_count, (matches or {}).get(finding["id"])))
                elif kind == "ingredient":
                    found = names.get(finding["id"], [])
                    if not found:
                        results.append(check("ingredient", "needs_input", "문구에서 성분 이름을 특정하지 못했습니다. 전성분표로 확인하세요."))
                    results += [ingredient_results[n] for n in found]
                elif kind == "gmp" and gmp_result:
                    results.append(gmp_result)
                elif kind == "patent":
                    results.append(check("patent", "needs_input" if not patent_numbers else "evidence_submitted",
                                         "특허 표현이 있지만 광고에서 특허번호를 찾지 못했습니다. 번호와 등록 여부를 확인하세요."
                                         if not patent_numbers else f"특허번호 {', '.join(patent_numbers[:3])}의 KIPRIS 조회 결과를 확인하세요.",
                                         source="KIPRIS Plus"))
            for r in results:
                summary.setdefault(r["type"], {}).setdefault(r["status"], 0)
                summary[r["type"]][r["status"]] += 1
            out.append({**finding, "checks": results})
        return out, summary, reports

    try:
        if call is None:
            async with McpTools() as session:
                out, summary, reports = await run(session)
        else:
            out, summary, reports = await run(call)
    except Exception:
        return {"findings": findings, "summary": {}, "warnings": ["CROSS_CHECK_UNAVAILABLE"]}
    return {"findings": out, "summary": summary, "warnings": [], "functionalReports": reports}
