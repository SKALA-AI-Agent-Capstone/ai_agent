"""Read-only external facts. No invented approval, patent status or legal sources."""
from datetime import datetime, timezone
import asyncio
import time
import os
import re
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

import httpx
from capstone_shared.settings import internal_token, required_env


import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def now():
    return datetime.now(timezone.utc).isoformat()


def endpoint(name, hostname, default=""):
    value=os.getenv(name,"").strip() or default
    parsed=urlsplit(value)
    if parsed.scheme!="https" or parsed.hostname!=hostname or parsed.query or parsed.fragment:
        return None
    return value


def normalize_items(body):
    items=body.get("items",[])
    if isinstance(items,dict):items=items.get("item",[items])
    if isinstance(items,dict):items=[items]
    if not isinstance(items,list):raise ValueError("INVALID_ITEMS")
    return items


def functional_item(row):
    functions=[label for field,label in [("EFFECT_YN1","미백"),("EFFECT_YN2","주름개선"),("EFFECT_YN3","자외선차단")]
               if str(row.get(field,"")).strip().upper()=="Y"]
    return {"productName":row.get("ITEM_NAME"),"reportSequence":row.get("COSMETIC_REPORT_SEQ"),
            "receiptNo":row.get("DEPT_RECEIPT_NO"),"functions":functions,"reportedAt":row.get("REPORT_DATE"),
            "efficacyText":row.get("EE_DOC_DATA", ""),"spf":row.get("SPF"),"pa":row.get("PA"),
            "manufacturer":row.get("MANUF_NAME"),"company":row.get("ENTP_NAME"),
            "scopeFields":{key:row.get(key) for key in ["EFFECT_YN1","EFFECT_YN2","EFFECT_YN3","EE_CODE","EE_NAME","WATER_PROOFING_FLAG"]},
            "note":"공식 보고품목의 필드를 반환한 보조 정보이며 광고 적법성이나 시험자료 충족 판정은 아닙니다."}


async def functional_report(functional_approval_no, product_name):
    source="식약처 기능성화장품 보고품목정보"
    result={"status":"error","source":source,"checkedAt":now(),"reportNo":functional_approval_no,
            "error":"API_NOT_CONFIGURED","data":None}
    key=os.getenv("MFDS_SERVICE_KEY","")
    url=endpoint("MFDS_REPORT_URL","apis.data.go.kr",
        "https://apis.data.go.kr/1471000/FtnltCosmRptPrdlstInfoService01/getRptPrdlstInq01")
    if not key or not url:return result
    try:
        matches=[];total=0;seen=0
        async with httpx.AsyncClient(timeout=12,follow_redirects=False) as client:
            for page in range(1,4):
                params={"serviceKey":unquote(key),"type":"json","pageNo":page,"numOfRows":100,"item_name":product_name}
                if re.fullmatch(r"[0-9]+",functional_approval_no):params["cosmetic_report_seq"]=functional_approval_no
                response=await client.get(url,params=params)
                response.raise_for_status();data=response.json()
                envelope=data.get("response",data)
                if str(envelope.get("header",{}).get("resultCode")) not in {"00","0"}:
                    return {**result,"error":"PROVIDER_REJECTED_REQUEST"}
                body=envelope.get("body",{});items=normalize_items(body)
                total=int(body.get("totalCount",0));seen+=len(items)
                matches.extend(row for row in items if str(row.get("ITEM_NAME","")).strip()==product_name.strip() and
                    functional_approval_no.strip() in [str(row.get(k,"")).strip() for k in ("COSMETIC_REPORT_SEQ","DEPT_RECEIPT_NO")])
                if seen>=total or not items:break
        if seen==0 and total==0:return {**result,"status":"not_found","error":None}
        if len(matches)!=1 or seen<total:
            return {**result,"status":"ambiguous","error":"IDENTIFIER_MAPPING_OR_PAGINATION_REQUIRES_CONFIRMATION"}
        row=matches[0]
        if str(row.get("CANCEL_APPROVAL_YN","")).upper()=="Y":
            return {**result,"status":"cancelled","error":"REPORTED_ITEM_CANCELLED","data":functional_item(row)}
        return {**result,"status":"found","error":None,"data":functional_item(row)}
    except Exception:
        return {**result,"error":"PROVIDER_REQUEST_FAILED"}


def parse_patents(xml,number,source):
    result={"id":number,"number":number,"status":"unavailable","source":source,"checkedAt":now(),
            "match":{"status":"unknown","verdict":"unknown","summary":"청구항 원문이 없어 광고 주장과 특허 권리범위를 비교하지 못했습니다.","reason":"청구항 원문 미확보"},
            "data":None,"error":None}
    root=ET.fromstring(xml)
    success=root.findtext(".//successYN")
    code=root.findtext(".//resultCode")
    if success!="Y" or (code and code not in {"00","0"}):
        return {**result,"status":"error","error":"PROVIDER_REJECTED_REQUEST"}
    digits=re.sub(r"\D","",number)
    matched=[]
    for item in root.findall(".//item"):
        raw=item.findtext("registerNumber","")
        normalized=re.sub(r"\D","",raw)
        # Seven-digit registration body versus the official kind/suffix representation.
        exact=normalized==digits or (len(digits)==9 and normalized==digits+"0000")
        if exact:matched.append(item)
    if len(matched)!=1:
        return {**result,"error":"NO_EXACT_MATCH_OR_INCOMPLETE_SEARCH"}
    item=matched[0];state=item.findtext("registerStatus","")
    status = "registered" if state == "등록" else "expired" if state == "소멸" else "unavailable"
    return {**result,"status":status,
        "data":{"title":item.findtext("inventionTitle",""),"inventionTitle":item.findtext("inventionTitle",""),
            "abstract":item.findtext("astrtCont",""),"registerStatus":state,"registerNumber":item.findtext("registerNumber",""),
            "applicant":item.findtext("applicantName",""),"registerDate":item.findtext("registerDate",""),"ipc":item.findtext("ipcNumber","")}}


async def patents(numbers):
    result=[];source="KIPRIS Plus getWordSearch"
    key=os.getenv("KIPRIS_SERVICE_KEY","")
    url=endpoint("KIPRIS_WORD_SEARCH_URL","plus.kipris.or.kr")
    for number in numbers[:10]:
        base={"id":number,"number":number,"status":"error","source":source,"checkedAt":now(),"data":None,
              "error":"API_NOT_CONFIGURED","match":{"status":"unknown","verdict":"unknown","summary":"원문 대조 미실행","reason":"원문 대조 미실행"}}
        if not re.fullmatch(r"10-\d{7}",number):
            result.append({**base,"status":"needs_confirm","error":"NUMBER_REQUIRES_CONFIRMATION"});continue
        if not key or not url:
            result.append(base);continue
        try:
            async with httpx.AsyncClient(timeout=12,follow_redirects=False) as client:
                response=await client.get(url,params={"word":re.sub(r"\D","",number),"year":0,"patent":"true","utility":"false","numOfRows":100,"pageNo":1,"ServiceKey":unquote(key)})
                response.raise_for_status()
            result.append(parse_patents(response.content,number,source))
        except Exception:
            result.append({**base,"error":"PROVIDER_REQUEST_FAILED"})
    return result


async def structure(tenant_id,review_id,documents,version_no,run_token):
    """Evidence for the Agent. Nothing is read here: product evidence is read once when it is registered on the product
    (Agent /internal/evidence/parse) and Spring serves the stored result. For a report that covers several products only
    the registered product's pages come back. Pages that could not be read are listed as failures, not hidden."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response=await client.get(required_env("BACKEND_BASE_URL")+f"/internal/reviews/{review_id}/product-evidence",
                params={"versionNo":version_no},headers={"Authorization":f"Bearer {internal_token()}","X-Run-Token":run_token})
            response.raise_for_status()
    except Exception:
        return {"status":"failed","text":"","documents":[],"failures":[{"code":"PRODUCT_EVIDENCE_FETCH_FAILED"}],
                "claimTypes":[],"requirements":[],"flags":[],"note":"제품 증빙 읽기 결과를 가져오지 못했습니다."}
    stored=response.json().get("documents",[])
    read,failures=[],[]
    for doc in stored:
        if doc.get("productNotInReport"):
            failures.append({"documentId":doc["id"],"name":doc.get("name"),"code":"PRODUCT_NOT_IN_REPORT"});continue
        for number in doc.get("unreadPages",[]):
            failures.append({"documentId":doc["id"],"name":doc.get("name"),"code":"OCR_UNAVAILABLE","page":number})
        text=doc.get("text","")
        read.append({"id":doc["id"],"name":doc.get("name"),"docType":doc.get("docType",""),"pages":doc.get("pages"),
                     "characters":len(text),"text":text[:30000],"issuer":doc.get("issuer"),"codes":doc.get("codes",[]),
                     "version":doc.get("version"),"productPages":doc.get("productPages",[]),"results":doc.get("results",[]),
                     "readerVersion":doc.get("readerVersion")})
    status="extracted" if read and not failures else "partial" if read else "not_submitted" if not stored else "failed"
    return {"status":status,"text":"\n".join(d["text"] for d in read)[:60000],"documents":read,
            "failures":failures,"claimTypes":[],"requirements":[],"flags":[],
            "note":"제품에 등록된 증빙의 읽기 결과입니다. 광고 주장과의 대조는 Agent가 따로 합니다."}


MFDS_LISTS={
    "gmp":{"source":"식약처 화장품 GMP 적합 업체현황","env":"MFDS_GMP_URL","match":"ENTP_NAME",
           "url":"https://apis.data.go.kr/1471000/CsmtcsGmpStbltCompInfo/getCsmtcsGmpStbltCompInfo",
           "note":"GMP 적합 업체 목록의 보조 확인이며 제품별 품질이나 광고 효능을 증명하지 않습니다."},
}


def compact(value):
    return re.sub(r"[\s()\[\]\u318d\u00b7,.-]","",str(value or "")).lower()


LIST_CACHE={}
LIST_TTL_SECONDS=24*3600
LIST_MAX_PAGES=120


async def full_list(kind,client,key,url):
    """These lists have no search parameter. Fetch every page once and keep it for a day."""
    cached=LIST_CACHE.get(kind)
    if cached and time.monotonic()-cached[0]<LIST_TTL_SECONDS:
        return cached[1],cached[2]
    async def page(n):
        response=await client.get(url,params={"serviceKey":unquote(key),"type":"json","pageNo":n,"numOfRows":100})
        response.raise_for_status();data=response.json();envelope=data.get("response",data)
        if str(envelope.get("header",{}).get("resultCode")) not in {"00","0"}:
            raise PermissionError("PROVIDER_REJECTED_REQUEST")
        body=envelope.get("body",{})
        return int(body.get("totalCount",0)),normalize_items(body)
    total,rows=await page(1)
    pages=min(LIST_MAX_PAGES,-(-total//100))
    limit=asyncio.Semaphore(5)
    async def guarded(n):
        async with limit:return (await page(n))[1]
    for chunk in await asyncio.gather(*(guarded(n) for n in range(2,pages+1))):
        rows+=chunk
    LIST_CACHE[kind]=(time.monotonic(),rows,total)
    return rows,total


async def mfds_list(kind,query):
    spec=MFDS_LISTS[kind]
    result={"status":"error","source":spec["source"],"checkedAt":now(),"query":query,
            "error":"API_NOT_CONFIGURED","items":[],"note":spec["note"]}
    key=os.getenv("MFDS_SERVICE_KEY","")
    url=endpoint(spec["env"],"apis.data.go.kr",spec["url"])
    if not key or not url:return result
    target=compact(query)
    if not target:return {**result,"error":"QUERY_REQUIRED"}
    try:
        async with httpx.AsyncClient(timeout=15,follow_redirects=False) as client:
            rows,total=await full_list(kind,client,key,url)
    except PermissionError:
        return {**result,"error":"PROVIDER_REJECTED_REQUEST"}
    except Exception:
        return {**result,"error":"PROVIDER_REQUEST_FAILED"}
    names=[compact(row.get(spec["match"])) for row in rows]
    exact=[row for row,name in zip(rows,names) if name==target]
    matches=exact or [row for row,name in zip(rows,names) if target in name]
    if len(rows)<total:
        return {**result,"status":"incomplete","error":"SCAN_LIMIT_REACHED","items":matches[:20],"scanned":len(rows),"total":total}
    return {**result,"status":"found" if matches else "not_found","error":None,"exact":bool(exact),
            "items":matches[:20],"scanned":len(rows),"total":total}


FUNCTION_FLAGS=[("EFFECT_YN1","미백"),("EFFECT_YN2","주름개선"),("EFFECT_YN3","자외선차단")]


EFFICACY_WORDS=[("미백","미백"),("주름","주름개선"),("자외선","자외선차단"),("탈모","탈모 증상 완화"),
                ("여드름","여드름성 피부 완화"),("튼살","튼살 완화"),("아토피","가려움 등 건조함 완화"),("염모","염모"),("제모","제모")]


def report_candidate(row):
    flagged=[label for field,label in FUNCTION_FLAGS if str(row.get(field,"")).strip().upper()=="Y"]
    efficacy=f"{row.get('EE_NAME') or ''} {row.get('EE_DOC_DATA') or ''}"
    flagged+=[label for word,label in EFFICACY_WORDS if word in efficacy and label not in flagged]
    return {"productName":row.get("ITEM_NAME"),"reportSequence":row.get("COSMETIC_REPORT_SEQ"),
            "receiptNo":row.get("DEPT_RECEIPT_NO"),"company":row.get("ENTP_NAME"),"manufacturer":row.get("MANUF_NAME"),
            "reportedAt":row.get("REPORT_DATE"),"cancelled":str(row.get("CANCEL_APPROVAL_YN","")).upper()=="Y",
            "functions":flagged,"efficacyName":row.get("EE_NAME"),
            "efficacyText":re.sub(r"<[^>]+>"," ",str(row.get("EE_DOC_DATA") or ""))[:600].strip(),
            "spf":row.get("SPF"),"pa":row.get("PA")}


async def functional_reports_by_name(product_name, report_no=None, max_pages=3):
    """보고품목을 제품명으로 찾는다. 이름이 정확히 같은 품목만 exact로 보고, 나머지는 후보로만 돌려준다."""
    source="식약처 기능성화장품 보고품목정보"
    result={"status":"error","source":source,"checkedAt":now(),"query":product_name,"reportNo":report_no or None,
            "error":"API_NOT_CONFIGURED","exact":[],"candidates":[],
            "note":"보고품목 조회 결과는 보조 정보이며 광고 적법성 판정이 아닙니다. 조회되지 않음은 미보고 확정이 아닙니다."}
    key=os.getenv("MFDS_SERVICE_KEY","")
    url=endpoint("MFDS_REPORT_URL","apis.data.go.kr",
        "https://apis.data.go.kr/1471000/FtnltCosmRptPrdlstInfoService01/getRptPrdlstInq01")
    if not key or not url:return result
    target=compact(product_name)
    if not target:return {**result,"error":"PRODUCT_NAME_REQUIRED"}
    rows=[];seen=0;total=0
    try:
        async with httpx.AsyncClient(timeout=12,follow_redirects=False) as client:
            for query in dict.fromkeys([product_name.replace(" ",""),product_name]):
                for page in range(1,max_pages+1):
                    params={"serviceKey":unquote(key),"type":"json","pageNo":page,"numOfRows":100,"item_name":query}
                    if report_no and re.fullmatch(r"[0-9]+",report_no):params["cosmetic_report_seq"]=report_no
                    response=await client.get(url,params=params)
                    response.raise_for_status();data=response.json()
                    envelope=data.get("response",data)
                    if str(envelope.get("header",{}).get("resultCode")) not in {"00","0"}:
                        return {**result,"error":"PROVIDER_REJECTED_REQUEST"}
                    body=envelope.get("body",{});items=normalize_items(body)
                    total=int(body.get("totalCount",0));seen+=len(items);rows+=items
                    if seen>=total or not items:break
                if rows:break
    except Exception:
        return {**result,"error":"PROVIDER_REQUEST_FAILED"}
    unique={str(r.get("ITEM_SEQ") or r.get("COSMETIC_REPORT_SEQ")):r for r in rows}.values()
    exact=[report_candidate(r) for r in unique if compact(r.get("ITEM_NAME"))==target]
    partial=[report_candidate(r) for r in unique if compact(r.get("ITEM_NAME"))!=target and target in compact(r.get("ITEM_NAME"))]
    status="found" if exact else "candidates" if partial else "not_found"
    return {**result,"status":status,"error":None,"exact":exact[:5],"candidates":partial[:10],"scanned":seen,"total":total}
