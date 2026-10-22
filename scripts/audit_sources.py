#!/usr/bin/env python3
"""Audit Notion's nine references and four APIs. Never turn HTTP 200 into legal verification."""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
from html import unescape
from io import BytesIO
import json
import os
from pathlib import Path
import re
from urllib.parse import urljoin, urlsplit, unquote, urlencode
import xml.etree.ElementTree as ET
import httpx
import pdfplumber
from capstone_rag_mcp.preview import chunk_pages
from capstone_business_mcp.review_tools import normalize_items
from local_config import ROOT, load_env

HOSTS = {'www.mfds.go.kr','mfds.go.kr','kcia.or.kr','www.kcia.or.kr','www.law.go.kr','law.go.kr',
         'www.data.go.kr','data.go.kr','plus.kipris.or.kr','apis.data.go.kr'}
MAX_BYTES = 32 * 1024 * 1024


async def fetch(client, url):
    for _ in range(4):
        parsed = urlsplit(url)
        if parsed.scheme != 'https' or parsed.hostname not in HOSTS:
            raise ValueError('UNAPPROVED_SOURCE_HOST')
        async with client.stream('GET', url) as response:
            if response.status_code in (301,302,303,307,308):
                url=urljoin(url,response.headers['location']);continue
            response.raise_for_status()
            body=bytearray()
            async for part in response.aiter_bytes():
                body.extend(part)
                if len(body)>MAX_BYTES:raise ValueError('SOURCE_TOO_LARGE')
            return bytes(body), response.status_code
    raise ValueError('REDIRECT_LIMIT')


def pdf_preview(raw, source, folder):
    with pdfplumber.open(BytesIO(raw)) as pdf:
        if len(pdf.pages)>500:raise ValueError('PUBLIC_DOCUMENT_PAGE_LIMIT')
        pages=[page.extract_text() or '' for page in pdf.pages]
    chunks=chunk_pages(pages)
    normalize=lambda value:re.sub('[^가-힣a-zA-Z0-9]', '', value)
    head=normalize(' '.join(pages[:3]))
    expected={'S01':['화장품','관리지침'],'S02':['화장품','광고자문'],
              'S04':['화장품','위반사례'],'S06':['화장품법시행규칙','별표5'],'S08':['기능성화장품','심사에관한규정']}
    if source['id'] in expected and not all(term in head for term in expected[source['id']]):
        raise ValueError('DOCUMENT_TITLE_REQUIRES_CONFIRMATION')
    sha=hashlib.sha256(raw).hexdigest()
    (folder/(source['id']+'.pdf')).write_bytes(raw)
    preview={'sourceId':source['id'],'sourceUrl':source['official_page'],'contentSha256':sha,
             'sourceVerified':False,'retrievalReady':False,'collection':source['collection'],'chunks':chunks,
             'note':'수집과 기본 문자 분할만 확인. 시행일과 예외, 표 및 rules/cases 분류 검증과 임베딩은 별도다.'}
    (folder/(source['id']+'.chunks.json')).write_text(json.dumps(preview,ensure_ascii=False,indent=2))
    return {'documentStatus':'pdf_downloaded','pages':len(pages),'textCharacters':sum(map(len,pages)),
            'chunks':len(chunks),'sha256':sha,'retrievalReady':False,'titleMatched':True}


def candidates(html, source):
    if source.get('verified_direct_download_url'):
        return [source['verified_direct_download_url']]
    if source.get('candidate_download_url'):
        return [source['candidate_download_url']]
    norm=lambda s:re.sub('[^가-힣a-zA-Z0-9]','',s).lower()
    keyword=norm(source.get('preferred_attachment_keyword',''))
    # MFDS lists the attachment name in <strong>, then the exact download anchor.
    options=[]
    for name,href in re.findall(r'<strong[^>]*>([^<]*?\.pdf)</strong>\s*.*?<a\s+href="([^"]+)"[^>]*class="bbs_icon_filedown"',html,re.S):
        title=re.sub('<[^>]+>','',unescape(name))
        if '.pdf' in title.lower() and keyword and keyword in norm(title):
            options.append(urljoin(source['official_page'],unescape(href)))
    return options


async def law_preview(client, source, folder):
    oc=os.getenv('LAW_OC','').strip()
    if not oc:return None
    names={'S05':'화장품법','S07':'화장품 표시광고 실증에 관한 규정','S09':'표시광고의 공정화에 관한 법률'}
    sid=source['id'];expected=names[sid]
    normalize=lambda value:re.sub('[^가-힣a-zA-Z0-9]','',value)
    endpoint='https://www.law.go.kr/DRF/lawService.do'
    if sid=='S07':
        params={'OC':oc,'target':'admrul','type':'XML','LM':'화장품 표시'+chr(0x00b7)+'광고 실증에 관한 규정'}
    else:
        raw,_=await fetch(client,'https://www.law.go.kr/DRF/lawSearch.do?'+urlencode(
            {'OC':oc,'target':'eflaw','type':'XML','query':('표시' if sid=='S09' else expected),'display':100,'nw':3}))
        root=ET.fromstring(raw);found=[]
        for node in root.iter():
            fields={child.tag.split('}')[-1]:(child.text or '').strip() for child in node if not len(child)}
            title=fields.get('법령명한글') or fields.get('법령명_한글') or fields.get('법령명')
            if title and normalize(title)==normalize(expected) and fields.get('법령ID'):
                found.append(fields['법령ID'])
        if len(set(found))!=1:raise ValueError('EXACT_LAW_ID_REQUIRES_CONFIRMATION')
        params={'OC':oc,'target':'eflaw','type':'XML','ID':found[0]}
    raw,_=await fetch(client,endpoint+'?'+urlencode(params));root=ET.fromstring(raw)
    text=' '.join(root.itertext())
    if normalize(expected) not in normalize(text) or len(text)<100:
        raise ValueError('LAW_BODY_REQUIRES_CONFIRMATION')
    sections=[]
    selected={'S05':{'13','14'},'S09':{'3'}}.get(sid)
    for node in root.iter():
        if node.tag.split('}')[-1]!='조문단위':continue
        number=next((c.text for c in node if c.tag.split('}')[-1]=='조문번호'),None)
        if selected and number not in selected:continue
        content='\n'.join((n.text or '').strip() for n in node.iter()
            if n.tag.split('}')[-1] in {'조문내용','항내용','호내용','목내용'} and n.text)
        if content:sections.append(content)
    if not sections and not selected:sections=[text]
    if not sections:raise ValueError('REQUESTED_LAW_ARTICLES_NOT_FOUND')
    chunks=chunk_pages(sections)
    for chunk in chunks:chunk['sectionIndex']=chunk['page'];chunk['page']=None
    (folder/(sid+'.xml')).write_bytes(raw)
    (folder/(sid+'.chunks.json')).write_text(json.dumps({'sourceId':sid,'sourceVerified':False,'retrievalReady':False,
        'sourceUrl':source['official_page'],'chunks':chunks},ensure_ascii=False,indent=2))
    return {'documentStatus':'xml_downloaded','chunks':len(chunks),'textCharacters':sum(map(len,sections)),
            'sha256':hashlib.sha256(raw).hexdigest(),'titleMatched':True,'retrievalReady':False}


async def check_source(client, source, folder, download):
    result={'id':source['id'],'kind':'document','title':source['title'],'officialPage':source['official_page'],
            'sourceVerified':False,'retrievalReady':False}
    try:
        raw,status=await fetch(client,source['official_page'])
        result.update(pageHttp=status,documentStatus='page_only')
        if download and (folder/(source['id']+'.pdf')).exists():
            cached=(folder/(source['id']+'.pdf')).read_bytes()
            result.update(await asyncio.to_thread(pdf_preview,cached,source,folder))
        if download and result['documentStatus']=='page_only':
            for url in candidates(raw.decode('utf-8','replace'),source):
                try:
                    body,_=await fetch(client,url)
                    if not body.lstrip().startswith(b'%PDF-'):continue
                    result.update(await asyncio.to_thread(pdf_preview,body,source,folder));break
                except Exception:
                    continue
        if download and result['documentStatus']=='page_only' and source['id'] in {'S05','S07','S09'}:
            result.update(await law_preview(client,source,folder) or {})
        if result['documentStatus']=='page_only':
            result['nextAction']='법제처 API는 LAW_OC가 필요하고, PDF는 공식 첨부파일을 직접 확인해야 한다.'
    except Exception as error:
        result.update(documentStatus='page_unavailable',errorType=type(error).__name__)
    return result


async def check_api(client, api, probe_without_key=False):
    result={'id':api['id'],'kind':'api','title':api['title'],'officialPage':api['official_page'],
            'operationUrl':api['request_url'],'role':api['role'],'runtimeStatus':'key_required'}
    try:
        _,status=await fetch(client,api['official_page']);result['pageHttp']=status
    except Exception as error:result['pageErrorType']=type(error).__name__
    key=os.getenv(api['service_key_env'],'').strip()
    if not key:
        if probe_without_key:
            try:
                params={'word':'10-1234567','numOfRows':1,'pageNo':1} if api['id']=='API02' else {'pageNo':1,'numOfRows':1,'type':'json'}
                response=await client.get(api['request_url'],params=params)
                result['unauthenticatedProbeHttp']=response.status_code
                result['unauthenticatedProbe']='response_received_key_still_required'
            except Exception as error:result['unauthenticatedProbe']=type(error).__name__
        return result
    try:
        if api['id']=='API02':
            from capstone_business_mcp.review_tools import patents
            found=await patents(['10-1234567'])
            result.update(runtimeStatus=found[0]['status'],providerError=found[0].get('error'))
        else:
            response=await client.get(api['request_url'],params={'serviceKey':unquote(key),'pageNo':1,'numOfRows':1,'type':'json'})
            response.raise_for_status();envelope=response.json();envelope=envelope.get('response',envelope)
            code=str(envelope.get('header',{}).get('resultCode'))
            result['providerCode']=code
            if code not in ('00','0'):
                result['runtimeStatus']='provider_rejected'
            else:
                items=normalize_items(envelope.get('body',{}))
                result.update(runtimeStatus='normal_response',itemCount=len(items),
                              responseColumns=sorted(items[0]) if items else [])
    except Exception as error:result.update(runtimeStatus='request_failed',errorType=type(error).__name__)
    return result


async def main(args):
    load_env(args.env_file)
    sources=json.loads((ROOT/'resources/reference-sources.json').read_text())
    apis=json.loads((ROOT/'resources/external-api-catalog.json').read_text())
    folder=ROOT/'data/source-audit';folder.mkdir(parents=True,exist_ok=True)
    async with httpx.AsyncClient(timeout=25,follow_redirects=False,headers={'User-Agent':'BeautyReview-Local-Source-Audit/1.0'}) as client:
        limit=asyncio.Semaphore(4)
        async def source(s):
            async with limit:return await check_source(client,s,folder,args.download)
        async def api(s):
            async with limit:return await check_api(client,s,args.probe_without_key)
        results=await asyncio.gather(*[source(s) for s in sources],*[api(s) for s in apis])
    report={'checkedAt':datetime.now(timezone.utc).isoformat(),'documentCount':len(sources),'apiCount':len(apis),
            'retrievalReady':False,'results':results}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    for result in results:
        print(json.dumps({k:v for k,v in result.items() if k in ('id','kind','pageHttp','documentStatus','pages','chunks','runtimeStatus','errorType')},ensure_ascii=False))
    print('Source audit saved. Original documents and chunk previews remain ignored local data.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--env-file')
    parser.add_argument('--download',action='store_true')
    parser.add_argument('--probe-without-key',action='store_true')
    parser.add_argument('--report',type=Path,default=ROOT/'reports/source-audit.json')
    asyncio.run(main(parser.parse_args()))
