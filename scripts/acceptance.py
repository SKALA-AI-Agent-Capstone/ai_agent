#!/usr/bin/env python3
"""Exercise real user-selected ads through HTTP. Reports and originals stay out of Git."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from capstone_shared.documents import prepare_document
from local_config import ROOT, load_env
from setup_demo import ensure_demo
from mvp_smoke import synthetic_pdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs='+', type=Path, required=True)
    parser.add_argument('--api-url', default='http://127.0.0.1:8080')
    parser.add_argument('--env-file')
    parser.add_argument('--include-pdf', action='store_true', help='Also test a synthetic native PDF')
    parser.add_argument('--report', type=Path, default=ROOT / 'reports' / 'acceptance.json')
    parser.add_argument('--wait-seconds', type=int, default=360)
    args = parser.parse_args()
    load_env(args.env_file)
    client, identity, _ = ensure_demo(args.api_url)
    client.headers['Authorization'] = 'Bearer ' + identity['accessToken']
    cases = [(p.name, p.read_bytes()) for p in args.input]
    if args.include_pdf:
        cases.append(('synthetic-native.pdf', synthetic_pdf()))
    report = {'checkedAt': datetime.now(timezone.utc).isoformat(), 'apiUrl': args.api_url,
              'customerAssetsInGit': False, 'cases': []}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    for index, (name, raw) in enumerate(cases, 1):
        case = {'case': f'A{index}', 'fileName': name, 'bytes': len(raw),
                'sha256': hashlib.sha256(raw).hexdigest()}
        started = time.monotonic()
        try:
            _, tiles = prepare_document(raw, {'id': f'A{index}', 'name': name})
            case['tiles'] = [{'page': t.page, 'box': list(t.box), 'sourceSize': list(t.source_size)} for t in tiles]
            case['tileCount'] = len(tiles)
            # Each original page is covered by content sections that touch without a gap or overlap.
            for page in {t.page for t in tiles}:
                group = [t for t in tiles if t.page == page]
                assert group[0].box[1] == 0 and group[-1].box[3] == group[-1].source_size[1]
                assert all(a.box[3] == b.box[1] for a, b in zip(group, group[1:]))
            case['tileCoverage'] = 'pass'
            metadata = {'product': {'name': f'검증 A{index} - {name}', 'functionalReportNo': ''}, 'media': [], 'evidence': []}
            response = client.post('/reviews', data={'metadata': json.dumps(metadata)},
                files={'files': (name, raw, 'application/pdf' if raw.startswith(b'%PDF-') else 'image/png')}, timeout=60)
            case['uploadHttp'] = response.status_code
            if response.status_code != 202:
                case['status'] = 'upload_failed'
            else:
                rid = response.json()['id'];case['reviewId'] = rid
                deadline = time.monotonic() + args.wait_seconds
                while time.monotonic() < deadline:
                    current = client.get(f'/reviews/{rid}');current.raise_for_status()
                    version = current.json()['versions'][-1]
                    if version['status'] not in ('queued', 'analyzing'):
                        break
                    time.sleep(1)
                case['status'] = version['status']
                case['error'] = version.get('error')
                if version.get('result'):
                    result = version['result']
                    case.update(textBlocks=len(result['texts']), textCharacters=sum(len(t['text']) for t in result['texts']),
                        textFindings=len(result['findings']), visualFindings=len(result['visualFindings']),
                        pipelineStatus=result['pipelineStatus'], retrievalStatus=result['retrievalStatus'],
                        legalStatus=result['legalStatus'], warnings=result['warnings'],
                        approvalState=version['approval']['state'], originalsDiscardedAt=version.get('originalsDiscardedAt'))
                    case['realModelCheck'] = ('pass' if all(v == 'complete' for v in result['pipelineStatus'].values())
                                              else 'not_passed')
        except Exception as error:
            case.update(status='error', errorType=type(error).__name__)
        case['elapsedSeconds'] = round(time.monotonic() - started, 2)
        report['cases'].append(case)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps({k: case[k] for k in ('case', 'status', 'uploadHttp', 'tileCount', 'textBlocks', 'pipelineStatus') if k in case}, ensure_ascii=False), flush=True)
    client.close()
    print('Detailed local report saved. No raw input or credentials were printed.')


if __name__ == '__main__':
    main()
