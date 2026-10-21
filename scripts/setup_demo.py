#!/usr/bin/env python3
"""Provision a tenant-scoped local test administrator; never print its secrets."""
import argparse
import os
from pathlib import Path
import httpx
from local_config import ROOT, load_env


def ensure_demo(api_url, access_file=None):
    email = os.environ.get('DEV_ADMIN_EMAIL', 'master@beauty-review.local')
    password = os.environ.get('DEV_ADMIN_PASSWORD', '')
    if len(password) < 16:
        raise SystemExit('Run bootstrap.py to generate DEV_ADMIN_PASSWORD. It is never committed.')
    client = httpx.Client(base_url=api_url.rstrip('/') + '/api/v1', timeout=30)
    login = client.post('/users/login', json={'email': email, 'password': password})
    if login.status_code == 401:
        login = client.post('/users/signup', json={'email': email, 'password': password,
            'name': 'MVP 검증 관리자', 'tenantName': 'Beauty Review 검증 작업공간'})
    if login.status_code != 200:
        raise SystemExit(f'Demo account provisioning failed ({login.status_code}); existing passwords are not reset.')
    identity = login.json()
    if identity['user']['role'] != 'admin':
        raise SystemExit('The local test account must be its own tenant administrator.')
    path = Path(access_file) if access_file else ROOT / 'data' / 'demo-access.md'
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, 'w') as handle:
        handle.write(f'''# 로컬 MVP 검증 계정\n\n- 이메일: `{email}`\n- 비밀번호: `{password}`\n- 역할: 검증 작업공간의 관리자\n- API: {api_url}\n\n이 파일과 .env는 Git에서 제외한다. 다른 기업의 권한을 우회하는 전역 관리자 계정이 아니다.\n팀원은 자신의 .env와 로컬 DB에서 setup_demo.py를 실행해 별도 비밀번호로 계정을 만든다.\n''')
    return client, identity, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-url', default='http://127.0.0.1:8080')
    parser.add_argument('--env-file')
    args = parser.parse_args()
    load_env(args.env_file)
    client, identity, path = ensure_demo(args.api_url)
    client.close()
    print('Local tenant administrator ready. Credentials are in the ignored data/demo-access.md file.')


if __name__ == '__main__':
    main()
