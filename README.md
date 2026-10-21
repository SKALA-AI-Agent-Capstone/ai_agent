# 화장품 광고 검토 AI 서비스

FastAPI Agent, LangGraph, 광고 PDF와 이미지 파싱, OCR와 VLM, Business MCP, RAG MCP를 관리한다.
기능과 모델 설정은 backend 기준 커밋 `0933ae1`에서 이동했으며 이번 분리에는 성능 개선을 넣지 않는다.

## 저장소 경계

| 저장소 | 책임 |
| --- | --- |
| frontend | Vue 화면과 HTTP, SSE 연결 |
| backend | Spring Boot, 인증과 기업 권한, 파일과 업무 DB, 과금, RabbitMQ와 통합 Compose |
| ai_agent | Agent와 MCP, 입력 처리, 공개 근거 적재와 하이브리드 검색, AI 평가 |

## 작업 전

요구사항은 [노션 기획서](https://app.notion.com/p/Agent-3fe616ab8fec80bfb92df115c66aa83d), 데이터 설계는 [노션 ERD](https://app.notion.com/p/Data-ERD-App-State-Vector-DB-3ff616ab8fec81c7b57ac75358ced570)를 확인한다.
공통 아키텍처와 개발 기록은 [backend 문서](https://github.com/SKALA-AI-Agent-Capstone/backend/tree/main/docs)에 한 번만 관리한다.
저장소 분리는 main에 반영한다. 후속 기능은 main에서 기능 브랜치를 만들고 PR로 검토한다.

## 개발 환경

세 저장소를 같은 상위 폴더에 둔다. 전체 실행과 DB는 backend가 소유한다.

```text
beauty-capstone/
  frontend/
  backend/
  ai_agent/
```

```sh
# ai_agent에서 실행
uv sync --all-packages --frozen
uv run --all-packages --frozen pytest
uv run --all-packages --frozen python scripts/export_contracts.py --check
python3 scripts/check_backend_registry.py --backend ../backend
```

전체 실행은 backend에서 `python3 scripts/bootstrap.py` 후 `docker compose --profile web up --build -d`를 사용한다.
backend README의 main 실행 절차와 고정 AI 커밋을 함께 사용한다.
기존 backend/.env, data와 Docker 볼륨을 유지한다. 키나 원본 파일, DB를 복사하거나 초기화하지 않는다.

호스트에서 각 AI 서비스를 따로 켤 때는 backend에서 생성한 환경을 사용한다.

```sh
uv run --all-packages --frozen --env-file ../backend/.env capstone-agent
uv run --all-packages --frozen --env-file ../backend/.env business-mcp
uv run --all-packages --frozen --env-file ../backend/.env rag-mcp
```

각 명령은 별도 터미널에서 실행한다. 같은 포트의 Compose 서비스를 동시에 켜지 않는다.
AI 스크립트의 local_config는 ai_agent/.env가 없으면 형제 backend/.env를 읽는다. 독립 환경에는 .env.example을 참고하되 DB 비밀번호를 새로 만들지 않는다.
AI Agent URL과 도구 이름, 모델과 프롬프트, 검색 기준값은 이식 전과 같다.

## 내부 구조

- services/agent: Graph, 모델 호출, 입력 예산, OCR와 사진 분석, 수정안 검증, HITL 재개
- services/business-mcp: 실행 범위가 검증된 증빙 조회, 식약처와 특허 조회. 업무 DB는 직접 수정하지 않는다.
- services/rag-mcp: 공개 문서 적재, 규칙, BM25와 벡터 검색, 원문 Resource
- packages/shared: Python 입출력 모델과 순수 파서
- contracts: 위 모델에서 생성한 JSON Schema와 외부 API 명세
- resources: 공개 출처와 외부 API 목록의 원본
- evals, tests, scripts: 합성 평가셋, 회귀 검사와 수집 및 평가 도구

생성 계약은 `scripts/export_contracts.py`로 검사한다. 계약 변경은 Backend와 Frontend 소비자를 함께 검토한다.
Spring이 제공하는 두 목록 JSON은 backend resources의 배포용 스냅샷이다. 원본을 고칠 때
`python3 scripts/check_backend_registry.py --backend ../backend --write`로 갱신하고 양쪽 diff를 리뷰한다.

## 평가와 통합 검사

모델 호출이 필요한 평가는 기존 scripts/evaluate_harness.py와 scripts/evaluate_text_review.py를 사용한다.
유료 호출과 원본 수집은 단위 테스트와 구분한다. 평가 보고서는 reports 아래에 저장하고 Git에는 넣지 않는다.
전체 HTTP, 큐와 MCP 검사는 backend의 통합 CI가 이 저장소의 고정 커밋을 checkout해 실행한다.

## 이식 검증

기준 0933ae1의 추적 AI 파일 99개를 이동했다. 94개는 해시가 동일하고 5개는 경로와 설정 로딩만 조정했다. Python 104개(Backend 환경 검사 2개 포함), Java 55개와 격리된 Compose의 합성 HTTP, MCP, 큐와 승인 재개 검사가 통과했다. 유료 모델 호출과 성능 벤치마크는 실행하지 않았다. 상세 기록은 backend/docs/07-validation.md를 본다.
