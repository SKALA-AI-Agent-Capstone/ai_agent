# AI 개발 작업 규칙

- 사용자 요청을 우선하고 작성 자료에 중앙점을 사용하지 않는다.
- 작업 전 README와 형제 backend/docs/00-project-context.md, 03-contracts.md, 08-project-memory.md를 읽는다. 공통 문서를 복제하지 않는다.
- 노션 기획서: https://app.notion.com/p/Agent-3fe616ab8fec80bfb92df115c66aa83d
- 데이터 설계: https://app.notion.com/p/Data-ERD-App-State-Vector-DB-3ff616ab8fec81c7b57ac75358ced570
- Agent, Business MCP, RAG MCP는 같은 저장소에서 관리하되 실행 서비스와 권한 경계는 유지한다.
- Python 계약 원본은 packages/shared, 생성 스키마는 contracts다. export_contracts.py --check를 실행한다.
- resources의 공개 목록은 원본이며 backend의 동일 목록은 배포용 스냅샷이다. check_backend_registry.py로 동기화를 검사한다.
- 기존 모델, 프롬프트, 임베딩과 검색 임계값 변경은 저장소 이동과 별도 브랜치로 한다.
- 업무 DB와 Redis의 업무 접근은 Backend 내부 API를 거친다. Agent DB 접근은 checkpoint, RAG DB 접근은 knowledge로 제한한다.
- 문구와 사진 분석 경로, 출처와 버전, 조회 실패와 미조회 구분을 유지한다.
- .env, 로컬 원본과 증빙, reports, 모델 캐시, DB dump와 비밀번호를 커밋하지 않는다. 실제 키를 로그에 출력하지 않는다.
- Python 테스트와 생성 계약 검사, 변경에 맞는 연동 검사를 수행한다. 수행하지 않은 검증을 통과로 기록하지 않는다.
- 승인 없이 main 병합이나 PR 생성, 데이터 초기화와 실행 서비스 재시작은 하지 않는다.
