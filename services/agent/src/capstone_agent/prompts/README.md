# Agent 프롬프트

모델에 넘기는 시스템 프롬프트(Harness의 Instructions)다. 파일 하나가 프롬프트 하나이고, 코드는 `instruction("파일명")`으로 읽는다.
줄바꿈은 공백 하나로 합쳐지므로(마크다운 문단과 같다) 문장 단위로 줄을 나눠도 모델이 받는 문장은 바뀌지 않는다.
문구를 바꾸면 동작이 바뀌므로 PR에 바꾼 이유와 확인한 결과를 함께 적는다.

| 파일 | 쓰는 곳 |
| --- | --- |
| ad_copy_reading.md | 구간 이미지의 광고 문구 읽기(번호 상자 참고, 의미 덩어리와 역할, 제품 표면 글자 제외) |
| literal_reread.md | OCR과 비전 읽기가 다를 때 그 영역만 확대해서 글자 그대로 다시 읽기 |
| text_review.md | 광고 문구의 확인할 주장 추출 |
| claim_judgment.md | 추출된 후보와 규칙 후보가 실제로 문제 될 주장인지 판단, 문제 글자 범위와 근거 목적 지정 |
| basis_tool_selection.md | 법령 근거 도구 선택 루프(규칙, 법령 검색, 사례 중 목적에 맞는 도구 고르기) |
| basis_verification.md | 찾은 조항이 문구에 적용되는지 독립 검증(앞 판단을 보지 않음) |
| severity_assessment.md | 문구별 검토 우선순위 등급 |
| rewrite_draft.md | 수정 문구 작성 |
| rewrite_verification.md | 수정 문구 재검증(자동 제안과 사람이 쓴 문구 모두) |
| ingredient_extraction.md | 공식 기록 대조용 성분 이름 추출 |
| business_tool_selection.md | business 도구 선택 루프(어떤 대조 도구를 부를지 고르기) |
| evidence_summary.md | 제출 증빙 문서별 시험 요약(종류, 측정 항목, 기간, 대상, 기관, 결과) |
| evidence_match.md | 실증이 필요한 광고 주장과 증빙 요약 대조 |
