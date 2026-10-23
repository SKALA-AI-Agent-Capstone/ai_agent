from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class AssetRef(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    name: str
    role: Literal["advertisement", "evidence"]
    docType: str = ""


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reviewId: str
    tenantId: str
    versionNo: int = Field(ge=1)
    runToken: str = Field(min_length=32, repr=False)
    product: dict
    productEvidenceCount: int = Field(default=0, ge=0)  # evidence of the product Spring already holds a read result for
    assets: list[AssetRef] = Field(max_length=15)


class OCRLines(BaseModel):
    lines: list[str]
    unreadable: bool


class TextIssue(BaseModel):
    phrase: str
    reason: str
    claimType: str
    rewrites: list[str]


class TextAssessment(BaseModel):
    findings: list[TextIssue]


class VisualObservation(BaseModel):
    observation: str
    concern: str
    comparisonLimitations: list[str]
    needsAdjacentContext: bool


class VisualAssessment(BaseModel):
    sectionTitle: str
    findings: list[VisualObservation]
    unreadable: bool


class ResultTile(BaseModel):
    """result.tiles의 한 항목. 분석에 쓴 타일의 좌표만 남기고 픽셀은 저장하지 않는다.
    region은 원본 페이지 기준 0~1 [x0, y0, x1, y1]이고 index는 분석 순서(파일, 페이지, 위에서 아래)다.
    findings와 visualFindings의 tileId는 이 id 중 하나이거나, 대표 타일을 정할 수 없으면 null이다."""
    model_config = ConfigDict(extra="forbid")
    id: str
    index: int = Field(ge=0)
    fileId: str
    fileName: str
    page: int = Field(ge=1)
    region: list[float] = Field(min_length=4, max_length=4)
    # 구간 내용을 한 줄로 부른 이름(예: "임상 시험 결과"). 사진 경로가 실패하면 없다.
    title: str | None = None


class ResumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reviewId: str
    tenantId: str
    versionNo: int = Field(ge=1)
    state: Literal["approved", "rejected", "changes_requested"]


class RewriteRequest(BaseModel):
    """Backend가 사람의 "다시쓰기" 요청을 Agent에 전달하는 입력. 한도와 저장은 Backend가 맡는다."""
    model_config = ConfigDict(extra="forbid")
    reviewId: str
    tenantId: str
    phrase: str = Field(min_length=1, max_length=2000)
    reason: str = Field(default="", max_length=4000)
    claimType: str = Field(default="", max_length=100)
    previousRewrites: list[str] = Field(default_factory=list, max_length=50)
    retryNote: str | None = Field(default=None, max_length=1000)


class RewriteSuggestion(BaseModel):
    text: str


class VerifyRewriteRequest(BaseModel):
    """Backend가 rewritten_manual로 확정하기 전에 사람이 쓴 문구를 검증받는 입력.
    app.violations.final_text_validated=true가 되려면 이 호출이 먼저 통과해야 한다."""
    model_config = ConfigDict(extra="forbid")
    reviewId: str
    tenantId: str
    text: str = Field(min_length=1, max_length=2000)
    reason: str = Field(default="", max_length=4000)
    # 지적된 원문. 있으면 원문의 규칙 표현이 남았는지, 새 주장이 들어갔는지까지 검사한다.
    phrase: str = Field(default="", max_length=2000)


class RewriteVerification(BaseModel):
    passed: bool
    reasoning: str
