"""Conservative screening for copy that states only a personal impression or wish."""

import re


_SUBJECTIVE_ENDING = re.compile(
    r"(?:(?:것|거)\s*같(?:다|아요|습니다|네요|은데요?|지만)|"
    r"느낌(?:이다|이에요|입니다|이네요|이야)?|"
    r"느껴(?:진다|져요|집니다)|"
    r"싶(?:다|어요|습니다|네요)|"
    r"(?:의사|의향)\s*(?:가|이)\s*(?:있다|있어요|있습니다|있음))$"
)
# Satisfaction ("높은 만족도의 모공개선 효과") reports how users felt, like "~것 같다"; a figure or a test beside it keeps it.
_SATISFACTION = re.compile(r"만족")
# A survey line often repeats itself in English ("지속적으로 사용할 의사가 있다 Be willing to continue to use").
_TRAILING_TRANSLATION = re.compile(r"\s+[A-Za-z][A-Za-z\s'’\-]*$")
_FACTUAL_SIGNAL = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:%|％|퍼센트|배)|"
    r"\d+\s*(?:초|분|시간|일|주|개월|회)\s*(?:만에|만으로|후|간|동안)|"
    r"시험\s*(?:결과|완료)|임상\s*(?:시험|결과|완료)|평가.{0,3}(?:완료|결과)|연구\s*결과|"
    r"입증(?:된|했다|한)|검증(?:된|했다|한)|"
    r"특허\s*(?:등록|취득|번호)|인증\s*(?:획득|완료)|허가\s*(?:완료|받은)|승인\s*(?:완료|받은)|"
    r"치료|완치|예방"
)


def is_subjective_only(phrase: str) -> bool:
    """Keep a mixed or measurable claim; dismiss only a stand-alone impression or wish."""
    copy = _TRAILING_TRANSLATION.sub("", (phrase or "").strip()).rstrip(".!?。…~ \"'”’")
    if re.search(r"[,;:\n]|[.!?]\s*\S", copy):
        return False  # Another clause may assert a separate fact.
    return bool(_SUBJECTIVE_ENDING.search(copy) or _SATISFACTION.search(copy)) and not _FACTUAL_SIGNAL.search(copy)
