"""Copy that states only a personal impression, wish or intention is not a claim to check."""
from capstone_agent.claim_screening import is_subjective_only


def test_an_intention_to_keep_using_is_subjective_like_a_wish():
    assert is_subjective_only("지속적으로 사용할 의사가 있다")
    assert is_subjective_only("계속 사용할 의향이 있습니다")
    assert is_subjective_only("지속적으로 사용할 의사가 있다 Be willing to continue to use")   # the English repeat


def test_a_figure_or_a_test_beside_the_intention_still_keeps_the_line():
    assert not is_subjective_only("95%가 지속적으로 사용할 의사가 있다")
    assert not is_subjective_only("임상 결과 지속적으로 사용할 의사가 있다")
    assert not is_subjective_only("모공부위 리프팅 개선 평가 완료")


def test_a_satisfaction_statement_is_subjective_unless_a_figure_or_test_backs_it():
    assert is_subjective_only("높은 만족도의 모공개선 효과")
    assert not is_subjective_only("만족도 95%의 모공개선 효과")
