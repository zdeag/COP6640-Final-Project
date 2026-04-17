import pandas as pd

from syco.kappa import compute_kappa_from_frame


def _frame(pairs):
    rows = []
    for i, (h, j) in enumerate(pairs):
        rows.append({
            "response_record_id": f"r{i}",
            "human_category": h,
            "judge_category": j,
            "human_appropriateness": 3,
            "judge_appropriateness": 3,
            "judge_model": "test",
            "judge_prompt_version": "v1",
        })
    return pd.DataFrame(rows)


def test_perfect_agreement_kappa_one():
    pairs = [(c, c) for c in "ABCDEFABCDEFABCDEF"]
    df = _frame(pairs)
    rep = compute_kappa_from_frame(df, rng_seed=0)
    assert rep.kappa_category == 1.0
    assert rep.passes


def test_random_agreement_low_kappa():
    # Judge always picks "A", human varies — agreement only by chance.
    pairs = [(c, "A") for c in "ABCDEFABCDEFABCDEF"]
    df = _frame(pairs)
    rep = compute_kappa_from_frame(df, rng_seed=0)
    assert rep.kappa_category < 0.2
    assert not rep.passes


def test_partial_agreement_2x2_known_value():
    # Classic 2x2 textbook example. Build with only two classes used.
    # 50 perfect matches + 10 swapped on a 60-item sample.
    pairs = [("D", "D")] * 40 + [("E", "E")] * 10 + [("D", "E")] * 5 + [("E", "D")] * 5
    df = _frame(pairs)
    rep = compute_kappa_from_frame(df, rng_seed=0)
    assert 0.55 < rep.kappa_category < 0.75


def test_confusion_shape():
    pairs = [(c, c) for c in "ABCDEF"]
    df = _frame(pairs)
    rep = compute_kappa_from_frame(df, rng_seed=0)
    assert rep.confusion.shape == (6, 6)
    assert list(rep.confusion.index) == list("ABCDEF")
    assert list(rep.confusion.columns) == list("ABCDEF")
