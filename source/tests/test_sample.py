import numpy as np
import pandas as pd

from syco.sample import draw_stratified
from syco.schemas import CATEGORIES, SEVERITIES


def _synthetic(n_per_cell: int = 5, models=("m1", "m2", "m3")) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    rid = 0
    for s in SEVERITIES:
        for c in CATEGORIES:
            for _ in range(n_per_cell):
                rows.append({
                    "response_record_id": f"r{rid}",
                    "severity": s,
                    "category": c,
                    "domain": "spiritual",
                    "model_label": rng.choice(models),
                    "prompt": f"p{rid}",
                    "response": f"resp{rid}",
                    "appropriateness": 3,
                })
                rid += 1
    return pd.DataFrame(rows)


def test_stratified_draw_target_size():
    df = _synthetic(n_per_cell=5)
    drawn = draw_stratified(df, n=60, rng_seed=7)
    assert 55 <= len(drawn) <= 60  # may be < 60 only if cells are short


def test_each_cell_represented():
    df = _synthetic(n_per_cell=5)
    drawn = draw_stratified(df, n=60, rng_seed=7)
    cells_seen = drawn.groupby(["severity", "category"]).ngroups
    assert cells_seen == len(SEVERITIES) * len(CATEGORIES)


def test_model_balance_within_tolerance():
    df = _synthetic(n_per_cell=5, models=("m1", "m2", "m3", "m4", "m5"))
    drawn = draw_stratified(df, n=60, rng_seed=7)
    counts = drawn["model_label"].value_counts()
    # Each model should be within +/- 50% of equal share.
    target = len(drawn) / counts.shape[0]
    assert counts.min() >= target * 0.5
    assert counts.max() <= target * 1.6


def test_handles_sparse_cells_gracefully():
    df = _synthetic(n_per_cell=1)  # only 30 rows total
    drawn = draw_stratified(df, n=60, rng_seed=7)
    # Can't return more than what's available, but shouldn't crash.
    assert len(drawn) <= 30
