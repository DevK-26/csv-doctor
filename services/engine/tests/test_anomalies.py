"""Tests for the anomaly ensemble (Phase A1).

Run:  python -m pytest tests/ -v   (from services/engine/)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from app.engine import anomalies
from app.engine.anomalies import (MODELS, calibrate, find_anomalies,
                                  prepare_features, robust_scale, score_matrix)
from app.engine.parse import infer_types, parse_csv_bytes

REPO = Path(__file__).resolve().parents[3]
SAMPLES = REPO / "samples"
FIXTURES = Path(__file__).parent / "fixtures"


def _load(path: Path):
    df = parse_csv_bytes(path.read_bytes())
    return df, infer_types(df)


def _clean_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Well-behaved correlated numeric data with nothing planted."""
    rng = np.random.default_rng(seed)
    units = rng.normal(40, 8, n).round()
    price = rng.normal(100, 5, n)
    return pd.DataFrame({
        "units": units,
        "revenue": (units * price).round(),
        "returns": rng.poisson(2, n),
        "region": rng.choice(["North", "South", "East", "West"], n),
    }).astype(str)


@pytest.fixture(scope="module")
def sales_result():
    df, types = _load(SAMPLES / "sales.csv")
    return df, find_anomalies(df, types)


# ---- the planted outlier ---------------------------------------------------

def test_planted_9800_units_row_is_flagged_with_reasons(sales_result):
    df, result = sales_result
    assert result["skipped"] is False
    planted = int(np.flatnonzero((df["units"] == "9800").to_numpy())[0])

    hit = [r for r in result["rows"] if r["row_index"] == planted]
    assert hit, "the 9800-units row was not flagged"
    row = hit[0]
    assert row["line_number"] == planted + 2
    assert row["score"] >= result["threshold"]

    # Reasons must exist, be human-readable and name the guilty columns.
    assert row["reasons"] and all(isinstance(r, str) and r for r in row["reasons"])
    named = {f["feature"] for f in row["top_features"]}
    assert named & {"units", "revenue"}
    text = " ".join(row["reasons"])
    assert "units" in text or "revenue" in text
    assert "9,800" in text or "980,000" in text
    for f in row["top_features"]:
        assert 0 < f["contribution"] <= 1


def test_planted_row_is_the_top_anomaly_and_all_detectors_agree(sales_result):
    df, result = sales_result
    top = result["rows"][0]
    assert df.iloc[top["row_index"]]["units"] == "9800"
    assert top["agreement"] == "3/3"
    assert all(top["votes"].values())


def test_sales_results_are_not_flooded(sales_result):
    df, result = sales_result
    assert result["flagged_count"] <= 0.05 * len(df)
    assert result["flagged_count"] == len(result["rows"])


def test_ids_are_never_used_as_features(sales_result):
    _, result = sales_result
    assert "order_id" not in result["features_used"]
    assert set(result["features_used"]) == {"units", "revenue", "returns"}


# ---- every flagged row is explained ---------------------------------------

@pytest.mark.parametrize("name", ["sales.csv", "analytics.csv", "survey.csv"])
def test_every_flagged_row_has_reasons_on_all_samples(name):
    df, types = _load(SAMPLES / name)
    result = find_anomalies(df, types)
    json.dumps(result, allow_nan=False)  # API-safe: no NaN/inf, no numpy types
    if result["skipped"]:
        assert result["reason"]
        return
    assert result["flagged_count"] <= 0.05 * len(df)
    for row in result["rows"]:
        assert row["reasons"]
        assert row["top_features"]
        assert set(row["model_scores"]) == set(MODELS)
        assert 0 <= row["score"] <= 1
        assert set(row["per_model_top_features"]) == set(MODELS)


def test_disagreement_between_detectors_is_reported():
    df = _clean_frame(300, seed=3)
    df.loc[10, "units"] = "4000"
    df.loc[10, "revenue"] = "400000"
    result = find_anomalies(df, infer_types(df), threshold=0.2)
    for row in result["rows"]:
        yes = sum(row["votes"].values())
        assert row["agreement"] == f"{yes}/3"
        assert f"{yes} of 3 detectors" in row["reasons"][-1]


# ---- clean / small / awkward inputs ---------------------------------------

def test_clean_data_does_not_flood():
    df = _clean_frame()
    result = find_anomalies(df, infer_types(df))
    assert result["skipped"] is False
    assert result["flagged_count"] <= 0.02 * len(df)


def test_small_clean_input_is_skipped_not_crashed():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5.0, 6.5, 7.1, 8.2]}).astype(str)
    result = find_anomalies(df, infer_types(df))
    assert result == {"skipped": True, "reason": result["reason"], "rows": []}
    assert "50 rows" in result["reason"]


def test_dirty_fixture_is_skipped_not_crashed():
    df, types = _load(FIXTURES / "dirty_sales.csv")
    result = find_anomalies(df, types)
    assert result["skipped"] is True and result["rows"] == []


def test_single_row_and_no_numeric_columns_do_not_crash():
    one = pd.DataFrame({"a": ["1"], "b": ["2"]})
    assert find_anomalies(one, infer_types(one))["skipped"] is True

    text_only = pd.DataFrame({"name": [f"n{i % 7}" for i in range(80)],
                              "city": [f"c{i % 5}" for i in range(80)]})
    result = find_anomalies(text_only, infer_types(text_only))
    assert result["skipped"] is True
    assert "numeric" in result["reason"]


def test_exactly_minimum_rows_runs():
    df = _clean_frame(anomalies.MIN_ROWS, seed=1)
    result = find_anomalies(df, infer_types(df))
    assert result["skipped"] is False
    assert result["rows_scored"] == anomalies.MIN_ROWS


def test_missing_and_garbage_cells_do_not_crash():
    df = _clean_frame(200, seed=2)
    df.loc[[3, 17, 40], "revenue"] = ""
    df.loc[[5, 9], "units"] = "n/a"
    result = find_anomalies(df, infer_types(df))
    assert result["skipped"] is False
    json.dumps(result, allow_nan=False)


def test_heavily_duplicated_rows_do_not_crash_or_flood():
    # Duplicates give zero neighbour distances — the classic LOF blow-up.
    base = _clean_frame(40, seed=4)
    df = pd.concat([base] * 6, ignore_index=True)
    result = find_anomalies(df, infer_types(df))
    assert result["skipped"] is False
    json.dumps(result, allow_nan=False)
    assert result["flagged_count"] <= 0.05 * len(df)


def test_numerically_constant_column_is_dropped():
    df = _clean_frame(120, seed=5)
    df["flat"] = ["1", "1.0"] * 60  # two spellings, one number
    types = infer_types(df)
    types["flat"] = "numeric"
    values, names, _ = prepare_features(df, types)
    assert "flat" not in names
    assert values.shape == (120, len(names))


# ---- determinism (CLAUDE.md hard rule 2) -----------------------------------

def test_same_seed_gives_identical_results():
    df, types = _load(SAMPLES / "sales.csv")
    first = find_anomalies(df, types, seed=7)
    second = find_anomalies(df, types, seed=7)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_scoring_does_not_disturb_global_torch_rng():
    import torch
    df, types = _load(SAMPLES / "sales.csv")
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    find_anomalies(df, types)
    assert torch.equal(torch.rand(3), expected)


# ---- building blocks -------------------------------------------------------

def test_calibrate_is_bounded_monotonic_and_quiet_on_constant_scores():
    rng = np.random.default_rng(0)
    raw = np.append(rng.normal(0, 1, 500), [15.0, 30.0])
    cal = calibrate(raw)
    assert cal.min() >= 0 and cal.max() <= 1
    order = np.argsort(raw)
    assert np.all(np.diff(cal[order]) >= 0)
    assert cal[-1] > 0.99 and cal[-2] > 0.99
    assert np.median(cal) < 0.05
    assert np.all(calibrate(np.full(50, 3.3)) < 0.05)


def test_robust_scale_is_not_stretched_by_the_outlier():
    col = np.append(np.arange(100, dtype=float), 1e9)
    scaled = robust_scale(np.column_stack([col, col[::-1]]))
    assert np.isfinite(scaled).all()
    assert np.abs(scaled[:100, 0]).max() < 3
    assert scaled[100, 0] == anomalies.CLIP


def test_score_matrix_shapes_and_weights():
    rng = np.random.default_rng(0)
    X = robust_scale(rng.normal(size=(120, 4)))
    out = score_matrix(X, seed=1, weights={"isolation_forest": 2, "autoencoder": 2})
    assert set(out["scores"]) == set(MODELS) | {"ensemble"}
    assert all(v.shape == (120,) for v in out["scores"].values())
    assert out["weights"]["local_outlier_factor"] == 0
    np.testing.assert_allclose(
        out["scores"]["ensemble"],
        0.5 * out["scores"]["isolation_forest"] + 0.5 * out["scores"]["autoencoder"])


def test_bad_inputs_raise_specific_errors():
    with pytest.raises(ValueError, match="unknown model"):
        score_matrix(np.zeros((10, 2)), weights={"knn": 1})
    with pytest.raises(ValueError, match="NaN"):
        score_matrix(np.full((10, 2), np.nan))
    with pytest.raises(ValueError, match="at least 3 rows"):
        score_matrix(np.zeros((2, 2)))


# ---- API wiring -------------------------------------------------------------

def test_analyse_endpoint_returns_anomalies_with_reasons():
    from fastapi.testclient import TestClient

    from app.main import app

    raw = (SAMPLES / "sales.csv").read_bytes()
    resp = TestClient(app).post("/analyse", files={"file": ("sales.csv", raw, "text/csv")})
    assert resp.status_code == 200
    body = resp.json()
    anom = body["anomalies"]
    assert anom["skipped"] is False
    assert anom["rows"], "API returned no anomaly rows for sales.csv"
    top = anom["rows"][0]
    assert top["reasons"] and top["top_features"]
    assert any("9,800" in r or "980,000" in r for r in top["reasons"])
    assert body["facts_pack"]["anomalies"]["flagged_count"] == anom["flagged_count"]


def test_analyse_endpoint_small_file_reports_skipped():
    from fastapi.testclient import TestClient

    from app.main import app

    raw = (FIXTURES / "dirty_sales.csv").read_bytes()
    resp = TestClient(app).post("/analyse", files={"file": ("d.csv", raw, "text/csv")})
    assert resp.status_code == 200
    assert resp.json()["anomalies"]["skipped"] is True
