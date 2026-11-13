"""Phase A3 tests for the hand-written drift engine (app/engine/drift.py).

Run:  python -m pytest tests/test_drift.py -v   (from services/engine/)
Every stream is generated from a fixed seed, so these tests are deterministic.
Sizes are kept small: the whole file runs in about a second.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from app.engine import drift
from app.engine.drift import (
    ADWIN,
    DriftInputError,
    PageHinkley,
    WindowedDetector,
    clip_to_fences,
    compare_categorical,
    compare_numeric,
    detect_stream,
    find_change_points,
    js_divergence,
    js_from_distributions,
    ks_statistic,
    ks_test,
    psi,
    psi_from_distributions,
)


def _normal(seed: int, n: int, mean: float = 0.0) -> np.ndarray:
    return np.random.default_rng(seed).normal(mean, 1.0, n)


def _shift_stream(seed: int = 42, n: int = 2000, at: int = 1000, jump: float = 1.5) -> np.ndarray:
    stream = _normal(seed, n)
    stream[at:] += jump
    return stream


# ---------------------------------------------------------------- batch: same distribution

def test_identical_samples_score_zero():
    x = _normal(1, 500)
    assert psi(x, x.copy()) == pytest.approx(0.0, abs=1e-12)
    assert js_divergence(x, x.copy()) == pytest.approx(0.0, abs=1e-12)
    assert ks_statistic(x, x.copy()) == 0.0
    assert ks_test(x, x.copy())[1] == pytest.approx(1.0)


def test_two_draws_of_one_distribution_are_stable():
    a, b = _normal(1, 1000), _normal(2, 1000)
    assert psi(a, b) < drift.PSI_MODERATE
    assert js_divergence(a, b) < drift.JS_MODERATE
    assert ks_test(a, b)[1] > drift.ALPHA
    verdict = compare_numeric(a, b)
    assert verdict["level"] == "stable"
    assert verdict["drifted"] is False


# ---------------------------------------------------------------- batch: clear shift

def test_clear_shift_is_significant_on_every_statistic():
    a, b = _normal(1, 1000), _normal(2, 1000, mean=1.5)
    assert psi(a, b) > drift.PSI_SIGNIFICANT
    assert js_divergence(a, b) > drift.JS_SIGNIFICANT
    statistic, p_value = ks_test(a, b)
    assert statistic > 0.4
    assert p_value < drift.ALPHA
    verdict = compare_numeric(a, b)
    assert verdict["level"] == "significant"
    assert verdict["drifted"] is True


def test_known_values_of_the_distribution_formulas():
    # No overlap at all: JS hits its ceiling of exactly 1 bit.
    assert js_from_distributions([1, 0], [0, 1]) == pytest.approx(1.0)
    # PSI of (0.5, 0.5) vs (0.9, 0.1): (0.4)ln(1.8) + (-0.4)ln(0.2)
    expected = 0.4 * np.log(0.9 / 0.5) + (-0.4) * np.log(0.1 / 0.5)
    assert psi_from_distributions([0.5, 0.5], [0.9, 0.1]) == pytest.approx(expected)
    # Symmetry of JS.
    assert js_from_distributions([3, 1, 1], [1, 1, 3]) == pytest.approx(
        js_from_distributions([1, 1, 3], [3, 1, 1]))


def test_categorical_shift_is_detected_and_labels_are_masked():
    ref = ["a"] * 300 + ["b"] * 300 + ["c"] * 300
    same = ["a"] * 290 + ["b"] * 310 + ["c"] * 300
    moved = ["a"] * 800 + ["b"] * 50 + ["c"] * 50
    assert compare_categorical(ref, same)["drifted"] is False
    verdict = compare_categorical(ref, moved)
    assert verdict["level"] == "significant"
    assert verdict["biggest_shifts"][0]["current_share"] > verdict["biggest_shifts"][0]["reference_share"]


# ---------------------------------------------------------------- bad input

@pytest.mark.parametrize("call", [psi, js_divergence, ks_statistic, ks_test, compare_numeric])
def test_empty_input_raises_a_clear_error(call):
    with pytest.raises(DriftInputError, match="no numeric values"):
        call([], [1.0, 2.0, 3.0])
    with pytest.raises(DriftInputError, match="no numeric values"):
        call([1.0, 2.0, 3.0], [])


def test_all_missing_input_counts_as_empty():
    with pytest.raises(DriftInputError, match="no numeric values"):
        psi([np.nan, np.nan], [1.0, 2.0])


def test_text_values_and_bad_parameters_raise_clear_errors():
    with pytest.raises(DriftInputError, match="numbers only"):
        psi(["a", "b"], [1.0, 2.0])
    with pytest.raises(DriftInputError, match="bins"):
        psi([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], bins=1)
    with pytest.raises(DriftInputError, match="no category labels"):
        compare_categorical([], ["a"])
    with pytest.raises(DriftInputError):
        PageHinkley(threshold=0)
    with pytest.raises(DriftInputError):
        ADWIN(delta=2.0)
    with pytest.raises(DriftInputError, match="unknown statistic"):
        WindowedDetector("chi2")


def test_streaming_detectors_reject_missing_values():
    with pytest.raises(DriftInputError):
        PageHinkley().update(float("nan"))
    with pytest.raises(DriftInputError):
        detect_stream([1.0, np.inf], ADWIN())


# ---------------------------------------------------------------- constant columns

def test_constant_columns_do_not_crash_and_give_sensible_results():
    same = [5.0] * 100
    assert psi(same, same) == pytest.approx(0.0, abs=1e-12)
    assert js_divergence(same, same) == pytest.approx(0.0, abs=1e-12)
    assert ks_statistic(same, same) == 0.0
    verdict = compare_numeric(same, same)
    assert verdict["level"] == "stable"

    # A constant reference against a column that moved away from it: drift.
    moved = compare_numeric(same, [9.0] * 100)
    assert moved["level"] == "significant"
    assert moved["ks_statistic"] == 1.0


def test_constant_stream_is_quiet_for_every_detector():
    flat = [3.0] * 400
    assert detect_stream(flat, PageHinkley()) == []
    assert detect_stream(flat, ADWIN(value_range=1.0)) == []
    assert detect_stream(flat, WindowedDetector("psi", window=50)) == []
    clipped, width = clip_to_fences(flat)
    assert width == 0.0
    assert (clipped == 3.0).all()


# ---------------------------------------------------------------- streaming detectors

def _detectors(stream: np.ndarray) -> dict:
    """The two product detectors, configured the way find_change_points does."""
    clipped, width = clip_to_fences(stream)
    return {"page_hinkley": (PageHinkley(), clipped),
            "adwin": (ADWIN(value_range=width), clipped)}


def test_page_hinkley_and_adwin_fire_on_a_planted_mean_shift():
    for name, (detector, clipped) in _detectors(_shift_stream()).items():
        events = detect_stream(clipped, detector)
        assert events, f"{name} missed a +1.5 sigma shift"
        first = events[0]["detected_at"]
        assert 1000 <= first <= 1200, f"{name} alarmed at {first}, drift starts at 1000"


def test_page_hinkley_and_adwin_catch_a_downward_shift_too():
    stream = _shift_stream(jump=-1.5)
    for name, (detector, clipped) in _detectors(stream).items():
        events = detect_stream(clipped, detector)
        assert events and 1000 <= events[0]["detected_at"] <= 1200, name


def test_page_hinkley_reports_the_direction_of_the_change():
    detector = PageHinkley()
    detect_stream(_shift_stream(), detector)
    assert detector.last_alarm["direction"] == "up"
    detector = PageHinkley()
    detect_stream(_shift_stream(jump=-1.5), detector)
    assert detector.last_alarm["direction"] == "down"


@pytest.mark.parametrize("seed", [42, 7])
def test_page_hinkley_and_adwin_stay_quiet_on_a_stationary_stream(seed):
    for name, (detector, clipped) in _detectors(_normal(seed, 2000)).items():
        assert detect_stream(clipped, detector) == [], f"{name} false alarm (seed {seed})"


def test_windowed_detectors_fire_on_shift_and_stay_quiet_without_one():
    for statistic in ("psi", "ks", "js"):
        drifting = detect_stream(_shift_stream(), WindowedDetector(statistic, window=200))
        assert drifting and drifting[0]["detected_at"] >= 1000, statistic
        assert detect_stream(_normal(42, 2000), WindowedDetector(statistic, window=200)) == [], statistic


def test_a_single_outlier_does_not_raise_a_page_hinkley_alarm():
    stream = _normal(3, 600)
    stream[300] = 1e6
    clipped, _ = clip_to_fences(stream)
    assert detect_stream(clipped, PageHinkley()) == []


# ---------------------------------------------------------------- find_change_points

def _dated_frame(n: int = 120, change_at: int = 60) -> tuple[pd.DataFrame, dict]:
    """Daily rows, as parse_csv_bytes would hand them over: every cell a string."""
    noise = np.random.default_rng(5).normal(0.0, 1.0, n)
    value = 100.0 + 5.0 * noise
    value[change_at:] += 60.0
    frame = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "value": [f"{v:.3f}" for v in value],
        "region": ["north", "south"] * (n // 2),
    })
    return frame, {"date": "datetime", "value": "numeric", "region": "categorical"}


def test_find_change_points_finds_the_planted_change():
    frame, types = _dated_frame()
    result = find_change_points(frame, types)
    assert result["skipped"] is False
    assert result["time_column"] == "date"
    assert result["rows_used"] == 120

    value = next(c for c in result["columns"] if c["column"] == "value")
    assert value["drifted"] is True
    region = next(c for c in result["columns"] if c["column"] == "region")
    assert region["drifted"] is False

    assert result["change_points"], "planted +12 sigma change not found"
    point = result["change_points"][0]
    assert point["column"] == "value"
    assert point["direction"] == "up"
    # Planted at row 60 = 2024-03-01, header is line 1 so the file line is 62.
    assert abs(point["change_line"] - 62) <= 5
    assert "2024-02-2" in point["change_at"] or "2024-03-0" in point["change_at"]
    assert point["reason"].startswith("value: mean rose")


def test_find_change_points_orders_rows_by_the_datetime_column():
    frame, types = _dated_frame()
    shuffled = frame.sample(frac=1.0, random_state=0).reset_index(drop=True)
    ordered = find_change_points(frame, types)
    result = find_change_points(shuffled, types)
    assert result["period"] == ordered["period"]
    assert result["change_points"]
    assert result["change_points"][0]["change_at"] == ordered["change_points"][0]["change_at"]


def test_find_change_points_skips_cleanly_without_a_datetime_column():
    frame, types = _dated_frame()
    frame, types = frame.drop(columns="date"), {k: v for k, v in types.items() if k != "date"}
    result = find_change_points(frame, types)
    assert result == {"skipped": True, "reason": "no datetime column in this file",
                      "change_points": []}


def test_find_change_points_skips_when_there_are_too_few_dated_rows():
    frame, types = _dated_frame(n=20, change_at=10)
    result = find_change_points(frame, types)
    assert result["skipped"] is True
    assert result["change_points"] == []
    assert "at least" in result["reason"]


def test_find_change_points_skips_constant_numeric_columns_without_crashing():
    frame, types = _dated_frame()
    frame["flat"] = "7"
    types["flat"] = "numeric"
    result = find_change_points(frame, types)
    flat = next(c for c in result["columns"] if c["column"] == "flat")
    assert flat["skipped"] is True
    assert flat["reason"] == "every value is the same"


# ---------------------------------------------------------------- determinism

def test_batch_statistics_are_deterministic():
    a, b = _normal(1, 800), _normal(2, 800, mean=0.4)
    assert compare_numeric(a, b) == compare_numeric(a.copy(), b.copy())
    cats = (["x", "y", "z"] * 100, ["x", "y", "y"] * 100)
    assert compare_categorical(*cats) == compare_categorical(*cats)


def test_streaming_detectors_are_deterministic():
    stream = _shift_stream()
    runs = [
        detect_stream(stream, factory())
        for factory in (PageHinkley, lambda: ADWIN(value_range=8.0),
                        lambda: WindowedDetector("psi", window=200))
        for _ in range(2)
    ]
    assert runs[0] == runs[1] and runs[2] == runs[3] and runs[4] == runs[5]
    assert all(runs[i] for i in (0, 2, 4))


def test_find_change_points_is_deterministic():
    frame, types = _dated_frame()
    assert find_change_points(frame, types) == find_change_points(frame.copy(), dict(types))
