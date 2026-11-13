"""Step E — drift engine, built by hand (BLUEPRINT §12 T1, sprint Phase A3).

Two questions, two families of detector, and no drift library anywhere:

Batch — "do these two samples come from the same distribution?"
- PSI (Population Stability Index): bucket both samples on the reference's
  quantiles, then sum (actual - expected) * ln(actual / expected).
- Kolmogorov-Smirnov: the largest gap between the two empirical CDFs, with
  its asymptotic p-value.
- Jensen-Shannon divergence: a symmetric, bounded (0 to 1 bit) relative of
  KL divergence, for numeric buckets and for category labels.

Streaming — "at which row did this stream change?"
- Page-Hinkley: a running sum of how far each value sits from the regime's
  mean; it alarms when the sum climbs `threshold` above its lowest point.
- ADWIN: an adaptive window that drops its oldest rows whenever two of its
  sub-windows have means too far apart to be chance.
- WindowedDetector: turns a batch statistic into a stream detector by
  comparing the latest window with a reference window.

`find_change_points` is the product entry point. With a datetime column it
puts the rows in time order, compares the first half of the period with the
second half for every numeric and categorical column, and runs both
streaming detectors over every numeric column. Without a datetime column
the section is skipped, never faked.

Nothing in this module is random: the same input always gives the same
output, so there is no seed to pass.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Protocol

import numpy as np
import pandas as pd

from .facts import mask

DEFAULT_BINS = 10
# PSI needs enough rows per bucket. Two samples of ONE distribution still
# score about (buckets - 1) * (1/n + 1/m), so with too few rows per bucket
# sampling noise alone crosses the "significant" line.
MIN_ROWS_PER_BIN = 25
# A bucket nobody landed in would make ln(actual / expected) infinite, so
# shares are floored at this before the PSI sum.
PSI_FLOOR = 1e-4
# The usual reading of PSI: under 0.1 stable, 0.1-0.25 moderate, above that
# a significant shift.
PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25
# JS has no such convention. For small shifts PSI ~= 8 ln(2) * JS (JS in
# bits), so these are the two PSI lines translated and rounded.
JS_MODERATE = 0.02
JS_SIGNIFICANT = 0.05
# Significance level for a one-off comparison of two halves. Strict because
# a file has many columns and every one of them is tested.
ALPHA = 0.001
# A monitor repeats its test every few rows, so each single test must be far
# stricter to keep the chance of any false alarm on a long stream low.
MONITOR_ALPHA = 1e-5
# Category labels seen fewer times than this (both samples together) are
# pooled into one "(other)" bucket before the categorical comparison.
MIN_CATEGORY_ROWS = 10
BATCH_STATISTICS = ("psi", "ks", "js")

# Product limits for find_change_points.
MIN_ROWS = 60  # two halves of 30: fewer rows can only show noise
MAX_CHANGE_POINTS = 20
# On long files ADWIN looks for a cut every few rows instead of every row,
# so its cost per column stays bounded (about this many scans).
ADWIN_MAX_CHECKS = 2000
DETECTOR_LABELS = {"page_hinkley": "Page-Hinkley", "adwin": "ADWIN"}


class DriftInputError(ValueError):
    """A drift function was handed something it cannot work with; the message says what to fix."""


# --------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------

def _numbers(values, name: str) -> np.ndarray:
    """`values` as a 1-D float64 array; DriftInputError if it holds anything but numbers."""
    try:
        return np.asarray(values, dtype=np.float64).ravel()
    except (TypeError, ValueError) as exc:
        raise DriftInputError(
            f"{name} must hold numbers only - convert or drop text values first") from exc


def _sample(values, name: str) -> np.ndarray:
    """The finite numbers in `values` (missing values dropped); DriftInputError if none are left."""
    numbers = _numbers(values, name)
    numbers = numbers[np.isfinite(numbers)]
    if numbers.size == 0:
        raise DriftInputError(f"{name} has no numeric values to compare")
    return numbers


def _stream(values) -> np.ndarray:
    """A stream as a 1-D float64 array. May be empty; must not hold missing values."""
    numbers = _numbers(values, "the stream")
    if not np.isfinite(numbers).all():
        raise DriftInputError("the stream holds missing or infinite values - drop them first")
    return numbers


def _finite(value, owner: str) -> float:
    """One stream value as a float; DriftInputError unless it is a finite number."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DriftInputError(f"{owner} needs a number, got {value!r}") from exc
    if not math.isfinite(number):
        raise DriftInputError(
            f"{owner} needs a finite number, got {value!r} - drop missing values first")
    return number


def _size(n, name: str) -> int:
    if n < 1:
        raise DriftInputError(f"{name} must be at least 1, got {n}")
    return int(n)


def _num(value: float) -> float:
    """Plain Python float at 6 significant digits: JSON-safe and stable to print."""
    return float(f"{float(value):.6g}")


# --------------------------------------------------------------------------
# Distributions: PSI and Jensen-Shannon on two aligned sets of bucket shares
# --------------------------------------------------------------------------

def _shares(counts, name: str) -> np.ndarray:
    """Counts or proportions per bucket -> proportions that sum to 1."""
    try:
        values = np.asarray(counts, dtype=np.float64).ravel()
    except (TypeError, ValueError) as exc:
        raise DriftInputError(f"{name} must be a list of counts or proportions") from exc
    if values.size == 0 or not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
        raise DriftInputError(f"{name} must be non-empty, non-negative and not all zero")
    return values / values.sum()


def _aligned(first, second, first_name: str, second_name: str) -> tuple[np.ndarray, np.ndarray]:
    p, q = _shares(first, first_name), _shares(second, second_name)
    if p.size != q.size:
        raise DriftInputError(f"{first_name} and {second_name} must describe the same buckets; "
                              f"got {p.size} and {q.size}")
    return p, q


def psi_from_distributions(expected, actual) -> float:
    """PSI between two aligned bucket distributions.

    Inputs: counts or proportions per bucket, same buckets in the same order
    (expected = reference, actual = current). Output: the sum over buckets of
    (actual - expected) * ln(actual / expected); 0 means identical, and every
    term is >= 0 so it only grows as the two drift apart.
    """
    e, a = _aligned(expected, actual, "expected", "actual")
    e, a = np.maximum(e, PSI_FLOOR), np.maximum(a, PSI_FLOOR)
    return float(np.sum((a - e) * np.log(a / e)))


def _kl_bits(p: np.ndarray, m: np.ndarray) -> float:
    held = p > 0  # an empty bucket adds nothing: 0 * log(0) counts as 0
    return float(np.sum(p[held] * np.log2(p[held] / m[held])))


def js_from_distributions(p, q) -> float:
    """Jensen-Shannon divergence, in bits, between two aligned bucket distributions.

    Inputs: counts or proportions per bucket. Output: (KL(p||m) + KL(q||m)) / 2
    with m = (p + q) / 2. Symmetric; 0 for identical distributions, 1 when
    they share no bucket at all.
    """
    p, q = _aligned(p, q, "p", "q")
    m = (p + q) / 2.0
    return float(min(max((_kl_bits(p, m) + _kl_bits(q, m)) / 2.0, 0.0), 1.0))


# --------------------------------------------------------------------------
# Batch comparison of two numeric samples
# --------------------------------------------------------------------------

def bucket_shares(reference, current, bins: int = DEFAULT_BINS) -> tuple[np.ndarray, np.ndarray]:
    """Put two numeric samples into the same buckets; return each one's share per bucket.

    Inputs: two numeric samples (missing values are dropped) and the number
    of quantile buckets. Output: (reference shares, current shares).
    The buckets come from the reference alone, as PSI prescribes:
    - more than `bins` distinct reference values: `bins` quantile buckets, so
      each holds about the same share of the reference. The two outer ones
      are open-ended, so values beyond the reference's range still count.
    - `bins` or fewer distinct values (a rating, a flag, a constant): every
      distinct value is its own bucket and so is every gap between them.
      Quantiles mean nothing there, and a value the reference never had
      must not hide inside a neighbour's bucket.
    """
    if not isinstance(bins, (int, np.integer)) or bins < 2:
        raise DriftInputError(f"bins must be a whole number of at least 2, got {bins!r}")
    ref, cur = _sample(reference, "reference"), _sample(current, "current")
    distinct = np.unique(ref)
    if distinct.size <= bins:
        # Left + right insertion points: odd = exactly a reference value,
        # even = a gap between two of them (or beyond either end).
        def bucket_of(x: np.ndarray) -> np.ndarray:
            return (np.searchsorted(distinct, x, side="left")
                    + np.searchsorted(distinct, x, side="right"))
        buckets = 2 * distinct.size + 1
    else:
        cuts = np.unique(np.quantile(ref, np.arange(1, bins) / bins))

        def bucket_of(x: np.ndarray) -> np.ndarray:
            return np.searchsorted(cuts, x, side="left")
        buckets = cuts.size + 1
    return (np.bincount(bucket_of(ref), minlength=buckets) / ref.size,
            np.bincount(bucket_of(cur), minlength=buckets) / cur.size)


def suggest_bins(n_reference: int, n_current: int) -> int:
    """How many PSI buckets the smaller of two samples can support (2 to DEFAULT_BINS)."""
    return int(max(2, min(DEFAULT_BINS, min(n_reference, n_current) // MIN_ROWS_PER_BIN)))


def psi(reference, current, bins: int = DEFAULT_BINS) -> float:
    """Population Stability Index of `current` measured against `reference`.

    Inputs: two numeric samples and the number of quantile buckets (use
    `suggest_bins` for small samples). Output: PSI >= 0, 0 = same distribution.
    """
    return psi_from_distributions(*bucket_shares(reference, current, bins))


def js_divergence(reference, current, bins: int = DEFAULT_BINS) -> float:
    """Jensen-Shannon divergence (bits, 0 to 1) between two numeric samples.

    Uses the same buckets as `psi`, so the two numbers describe one histogram pair.
    """
    return js_from_distributions(*bucket_shares(reference, current, bins))


def ks_statistic(reference, current) -> float:
    """Two-sample Kolmogorov-Smirnov statistic D.

    Inputs: two numeric samples. Output: the largest vertical gap between
    their empirical CDFs, from 0 (identical) to 1 (no overlap).
    """
    ref = np.sort(_sample(reference, "reference"))
    cur = np.sort(_sample(current, "current"))
    # Both CDFs are step functions that only move at observed values, so the
    # largest gap has to sit at one of those values.
    points = np.concatenate([ref, cur])
    cdf_ref = np.searchsorted(ref, points, side="right") / ref.size
    cdf_cur = np.searchsorted(cur, points, side="right") / cur.size
    return float(np.max(np.abs(cdf_ref - cdf_cur)))


def kolmogorov_sf(x: float) -> float:
    """P(K > x) for the Kolmogorov distribution: 2 * sum_{k>=1} (-1)^(k-1) * exp(-2 k^2 x^2)."""
    if x < 0.2:
        # The series converges too slowly to sum here, and the true value is
        # within 1e-12 of 1 anyway.
        return 1.0
    total = 0.0
    for k in range(1, 101):
        term = math.exp(-2.0 * k * k * x * x)
        total += term if k % 2 else -term
        if term < 1e-17:
            break
    return float(min(max(2.0 * total, 0.0), 1.0))


def _ks_scale(n_reference: int, n_current: int) -> float:
    """What a KS statistic is multiplied by before the Kolmogorov distribution applies.

    sqrt(n*m / (n+m)) is the asymptotic scaling; the two extra terms are
    Stephens' correction, which keeps the p-value honest for small samples.
    """
    n, m = _size(n_reference, "n_reference"), _size(n_current, "n_current")
    root = math.sqrt(n * m / (n + m))
    return root + 0.12 + 0.11 / root


def ks_p_value(statistic: float, n_reference: int, n_current: int) -> float:
    """Asymptotic two-sided p-value of a KS statistic for samples of the given sizes.

    Output: the chance of a gap at least this large if both samples shared
    one distribution. Small = drift. Conservative when the data has ties.
    """
    if not 0.0 <= statistic <= 1.0:
        raise DriftInputError(f"a KS statistic lies between 0 and 1, got {statistic!r}")
    return kolmogorov_sf(_ks_scale(n_reference, n_current) * statistic)


def ks_critical_value(alpha: float, n_reference: int, n_current: int) -> float:
    """Smallest KS statistic whose p-value is at most `alpha` (for alpha <= 0.1).

    Inverts the first term of the Kolmogorov series; the later terms are
    below alpha^4 at these levels.
    """
    if not 0.0 < alpha <= 0.1:
        raise DriftInputError(f"alpha must be above 0 and at most 0.1, got {alpha!r}")
    return min(1.0, math.sqrt(-0.5 * math.log(alpha / 2.0)) / _ks_scale(n_reference, n_current))


def ks_test(reference, current) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test. Returns (statistic, p_value)."""
    ref, cur = _sample(reference, "reference"), _sample(current, "current")
    statistic = ks_statistic(ref, cur)
    return statistic, ks_p_value(statistic, ref.size, cur.size)


# --------------------------------------------------------------------------
# Batch comparison of two categorical samples
# --------------------------------------------------------------------------

def _labels(values, name: str) -> list[str]:
    labels = [str(value).strip() for value in values if not pd.isna(value)]
    labels = [label for label in labels if label]
    if not labels:
        raise DriftInputError(f"{name} has no category labels to compare")
    return labels


def _category_counts(reference, current) -> tuple[list[str], np.ndarray, np.ndarray]:
    ref, cur = _labels(reference, "reference"), _labels(current, "current")
    labels = sorted(set(ref) | set(cur))
    position = {label: i for i, label in enumerate(labels)}

    def count(sample: list[str]) -> np.ndarray:
        return np.bincount([position[label] for label in sample], minlength=len(labels))
    return labels, count(ref), count(cur)


def category_shares(reference, current) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Share of every label in two samples of category labels.

    Inputs: two sequences of labels (blanks and missing values are dropped).
    Output: (sorted labels, reference shares, current shares) over the union
    of the labels, so a label seen in only one sample still has a bucket.
    """
    labels, ref, cur = _category_counts(reference, current)
    return labels, ref / ref.sum(), cur / cur.sum()


def js_divergence_categorical(reference, current) -> float:
    """Jensen-Shannon divergence (bits, 0 to 1) between the label mixes of two samples."""
    _, p, q = category_shares(reference, current)
    return js_from_distributions(p, q)


def chi_square_sf(statistic: float, dof: int) -> float:
    """P(X >= statistic) for a chi-square variable with `dof` degrees of freedom.

    Exact, by the recurrence Q(a + 1, y) = Q(a, y) + y^a e^-y / Gamma(a + 1)
    with y = statistic / 2, started from Q(1/2, y) = erfc(sqrt(y)) for odd
    `dof` or Q(1, y) = e^-y for even `dof`.
    """
    if not isinstance(dof, (int, np.integer)) or dof < 1:
        raise DriftInputError(f"dof must be a whole number of at least 1, got {dof!r}")
    if statistic <= 0:
        return 1.0
    y = statistic / 2.0
    a, q = (0.5, math.erfc(math.sqrt(y))) if dof % 2 else (1.0, math.exp(-y))
    while a < dof / 2.0:
        q += math.exp(a * math.log(y) - y - math.lgamma(a + 1.0))
        a += 1.0
    return float(min(q, 1.0))


def js_p_value(js: float, n_reference: int, n_current: int, buckets: int) -> float:
    """Approximate p-value of a JS divergence (bits) between samples of the given sizes.

    When both samples share one distribution over `buckets` buckets,
    8 ln(2) * N * JS with N = n*m / (n+m) is approximately chi-square with
    buckets - 1 degrees of freedom: JS is the likelihood-ratio statistic of
    the two-sample table in disguise. Small = the label mix really moved.
    """
    n, m = _size(n_reference, "n_reference"), _size(n_current, "n_current")
    if buckets < 2:
        return 1.0
    return chi_square_sf(8.0 * math.log(2.0) * (n * m / (n + m)) * js, int(buckets) - 1)


# --------------------------------------------------------------------------
# Verdicts: all statistics for one pair of samples
# --------------------------------------------------------------------------

def _level(p_value: float, effect: float, moderate: float, significant: float) -> str:
    """"stable" unless the difference is beyond chance AND large enough to matter."""
    if p_value >= ALPHA or effect < moderate:
        return "stable"
    return "moderate" if effect < significant else "significant"


def compare_numeric(reference, current, bins: int | None = None) -> dict:
    """Compare two numeric samples with all three batch statistics.

    Inputs: two numeric samples; `bins` defaults to `suggest_bins`.
    Output: JSON-safe dict with psi, ks_statistic, ks_p_value, js_divergence,
    the bucket count, both sample sizes, `level` ("stable", "moderate" or
    "significant") and `drifted`. A pair only leaves "stable" when the KS
    test rules out chance (p < ALPHA); PSI then grades how big the shift is.
    """
    ref, cur = _sample(reference, "reference"), _sample(current, "current")
    bins = suggest_bins(ref.size, cur.size) if bins is None else bins
    shares = bucket_shares(ref, cur, bins)
    psi_value = psi_from_distributions(*shares)
    statistic, p_value = ks_test(ref, cur)
    level = _level(p_value, psi_value, PSI_MODERATE, PSI_SIGNIFICANT)
    return {
        "n_reference": int(ref.size),
        "n_current": int(cur.size),
        "bins": int(bins),
        "psi": _num(psi_value),
        "ks_statistic": _num(statistic),
        "ks_p_value": _num(p_value),
        "js_divergence": _num(js_from_distributions(*shares)),
        "level": level,
        "drifted": level != "stable",
    }


def compare_categorical(reference, current) -> dict:
    """Compare the label mix of two categorical samples.

    Inputs: two sequences of labels. Output: JSON-safe dict with
    js_divergence (bits), its approximate p_value, the number of buckets,
    both sample sizes, `level`, `drifted` and the labels whose share moved
    most. Labels seen fewer than MIN_CATEGORY_ROWS times in total are pooled
    into "(other)" first: the p-value needs enough rows per bucket, and a
    verdict should not hinge on one-off labels.
    """
    labels, ref, cur = _category_counts(reference, current)
    rare = (ref + cur) < MIN_CATEGORY_ROWS
    if rare.sum() > 1:
        labels = [label for label, is_rare in zip(labels, rare) if not is_rare] + ["(other)"]
        ref = np.append(ref[~rare], ref[rare].sum())
        cur = np.append(cur[~rare], cur[rare].sum())
    n_ref, n_cur = int(ref.sum()), int(cur.sum())
    p, q = ref / n_ref, cur / n_cur
    js = js_from_distributions(p, q)
    p_value = js_p_value(js, n_ref, n_cur, len(labels))
    level = _level(p_value, js, JS_MODERATE, JS_SIGNIFICANT)
    movers = np.argsort(-np.abs(q - p), kind="stable")[:3]
    return {
        "n_reference": n_ref,
        "n_current": n_cur,
        "categories": len(labels),
        "js_divergence": _num(js),
        "p_value": _num(p_value),
        "level": level,
        "drifted": level != "stable",
        # Labels are data, and this dict travels into the Facts Pack: mask them.
        "biggest_shifts": [{"label": mask(labels[i]),
                            "reference_share": round(float(p[i]), 4),
                            "current_share": round(float(q[i]), 4)} for i in movers],
    }


# --------------------------------------------------------------------------
# Streaming detectors
# --------------------------------------------------------------------------

class StreamDetector(Protocol):
    """What `detect_stream` needs: feed one value, learn whether it raised an alarm."""

    last_alarm: dict | None

    def update(self, value: float) -> bool: ...


class PageHinkley:
    """Page-Hinkley change detector for the mean of a stream: two-sided and scale-free.

    Classic Page-Hinkley keeps m_t = sum(x_i - mean_i - delta) and alarms when
    m_t rises `threshold` above its running minimum (an upward shift); the
    mirror image catches a downward shift. Here each deviation is first
    divided by the regime's running standard deviation, so `delta` and
    `threshold` are in standard deviations and need no tuning per column:

    - delta: shifts smaller than this many standard deviations are tolerated;
    - threshold: how much evidence must pile up before the alarm;
    - min_samples: values used to learn a regime's mean and spread before
      testing starts;
    - clip: the most one value may count for, so a single outlier never
      raises the alarm - a change has to persist.

    After an alarm the detector forgets everything and learns the new regime.
    `update` returns True on the value that raises the alarm; `last_alarm`
    then holds the statistic, the threshold, the direction and
    `samples_since_change`: how many of the latest values lie after the
    turning point of the sum, i.e. where the detector thinks the change began.
    """

    def __init__(self, delta: float = 0.5, threshold: float = 20.0,
                 min_samples: int = 30, clip: float = 4.0) -> None:
        if delta < 0 or threshold <= 0 or min_samples < 2 or clip <= delta:
            raise DriftInputError("PageHinkley needs delta >= 0, threshold > 0, "
                                  "min_samples >= 2 and clip > delta")
        self.delta = float(delta)
        self.threshold = float(threshold)
        self.min_samples = int(min_samples)
        self.clip = float(clip)
        self.n_seen = 0
        self.last_alarm: dict | None = None
        self.reset()

    def reset(self) -> None:
        """Forget the current regime (done automatically after an alarm)."""
        self._n = 0
        self._mean = 0.0
        self._sq_dev = 0.0
        self._up = self._up_min = 0.0
        self._down = self._down_max = 0.0
        self._since_up = self._since_down = 0

    def _learn(self, x: float) -> None:
        # Welford's running mean and sum of squared deviations.
        self._n += 1
        step = x - self._mean
        self._mean += step / self._n
        self._sq_dev += step * (x - self._mean)

    def update(self, value: float) -> bool:
        """Feed one value. Returns True if it raises the alarm."""
        x = _finite(value, "PageHinkley.update")
        self.n_seen += 1
        if self._n < self.min_samples:
            self._learn(x)
            return False
        spread = math.sqrt(self._sq_dev / (self._n - 1))
        gap = x - self._mean
        if spread > 0.0:
            z = max(-self.clip, min(self.clip, gap / spread))
        else:
            # A perfectly flat regime: any other value is as surprising as a
            # value is allowed to be.
            z = 0.0 if gap == 0.0 else math.copysign(self.clip, gap)
        # The regime learns the clipped value, so one absurd row cannot
        # inflate the spread and blind the test from then on.
        self._learn(self._mean + z * spread)

        self._up += z - self.delta
        if self._up < self._up_min:
            self._up_min, self._since_up = self._up, 0
        else:
            self._since_up += 1
        self._down += z + self.delta
        if self._down > self._down_max:
            self._down_max, self._since_down = self._down, 0
        else:
            self._since_down += 1

        rise = self._up - self._up_min
        fall = self._down_max - self._down
        if max(rise, fall) <= self.threshold:
            return False
        upward = rise >= fall
        self.last_alarm = {
            "statistic": rise if upward else fall,
            "threshold": self.threshold,
            "direction": "up" if upward else "down",
            "samples_since_change": self._since_up if upward else self._since_down,
        }
        self.reset()
        return True


class ADWIN:
    """ADaptive WINdowing (Bifet & Gavalda, 2007) for the mean of a bounded stream.

    Keeps a window of the most recent values. After a new value it tries
    every split of the window into an older and a newer part; if the two
    means differ by more than `epsilon` - the largest gap chance allows at
    confidence `delta` - the oldest rows are dropped and a change is
    reported. So the window grows while the stream is stable and shrinks
    when it is not.

    The window is stored as an exponential histogram: buckets summarising
    1, 2, 4, ... values, at most `max_buckets` of each size, newest smallest.
    Memory and the number of splits tried stay O(log window).

        epsilon = sqrt(2 * variance * L / m) + (2/3) * value_range * L / m
        m = 1 / (1/n0 + 1/n1)      L = ln(2 * ln(n) / delta)

    The first term is the normal approximation; the second (Bernstein) term
    protects small sub-windows, where that approximation fails. delta is
    shared between the ~ln(n) splits that are actually tested.

    - delta: confidence; smaller = fewer false alarms, slower detection;
    - value_range: width of the interval the values lie in. The bound needs
      bounded values - clip the stream first (see `clip_to_fences`);
    - clock: look for a cut only every `clock` values (1 = every value).

    `update` returns True when the window shrank; `last_alarm` then holds the
    gap between the two means, the epsilon it beat and
    `samples_since_change`: the size of the newer part at the most
    significant split, i.e. where the detector thinks the change began.
    """

    def __init__(self, delta: float = 0.002, value_range: float = 1.0,
                 max_buckets: int = 5, clock: int = 1) -> None:
        if not 0.0 < delta < 1.0 or value_range <= 0 or max_buckets < 2 or clock < 1:
            raise DriftInputError("ADWIN needs 0 < delta < 1, value_range > 0, "
                                  "max_buckets >= 2 and clock >= 1")
        self.delta = float(delta)
        self.value_range = float(value_range)
        self.max_buckets = int(max_buckets)
        self.clock = int(clock)
        self.n_seen = 0
        self.last_alarm: dict | None = None
        self.reset()

    def reset(self) -> None:
        """Empty the window."""
        # _rows[i] holds buckets of 2**i values as (count, sum, sum of
        # squares), oldest on the left.
        self._rows: list[deque] = [deque()]
        self.width = 0
        self._origin = 0.0
        self._total = 0.0
        self._total_sq = 0.0

    @property
    def mean(self) -> float:
        """Mean of the values currently in the window (0.0 while it is empty)."""
        return self._origin + self._total / self.width if self.width else 0.0

    @property
    def bucket_count(self) -> int:
        """How many buckets currently summarise the window."""
        return sum(len(row) for row in self._rows)

    def update(self, value: float) -> bool:
        """Feed one value. Returns True if the window shrank (a change was detected)."""
        x = _finite(value, "ADWIN.update")
        self.n_seen += 1
        if self.width == 0:
            # Values are stored relative to the first one in the window, so
            # the variance (mean of squares minus squared mean) keeps its
            # precision on columns like 1,000,000 +- 3.
            self._origin = x
        self._insert(x - self._origin)
        if self.n_seen % self.clock:
            return False
        cut = self._most_significant_cut()
        if cut is None:
            return False
        self.last_alarm = cut
        while cut is not None:  # shrink until what is left no longer looks like two regimes
            self._drop_oldest_bucket()
            cut = self._most_significant_cut()
        return True

    def _insert(self, x: float) -> None:
        self._rows[0].append((1, x, x * x))
        self.width += 1
        self._total += x
        self._total_sq += x * x
        level = 0
        while len(self._rows[level]) > self.max_buckets:
            # Too many buckets of this size: merge the two oldest into one
            # bucket of twice the size, a level up.
            n1, s1, q1 = self._rows[level].popleft()
            n2, s2, q2 = self._rows[level].popleft()
            if level + 1 == len(self._rows):
                self._rows.append(deque())
            self._rows[level + 1].append((n1 + n2, s1 + s2, q1 + q2))
            level += 1

    def _drop_oldest_bucket(self) -> None:
        count, total, total_sq = self._rows[-1].popleft()
        self.width -= count
        self._total -= total
        self._total_sq -= total_sq
        if not self._rows[-1] and len(self._rows) > 1:
            self._rows.pop()

    def _most_significant_cut(self) -> dict | None:
        """The split whose two means differ most relative to epsilon, or None if none exceeds it."""
        n = self.width
        if n < 2:
            return None
        variance = max(self._total_sq / n - (self._total / n) ** 2, 0.0)
        log_term = math.log(2.0 * math.log(n) / self.delta)
        normal_part = 2.0 * variance * log_term
        range_part = 2.0 * self.value_range * log_term / 3.0
        best, best_ratio = None, 1.0
        n0, sum0 = 0, 0.0
        for row in reversed(self._rows):  # oldest buckets first
            for count, total, _ in row:
                n0 += count
                sum0 += total
                n1 = n - n0
                if n1 == 0:
                    break
                m = n0 * n1 / n
                epsilon = math.sqrt(normal_part / m) + range_part / m
                gap = abs(sum0 / n0 - (self._total - sum0) / n1)
                if gap > epsilon * best_ratio:
                    best_ratio = gap / epsilon
                    best = {"statistic": gap, "threshold": epsilon, "samples_since_change": n1}
        return best


class WindowedDetector:
    """Turn a batch statistic into a stream detector.

    The first `window` values after a (re)start are the reference. Once
    `window` further values have arrived, the latest `window` values are
    compared with the reference every `step` values. A statistic at or above
    `threshold` raises the alarm and restarts the detector, so its next
    reference is learned from the new regime.

    - statistic: "psi", "ks" or "js";
    - threshold: defaults to PSI_SIGNIFICANT, JS_SIGNIFICANT, or for "ks" the
      critical value at MONITOR_ALPHA for two windows of this size.

    A window test says that the latest window differs, not where inside it
    the change began, so `last_alarm["samples_since_change"]` is None.
    """

    def __init__(self, statistic: str, window: int = 500, step: int = 10,
                 threshold: float | None = None, bins: int = DEFAULT_BINS) -> None:
        if statistic not in BATCH_STATISTICS:
            raise DriftInputError(f"unknown statistic {statistic!r}; "
                                  f"choose one of {', '.join(BATCH_STATISTICS)}")
        if window < 2 or step < 1:
            raise DriftInputError("WindowedDetector needs window >= 2 and step >= 1")
        if threshold is None:
            threshold = {"psi": PSI_SIGNIFICANT, "js": JS_SIGNIFICANT,
                         "ks": ks_critical_value(MONITOR_ALPHA, window, window)}[statistic]
        self.statistic = statistic
        self.window = int(window)
        self.step = int(step)
        self.threshold = float(threshold)
        self.bins = bins
        self.n_seen = 0
        self.last_alarm: dict | None = None
        self.reset()

    def reset(self) -> None:
        """Forget the reference and the recent window (done automatically after an alarm)."""
        self._reference: list[float] = []
        self._recent: deque = deque(maxlen=self.window)
        self._count = 0

    def update(self, value: float) -> bool:
        """Feed one value. Returns True if it raises the alarm."""
        x = _finite(value, "WindowedDetector.update")
        self.n_seen += 1
        self._count += 1
        if self._count <= self.window:
            self._reference.append(x)
            return False
        self._recent.append(x)
        waited = self._count - 2 * self.window  # values since the two windows stopped overlapping
        if waited < 0 or waited % self.step:
            return False
        measured = self._measure()
        if measured < self.threshold:
            return False
        self.last_alarm = {"statistic": measured, "threshold": self.threshold,
                           "samples_since_change": None}
        self.reset()
        return True

    def _measure(self) -> float:
        reference = np.asarray(self._reference, dtype=np.float64)
        recent = np.fromiter(self._recent, dtype=np.float64, count=len(self._recent))
        if self.statistic == "ks":
            return ks_statistic(reference, recent)
        shares = bucket_shares(reference, recent, self.bins)
        if self.statistic == "psi":
            return psi_from_distributions(*shares)
        return js_from_distributions(*shares)


# --------------------------------------------------------------------------
# Running a detector over a stream
# --------------------------------------------------------------------------

def tukey_fences(values, k: float = 1.5) -> tuple[float, float]:
    """Box-plot outlier fences (Q1 - k*IQR, Q3 + k*IQR), never wider than the data itself.

    Input: a numeric sample. If its middle half is a single value (IQR = 0)
    the fences would collapse onto it, so the data's own min and max are
    returned instead.
    """
    sample = _sample(values, "values")
    q1, q3 = (float(q) for q in np.percentile(sample, [25, 75]))
    low, high = float(sample.min()), float(sample.max())
    if q3 > q1:
        low, high = max(low, q1 - k * (q3 - q1)), min(high, q3 + k * (q3 - q1))
    return low, high


def clip_to_fences(values, reference=None) -> tuple[np.ndarray, float]:
    """Clip a stream to the Tukey fences of `reference` (default: the stream itself).

    Output: (clipped values, fence width). One absurd value is an anomaly,
    not a drift, so it must not move a drift detector; and ADWIN has to know
    the range its input lives in - the fence width is that `value_range`
    (0.0 when the reference is constant).
    """
    stream = _stream(values)
    low, high = tukey_fences(stream if reference is None else reference)
    return np.clip(stream, low, high), high - low


def detect_stream(values, detector: StreamDetector) -> list[dict]:
    """Feed `values` to a stream detector in order; return one event per alarm.

    Inputs: a 1-D sequence of finite numbers and a PageHinkley, ADWIN or
    WindowedDetector. Output: a list of dicts with
    - detected_at: index of the value that raised the alarm;
    - change_at: the detector's estimate of the first changed index (None if
      it has none);
    - statistic, threshold: what was measured and the line it crossed.
    """
    events: list[dict] = []
    last_reported = -1
    for index, value in enumerate(_stream(values).tolist()):
        if not detector.update(value):
            continue
        alarm = detector.last_alarm
        since = alarm["samples_since_change"]
        change_at = None if since is None else max(0, index - since + 1)
        # ADWIN keeps trimming its window while more post-change rows
        # arrive. A cut that places the change before the alarm already
        # reported is that same change again, not a new one.
        if change_at is not None and change_at <= last_reported:
            continue
        events.append({"detected_at": index, "change_at": change_at,
                       "statistic": float(alarm["statistic"]),
                       "threshold": float(alarm["threshold"])})
        last_reported = index
    return events


# --------------------------------------------------------------------------
# Product entry point
# --------------------------------------------------------------------------

def _skipped(reason: str) -> dict:
    return {"skipped": True, "reason": reason, "change_points": []}


def _describe(values: np.ndarray) -> dict:
    return {"rows": int(values.size), "mean": _num(values.mean()), "std": _num(values.std())}


def _fmt(value: float) -> str:
    return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.4g}"


def _change_point(column: str, detector: str, alarm: dict, before: np.ndarray, after: np.ndarray,
                  times: list[str], lines: np.ndarray, kept: np.ndarray) -> dict:
    """One alarm as a report row: where, when, how big, and a sentence a person can read."""
    start, seen = kept[alarm["change_at"]], kept[alarm["detected_at"]]
    old, new = float(before.mean()), float(after.mean())
    if new > old:
        direction, moved = "up", f"mean rose from {_fmt(old)} to {_fmt(new)}"
    elif new < old:
        direction, moved = "down", f"mean fell from {_fmt(old)} to {_fmt(new)}"
    else:
        direction, moved = "flat", f"mean stayed at {_fmt(old)}"
    percent = None if old == 0 else 100.0 * (new - old) / abs(old)
    delay = int(alarm["detected_at"] - alarm["change_at"])
    reason = (f"{column}: {moved}"
              + ("" if percent is None or direction == "flat" else f" ({percent:+.0f}%)")
              + f" around {times[start]}; {DETECTOR_LABELS[detector]} raised the alarm "
              + f"{delay} row{'' if delay == 1 else 's'} later")
    return {
        "column": column,
        "detector": detector,
        "statistic": _num(alarm["statistic"]),
        "threshold": _num(alarm["threshold"]),
        "change_at": times[start],
        "change_line": int(lines[start]) + 2,      # line in the file (header is line 1)
        "detected_at": times[seen],
        "detected_line": int(lines[seen]) + 2,
        # Measured from the detector's own estimate of where the change
        # began; the benchmark measures it against a planted change instead.
        "detection_delay_rows": delay,
        "before": _describe(before),
        "after": _describe(after),
        "direction": direction,
        "pct_change": None if percent is None else round(percent, 2),
        "reason": reason,
    }


def _numeric_column(name: str, cells: pd.Series, times: list[str],
                    lines: np.ndarray) -> tuple[dict, list[dict]]:
    """One numeric column, cells already in time order: its half-vs-half entry and its change-points."""
    numbers = pd.to_numeric(cells, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
    kept = np.flatnonzero(np.isfinite(numbers))  # time-ordered positions that hold a number
    values = numbers[kept]
    entry = {"column": name, "kind": "numeric", "rows": int(values.size)}
    if values.size < MIN_ROWS:
        return {**entry, "skipped": True,
                "reason": f"needs at least {MIN_ROWS} numeric values (has {values.size})"}, []
    if np.ptp(values) == 0:
        return {**entry, "skipped": True, "reason": "every value is the same"}, []

    clipped, width = clip_to_fences(values)
    half = values.size // 2
    entry.update({"skipped": False, "reason": None, "split_at": times[kept[half]],
                  "reference": _describe(clipped[:half]), "current": _describe(clipped[half:]),
                  **compare_numeric(values[:half], values[half:])})

    detectors = {
        "page_hinkley": PageHinkley(),
        "adwin": ADWIN(value_range=width, clock=max(1, values.size // ADWIN_MAX_CHECKS)),
    }
    found = []
    for detector_name, detector in detectors.items():
        alarms = detect_stream(clipped, detector)
        starts = [alarm["change_at"] for alarm in alarms] + [values.size]
        for k, alarm in enumerate(alarms):
            before = clipped[(starts[k - 1] if k else 0):starts[k]]
            after = clipped[starts[k]:starts[k + 1]]
            found.append((starts[k], _change_point(name, detector_name, alarm, before, after,
                                                   times, lines, kept)))
    found.sort(key=lambda item: item[0])  # stable: time order, Page-Hinkley first on ties
    entry["change_points"] = len(found)
    return entry, [event for _, event in found]


def _categorical_column(name: str, kind: str, cells: pd.Series, times: list[str]) -> dict:
    """One categorical column, cells already in time order: its half-vs-half entry."""
    labels = cells.astype(str).str.strip().to_numpy()
    kept = np.flatnonzero(labels != "")
    values = labels[kept]
    entry = {"column": name, "kind": kind, "rows": int(values.size)}
    if values.size < MIN_ROWS:
        return {**entry, "skipped": True,
                "reason": f"needs at least {MIN_ROWS} non-blank values (has {values.size})"}
    half = values.size // 2
    return {**entry, "skipped": False, "reason": None, "split_at": times[kept[half]],
            **compare_categorical(values[:half], values[half:])}


def find_change_points(df: pd.DataFrame, types: dict[str, str]) -> dict:
    """Find drift and change-points in a parsed CSV that has a datetime column.

    Inputs: `df` as returned by parse_csv_bytes (all strings), `types` from
    infer_types. Output: JSON-safe dict. Without a datetime column, or with
    fewer than MIN_ROWS dated rows, it is {"skipped": True, "reason": ...,
    "change_points": []}. Otherwise the rows are put in time order (by the
    first datetime column) and the dict holds
    - "columns": for every numeric / categorical column, the first half of
      the period compared with the second half (PSI, KS and JS for numbers,
      JS for labels, plus `level` and `drifted`);
    - "change_points": for every numeric column, each Page-Hinkley and ADWIN
      alarm: when the change began, when it was detected, the delay in rows,
      before/after statistics and a plain-English reason.
    Values beyond a column's box-plot fences are clipped before the
    streaming detectors and the before/after figures see them, so a single
    extreme row can neither fake nor hide a change.
    """
    time_positions = [i for i, name in enumerate(df.columns) if types.get(name) == "datetime"]
    if not time_positions:
        return _skipped("no datetime column in this file")
    time_position = time_positions[0]
    time_column = str(df.columns[time_position])
    try:
        when = pd.to_datetime(df.iloc[:, time_position], errors="coerce", format="mixed", utc=True)
        stamps = when.dt.tz_localize(None).to_numpy()
    except (ValueError, TypeError, OverflowError, AttributeError):
        return _skipped(f"could not read '{time_column}' as dates")
    dated = np.flatnonzero(~np.isnat(stamps))
    if dated.size < MIN_ROWS:
        return _skipped(f"needs at least {MIN_ROWS} rows with a readable '{time_column}' "
                        f"(file has {dated.size})")
    # Stable sort: rows sharing a timestamp keep their file order, so the
    # result never depends on how the sort breaks ties.
    order = dated[np.argsort(stamps[dated], kind="stable")]
    stamps = stamps[order]
    whole_days = bool((stamps == stamps.astype("datetime64[D]")).all())
    times = np.datetime_as_string(stamps, unit="D" if whole_days else "s").tolist()

    columns, change_points = [], []
    for position, name in enumerate(df.columns):
        kind = types.get(name)
        if position == time_position or kind not in ("numeric", "categorical", "boolean"):
            continue
        cells = df.iloc[order, position]
        if kind == "numeric":
            entry, events = _numeric_column(str(name), cells, times, order)
            change_points.extend(events)
        else:
            entry = _categorical_column(str(name), kind, cells, times)
        columns.append(entry)
    if not columns:
        return _skipped("no numeric or categorical column to follow over time")

    return {
        "skipped": False,
        "reason": None,
        "time_column": time_column,
        "period": {"start": times[0], "end": times[-1]},
        "rows_used": int(order.size),
        "method": ("rows sorted by the time column; each column's first half compared with its "
                   "second half (PSI, Kolmogorov-Smirnov, Jensen-Shannon); Page-Hinkley and "
                   "ADWIN run over each numeric column in time order"),
        "columns": columns,
        "change_points": change_points[:MAX_CHANGE_POINTS],
        "change_points_total": len(change_points),
        "truncated": len(change_points) > MAX_CHANGE_POINTS,
    }
