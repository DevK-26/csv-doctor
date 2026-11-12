"""Step D — explainable anomaly ensemble (BLUEPRINT §12 T2, sprint Phase A1).

Three detectors from three families score every row on the numeric columns:

- Isolation Forest (trees): anomalies are isolated in few random splits.
- Local Outlier Factor (density): anomalies sit in sparser regions than
  their neighbours.
- A small PyTorch autoencoder (reconstruction): anomalies break the
  relationships between columns that normal rows follow.

Their raw scores live on unrelated scales, so each is calibrated onto a
common 0-1 scale (see `calibrate`) before a weighted average gives the single
ensemble score. The calibrated score is a severity scale, not a probability:
0.5 means "FLAG_Z robust standard deviations above the typical row".

Every flagged row carries reasons: which columns drove the score, measured
by per-column reconstruction error (autoencoder) and by perturbation
(Isolation Forest / LOF: reset one column to its typical value, see how far
the score drops).

Everything is deterministic for a given `seed`.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

MODELS = ("isolation_forest", "local_outlier_factor", "autoencoder")
MODEL_LABELS = {
    "isolation_forest": "Isolation Forest",
    "local_outlier_factor": "Local Outlier Factor",
    "autoencoder": "Autoencoder",
}

MIN_ROWS = 50
MIN_NUMERIC_COLS = 2
DEFAULT_SEED = 42
DEFAULT_THRESHOLD = 0.5
DEFAULT_MAX_ROWS = 50

# Calibration: a raw score FLAG_Z robust standard deviations above the median
# maps to 0.5; FLAG_SLOPE controls how fast it saturates around that point.
FLAG_Z = 4.0
FLAG_SLOPE = 1.0
# Scaled features are clipped to this before model fitting so one absurd
# value cannot dominate the autoencoder's loss or LOF's distances.
CLIP = 10.0

IF_TREES = 200
LOF_NEIGHBORS = 20
AE_STEPS = 600
AE_BATCH = 256
AE_LR = 5e-3
# Fraction of each batch (the worst-reconstructed rows) excluded from the
# gradient, so the autoencoder cannot simply memorise the anomalies.
AE_TRIM = 0.1


# --------------------------------------------------------------------------
# Feature preparation
# --------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame, types: dict[str, str]) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Turn the numeric columns of a parsed (all-string) frame into a matrix.

    Returns (values, names, missing): `values` is float64 [rows, features]
    with blanks/unparseable cells imputed by the column median, `missing`
    marks the imputed cells. Columns that are numerically constant are
    dropped — they carry no signal and break scaling.
    """
    names, cols, miss = [], [], []
    for col in df.columns:
        if types.get(col) != "numeric":
            continue
        nums = pd.to_numeric(df[col], errors="coerce").astype("float64")
        nums = nums.where(np.isfinite(nums))
        if nums.notna().sum() == 0:
            continue
        median = float(nums.median())
        filled = nums.fillna(median).to_numpy()
        if np.ptp(filled) == 0:
            continue
        names.append(col)
        cols.append(filled)
        miss.append(nums.isna().to_numpy())
    if not names:
        return np.empty((len(df), 0)), [], np.empty((len(df), 0), dtype=bool)
    return np.column_stack(cols), names, np.column_stack(miss)


def robust_scale(values: np.ndarray) -> np.ndarray:
    """Centre each column on its median and divide by its IQR-based spread.

    Median/IQR instead of mean/std so the anomalies we are hunting do not
    stretch the scale they are measured on. Result is clipped to +-CLIP.
    """
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values, axis=0)
    q25, q75 = np.percentile(values, [25, 75], axis=0)
    spread = (q75 - q25) / 1.349  # IQR of a normal distribution = 1.349 sigma
    std = values.std(axis=0)
    spread = np.where(spread > 0, spread, std)
    spread = np.where(spread > 0, spread, 1.0)
    return np.clip((values - median) / spread, -CLIP, CLIP)


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

def calibrate(raw: np.ndarray) -> np.ndarray:
    """Map raw anomaly scores (higher = stranger) onto a common 0-1 scale.

    z = (score - median) / robust_sigma, then a logistic centred on FLAG_Z.
    Median and MAD are used because the mean/std of the scores would be
    inflated by the very anomalies being scored. If every row scores the
    same, nothing stands out and all calibrated scores are ~0.
    """
    raw = np.asarray(raw, dtype=np.float64)
    median = np.median(raw)
    sigma = 1.4826 * np.median(np.abs(raw - median))
    if sigma <= 0:
        q25, q75 = np.percentile(raw, [25, 75])
        sigma = (q75 - q25) / 1.349
    if sigma <= 0:
        sigma = raw.std()
    if sigma <= 0:
        z = np.zeros_like(raw)
    else:
        z = (raw - median) / sigma
    return 1.0 / (1.0 + np.exp(-np.clip((z - FLAG_Z) / FLAG_SLOPE, -50, 50)))


# --------------------------------------------------------------------------
# The three detectors. Each returns raw scores where higher = more anomalous
# plus whatever it needs to explain a row later.
# --------------------------------------------------------------------------

def _fit_isolation_forest(X: np.ndarray, seed: int) -> tuple[np.ndarray, IsolationForest]:
    model = IsolationForest(n_estimators=IF_TREES, random_state=seed)
    model.fit(X)
    return -model.score_samples(X), model


def _fit_lof(X: np.ndarray) -> tuple[np.ndarray, LocalOutlierFactor | None]:
    # Exact duplicate rows give zero neighbour distances and make LOF blow
    # up to ~1e10, so LOF is fitted on distinct rows and mapped back.
    unique, inverse = np.unique(X, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    if len(unique) < 3:
        return np.zeros(len(X)), None
    k = min(LOF_NEIGHBORS, len(unique) - 1)
    # novelty=True only so score_samples() exists for the perturbation
    # explanations; negative_outlier_factor_ is still the leave-self-out
    # score of each fitted row.
    model = LocalOutlierFactor(n_neighbors=k, novelty=True)
    model.fit(unique)
    lof = -model.negative_outlier_factor_
    # LOF is a density ratio (1 = same density as neighbours): log makes it
    # symmetric around 0 so calibration is not dominated by the ratio's tail.
    return np.log(np.clip(lof, 1e-6, 1e6))[inverse], model


def _lof_score_new(model: LocalOutlierFactor, rows: np.ndarray) -> np.ndarray:
    return np.log(np.clip(-model.score_samples(rows), 1e-6, 1e6))


def _fit_autoencoder(X: np.ndarray, seed: int) -> np.ndarray:
    """Train a d -> h -> b -> h -> d autoencoder; return per-cell squared error.

    Raw score per row is the mean of its row in the returned matrix; the
    per-column errors are the autoencoder's own explanation.
    """
    import torch  # local import: keeps importing the engine cheap
    from torch import nn

    n, d = X.shape
    hidden = max(8, 2 * d)
    bottleneck = max(1, d // 2)

    rng_state = torch.random.get_rng_state()  # do not disturb the caller's RNG
    try:
        torch.manual_seed(seed)
        model = nn.Sequential(
            nn.Linear(d, hidden), nn.Tanh(),
            nn.Linear(hidden, bottleneck), nn.Tanh(),
            nn.Linear(bottleneck, hidden), nn.Tanh(),
            nn.Linear(hidden, d),
        ).double()
        data = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float64))
        optim = torch.optim.Adam(model.parameters(), lr=AE_LR)
        gen = torch.Generator().manual_seed(seed)
        batch = min(n, AE_BATCH)
        keep = max(1, int(math.ceil(batch * (1.0 - AE_TRIM))))
        for _ in range(AE_STEPS):
            idx = torch.randperm(n, generator=gen)[:batch]
            xb = data[idx]
            per_row = ((model(xb) - xb) ** 2).mean(dim=1)
            loss = torch.topk(per_row, keep, largest=False).values.mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
        with torch.no_grad():
            errors = ((model(data) - data) ** 2).numpy()
    finally:
        torch.random.set_rng_state(rng_state)
    return errors


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def _normalise_weights(weights: dict[str, float] | None) -> dict[str, float]:
    if weights is None:
        return {m: 1.0 / len(MODELS) for m in MODELS}
    unknown = set(weights) - set(MODELS)
    if unknown:
        raise ValueError(f"unknown model(s) in weights: {sorted(unknown)}; expected {list(MODELS)}")
    w = {m: float(weights.get(m, 0.0)) for m in MODELS}
    total = sum(w.values())
    if total <= 0 or min(w.values()) < 0:
        raise ValueError("ensemble weights must be non-negative and sum to more than 0")
    return {m: v / total for m, v in w.items()}


def score_matrix(X: np.ndarray, seed: int = DEFAULT_SEED,
                 weights: dict[str, float] | None = None) -> dict:
    """Score an already-scaled numeric matrix with all three models.

    Input: float matrix [rows, features] (use `robust_scale` first).
    Output: {"scores": {model: calibrated 0-1 array, "ensemble": array},
             "raw": {model: raw array}, "weights": {...}, "_fitted": {...}}.
    This is the exact code path the benchmark measures.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 3 or X.shape[1] < 1:
        raise ValueError("score_matrix needs a 2-D matrix with at least 3 rows and 1 column")
    if not np.isfinite(X).all():
        raise ValueError("score_matrix input contains NaN or infinite values; impute first")
    w = _normalise_weights(weights)

    if_raw, if_model = _fit_isolation_forest(X, seed)
    lof_raw, lof_model = _fit_lof(X)
    ae_cells = _fit_autoencoder(X, seed)
    # log: reconstruction errors spread multiplicatively, like LOF's ratio.
    ae_raw = np.log(ae_cells.mean(axis=1) + 1e-12)

    raw = {"isolation_forest": if_raw, "local_outlier_factor": lof_raw, "autoencoder": ae_raw}
    scores = {m: calibrate(raw[m]) for m in MODELS}
    scores["ensemble"] = sum(w[m] * scores[m] for m in MODELS)
    return {
        "scores": scores,
        "raw": raw,
        "weights": w,
        "_fitted": {"isolation_forest": if_model, "local_outlier_factor": lof_model,
                    "autoencoder_cell_errors": ae_cells},
    }


def flag_mask(scores: dict[str, np.ndarray],
              threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """The product's flag rule, as a boolean mask over rows.

    Input: the "scores" dict from `score_matrix`. Output: bool array, True
    where the ensemble score is >= threshold AND at least 2 of the 3
    detectors are individually >= threshold. `find_anomalies` and the
    benchmark both call this, so the benchmark grades the rule the API ships.
    """
    # Calibrated scores are relative to the data's own bulk, so a single
    # detector's tail always exists even in perfectly clean data — majority
    # agreement is what keeps clean files quiet while planted outliers,
    # which trip every detector family at once, still flag.
    votes = np.column_stack([scores[m] >= threshold for m in MODELS])
    return (scores["ensemble"] >= threshold) & (votes.sum(axis=1) >= 2)


# --------------------------------------------------------------------------
# Explanations
# --------------------------------------------------------------------------

def _shares(contrib: np.ndarray) -> np.ndarray:
    contrib = np.clip(contrib, 0.0, None)
    total = contrib.sum()
    return contrib / total if total > 0 else np.zeros_like(contrib)


def _perturbation_shares(x: np.ndarray, base: float, score_fn) -> np.ndarray:
    """Share of a row's raw score owed to each column.

    Column j's contribution = how much the score drops when x[j] alone is
    reset to the column's typical value (0 after robust scaling).
    """
    variants = np.repeat(x[None, :], len(x), axis=0)
    np.fill_diagonal(variants, 0.0)
    return _shares(base - score_fn(variants))


def _explain_row(i: int, X: np.ndarray, result: dict) -> dict[str, np.ndarray]:
    fitted, raw = result["_fitted"], result["raw"]
    d = X.shape[1]
    out = {"autoencoder": _shares(fitted["autoencoder_cell_errors"][i])}
    forest = fitted["isolation_forest"]
    out["isolation_forest"] = _perturbation_shares(
        X[i], raw["isolation_forest"][i], lambda rows: -forest.score_samples(rows))
    lof = fitted["local_outlier_factor"]
    out["local_outlier_factor"] = (
        _perturbation_shares(X[i], raw["local_outlier_factor"][i],
                             lambda rows: _lof_score_new(lof, rows))
        if lof is not None else np.zeros(d))
    return out


def _fmt(value: float) -> str:
    if not math.isfinite(value):
        return str(value)
    if float(value).is_integer() and abs(value) < 1e15:
        return f"{int(value):,}"
    return f"{value:,.4g}" if abs(value) < 1e6 else f"{value:,.0f}"


def _reason_text(feature: str, value: float, median: float, lo: float, hi: float,
                 share: float) -> str:
    if value > hi:
        where = f"far above the usual range ({_fmt(lo)} to {_fmt(hi)}, typical {_fmt(median)})"
    elif value < lo:
        where = f"far below the usual range ({_fmt(lo)} to {_fmt(hi)}, typical {_fmt(median)})"
    else:
        where = (f"within the usual range ({_fmt(lo)} to {_fmt(hi)}) but in an unusual "
                 f"combination with the other columns")
    return f"{feature} = {_fmt(value)} is {where}; drove {share:.0%} of the anomaly score"


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def _skipped(reason: str) -> dict:
    return {"skipped": True, "reason": reason, "rows": []}


def find_anomalies(df: pd.DataFrame, types: dict[str, str], *,
                   seed: int = DEFAULT_SEED,
                   threshold: float = DEFAULT_THRESHOLD,
                   max_rows: int = DEFAULT_MAX_ROWS,
                   top_k: int = 3,
                   weights: dict[str, float] | None = None) -> dict:
    """Find anomalous rows in a parsed CSV and explain each one.

    Inputs: `df` as returned by parse_csv_bytes (all strings), `types` from
    infer_types. Only columns typed "numeric" are used (IDs never are).
    Output: JSON-safe dict. If the file is too small it is
    {"skipped": True, "reason": ..., "rows": []}; otherwise it lists the
    flagged rows (ensemble score >= threshold AND at least 2 of 3
    detectors individually >= threshold, strongest first, at most
    `max_rows`), each with per-model scores, votes, top features and
    human-readable `reasons`.
    """
    need = f"needs at least {MIN_ROWS} rows and {MIN_NUMERIC_COLS} numeric columns"
    if len(df) < MIN_ROWS:
        return _skipped(f"{need} (file has {len(df)} rows)")
    values, names, missing = prepare_features(df, types)
    if len(names) < MIN_NUMERIC_COLS:
        return _skipped(f"{need} (file has {len(names)} usable numeric columns)")

    X = robust_scale(values)
    result = score_matrix(X, seed=seed, weights=weights)
    scores, w = result["scores"], result["weights"]
    ensemble = scores["ensemble"]

    flagged = np.flatnonzero(flag_mask(scores, threshold))
    # Stable sort: ties resolve by row order, so output is reproducible.
    flagged = flagged[np.argsort(-ensemble[flagged], kind="stable")]

    median = np.median(values, axis=0)
    lo, hi = np.percentile(values, [5, 95], axis=0)

    rows = []
    for i in flagged[:max_rows]:
        per_model = _explain_row(int(i), X, result)
        # Models that scored the row higher get more say in its explanation.
        mix = sum(w[m] * scores[m][i] * per_model[m] for m in MODELS)
        combined = _shares(mix)
        order = [j for j in np.argsort(-combined, kind="stable")[:top_k] if combined[j] > 0]
        votes = {m: bool(scores[m][i] >= threshold) for m in MODELS}
        top_features = [{
            "feature": names[j],
            "value": float(values[i, j]),
            "value_was_missing": bool(missing[i, j]),
            "typical": float(median[j]),
            "usual_low": float(lo[j]),
            "usual_high": float(hi[j]),
            "direction": ("high" if values[i, j] > hi[j] else
                          "low" if values[i, j] < lo[j] else "unusual combination"),
            "contribution": round(float(combined[j]), 4),
        } for j in order]
        reasons = [_reason_text(names[j], values[i, j], median[j], lo[j], hi[j], combined[j])
                   for j in order]
        agree = [MODEL_LABELS[m] for m in MODELS if votes[m]]
        dissent = [MODEL_LABELS[m] for m in MODELS if not votes[m]]
        verdict = f"{len(agree)} of {len(MODELS)} detectors flag this row"
        if agree:
            verdict += f" ({', '.join(agree)})"
        if dissent and agree:
            verdict += f"; {', '.join(dissent)} did not"
        reasons.append(verdict)
        rows.append({
            "row_index": int(i),          # 0-based position among data rows
            "line_number": int(i) + 2,    # line in the file (header is line 1)
            "score": round(float(ensemble[i]), 4),
            "model_scores": {m: round(float(scores[m][i]), 4) for m in MODELS},
            "votes": votes,
            "agreement": f"{len(agree)}/{len(MODELS)}",
            "top_features": top_features,
            "per_model_top_features": {
                m: [{"feature": names[j], "contribution": round(float(per_model[m][j]), 4)}
                    for j in np.argsort(-per_model[m], kind="stable")[:top_k]
                    if per_model[m][j] > 0]
                for m in MODELS
            },
            "reasons": reasons,
        })

    return {
        "skipped": False,
        "reason": None,
        "method": ("ensemble of Isolation Forest, Local Outlier Factor and an autoencoder; "
                   "each score calibrated to 0-1, then a weighted average"),
        "features_used": names,
        "rows_scored": int(len(df)),
        "threshold": float(threshold),
        "weights": {m: round(v, 4) for m, v in w.items()},
        "seed": int(seed),
        "flagged_count": int(len(flagged)),
        "flagged_pct": round(100.0 * len(flagged) / len(df), 2),
        "rows_returned": len(rows),
        "truncated": bool(len(flagged) > len(rows)),
        "per_model_flagged": {m: int((scores[m] >= threshold).sum()) for m in MODELS},
        "rows": rows,
    }
