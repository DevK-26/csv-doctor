"""Step F — fuzzy duplicate detection (BLUEPRINT §12 T3, sprint Phase B1).

`df.duplicated()` only finds byte-identical rows. The same customer typed as
"Jon Smith, 12 M.G. Road, delhi " and "John Smith, 12 MG Road, Delhi" slips
past it. This module does record linkage in four steps:

1. Normalise every value (case, accents, punctuation, whitespace, a few
   address abbreviations) so formatting noise never counts as a difference.
2. Block: cheap keys put only plausibly-equal rows in the same bucket, so we
   never compare all n*(n-1)/2 pairs. Keys come from a *phonetic* view of
   the first and last word (Soundex) and a *spelling* view (first two letters
   of each). Phonetic keys survive "Jon"/"John"; the spelling key survives
   the typos Soundex is fragile to ("Rajesh"/"Rajseh" get different Soundex
   codes). A pair needs to share just one key on one column to be compared.
   Both are order-invariant, so "Smith John" still meets "John Smith".
3. Score each candidate pair: rapidfuzz token-sort ratio (word-order proof),
   Jaro-Winkler (typo proof) and a weakest-word check per column, combined across
   columns. Identity columns (mostly-distinct names/addresses) carry the
   score; low-cardinality columns such as city only nudge it, because two
   different people living in the same city is not evidence of identity.
4. Cluster matched pairs into groups (union-find) and give each group a
   confidence: the mean similarity over EVERY pair of its members, so a
   weak "A~B, B~C but A!~C" chain scores low instead of looking certain.

Nothing is deleted or merged: the output is groups for a human to review,
each with member row indices, the fields that matched and why.

Scope: this targets entity tables (one row per person/company/place). In a
transaction log the same customer legitimately appears many times; there
no column is mostly-distinct, so only exact full-row duplicates are reported.

Fully deterministic: no randomness, ordered iteration everywhere.
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from itertools import combinations

import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

# --- tunables -------------------------------------------------------------
# Thresholds were chosen on benchmarks/ data with precision in mind: a missed
# duplicate costs the user nothing, a wrong "duplicate" wastes their review.
MATCH_THRESHOLD = 0.88       # combined pair score needed to link two rows
MIN_COLUMN_SIM = 0.75        # no identity column may fall below this
CONTEXT_WEIGHT = 0.10        # share of the score from low-cardinality columns
MIN_IDENTITY_UNIQUE = 0.5    # "mostly distinct" = this share of values unique
MAX_VALUE_LEN = 80           # longer median values are free text, not names
NUMBER_MISMATCH_FACTOR = 0.6  # "Flat 12" vs "Flat 21" are different places
MAX_BLOCK_SIZE = 200         # bigger buckets fall back to a sliding window
SORTED_WINDOW = 10
MAX_PAIRWISE_CLUSTER = 100   # above this, confidence uses matched edges only
MAX_GROUPS_RETURNED = 200

MATCHABLE_TYPES = ("text", "categorical")
ABBREVIATIONS = {
    "road": "rd", "street": "st", "avenue": "ave", "lane": "ln",
    "apartment": "apt", "apartments": "apt", "building": "bldg",
    "mount": "mt", "saint": "st", "nagar": "ngr",
}
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES = re.compile(r"\s+")


# --- normalisation and blocking keys ----------------------------------------

def normalise(value: str) -> str:
    """Lower-case, strip accents/punctuation/extra spaces, abbreviate road words.

    "M.G. Road " -> "mg rd"; "Café-Noir" -> "cafe noir". Periods are dropped
    (not spaced) so initials stay glued: "M.G." == "MG".
    """
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    text = _PUNCT.sub(lambda m: "" if m.group() in ".'’" else " ", text)
    tokens = [ABBREVIATIONS.get(t, t) for t in _SPACES.split(text.strip()) if t]
    return " ".join(tokens)


def soundex(word: str) -> str:
    """American Soundex of one word (letters only); '' if it has no letters."""
    codes = {c: d for d, cs in {"1": "bfpv", "2": "cgjkqsxz", "3": "dt",
                                "4": "l", "5": "mn", "6": "r"}.items() for c in cs}
    letters = [c for c in word.lower() if "a" <= c <= "z"]
    if not letters:
        return ""
    out = letters[0].upper()
    prev = codes.get(letters[0], "")
    for c in letters[1:]:
        code = codes.get(c, "")
        if code and code != prev:
            out += code
        if c not in "hw":  # h/w do not separate equal codes; vowels do
            prev = code
    return (out + "000")[:4]


def blocking_keys(normalised: str) -> list[str]:
    """Cheap bucket keys for one normalised value (see module docstring)."""
    if not normalised:
        return []
    words = [w for w in normalised.split() if any(c.isalpha() for c in w)]
    if not words:
        return ["num:" + normalised]
    first, last = words[0], words[-1]
    phonetic = "|".join(sorted((soundex(first), soundex(last))))
    spelling = "|".join(sorted((first[:2], last[:2])))
    return ["sx:" + phonetic, "pf:" + spelling]


# --- similarity --------------------------------------------------------------

def _sorted_tokens(normalised: str) -> str:
    return " ".join(sorted(normalised.split()))


def _numbers(normalised: str) -> frozenset[str]:
    return frozenset(t for t in normalised.split() if t.isdigit())


def _weakest_token(a: str, b: str) -> float:
    """Worst best-match Jaro-Winkler over the words of either value.

    Catches "Priya Sharma" vs "Rahul Sharma": the shared surname makes the
    whole-string scores high, but "priya" has no counterpart in the other value.
    """
    ta, tb = a.split(), b.split()
    best = [max(JaroWinkler.similarity(t, u) for u in tb) for t in ta]
    best += [max(JaroWinkler.similarity(u, t) for t in ta) for u in tb]
    return min(best)


def value_similarity(a: str, b: str) -> float:
    """Similarity in [0, 1] of two already-normalised values.

    Mean of three views: rapidfuzz token-sort ratio (ignores word order),
    Jaro-Winkler on the sorted words (rewards shared prefixes, tolerates
    typos) and the weakest single-word match (one wholly different word is
    a different entity). A differing set of number tokens (house/flat
    numbers) is damped.
    """
    if a == b:
        return 1.0
    token = fuzz.token_sort_ratio(a, b) / 100.0
    jaro = JaroWinkler.similarity(_sorted_tokens(a), _sorted_tokens(b))
    sim = (token + jaro + _weakest_token(a, b)) / 3
    na, nb = _numbers(a), _numbers(b)
    if (na or nb) and na != nb:
        sim *= NUMBER_MISMATCH_FACTOR
    return sim


class _Linker:
    """Holds normalised columns and scores row pairs against them."""

    def __init__(self, norm: dict[str, list[str]], identity: list[str],
                 context: list[str]) -> None:
        self.norm, self.identity, self.context = norm, identity, context

    def column_sims(self, i: int, j: int) -> dict[str, float]:
        """Per-column similarity for columns where BOTH rows have a value."""
        sims: dict[str, float] = {}
        for col in self.identity + self.context:
            a, b = self.norm[col][i], self.norm[col][j]
            if a and b:
                sims[col] = value_similarity(a, b)
        return sims

    def score(self, i: int, j: int) -> float:
        """Combined pair score in [0, 1]; 0.0 when there is no identity evidence."""
        sims = self.column_sims(i, j)
        ident = [sims[c] for c in self.identity if c in sims]
        if not ident or min(ident) < MIN_COLUMN_SIM:
            return 0.0
        ident_score = sum(ident) / len(ident)
        ctx = [sims[c] for c in self.context if c in sims]
        if not ctx:
            return ident_score
        return (1 - CONTEXT_WEIGHT) * ident_score + CONTEXT_WEIGHT * sum(ctx) / len(ctx)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:  # smaller index wins so the result is order-independent
            self.parent[max(ra, rb)] = min(ra, rb)


# --- column selection ---------------------------------------------------------

def select_columns(df: pd.DataFrame, types: dict[str, str],
                   norm: dict[str, list[str]]) -> tuple[list[str], list[str]]:
    """Split matchable columns into (identity, context).

    Identity = name/address-like: text/categorical, short values, mostly
    distinct. Context = the remaining text/categorical columns (city, status).
    """
    identity: list[str] = []
    context: list[str] = []
    for col in df.columns:
        if types.get(col) not in MATCHABLE_TYPES:
            continue
        values = [v for v in norm[col] if v]
        if not values:
            continue
        lengths = sorted(len(v) for v in values)
        if lengths[len(lengths) // 2] > MAX_VALUE_LEN:
            continue
        if len(set(values)) / len(values) >= MIN_IDENTITY_UNIQUE:
            identity.append(col)
        else:
            context.append(col)
    return identity, context


# --- blocking -------------------------------------------------------------------

def candidate_pairs(norm: dict[str, list[str]], blocking_cols: list[str],
                    rows: list[int]) -> tuple[set[tuple[int, int]], int]:
    """Candidate row pairs from blocking over `rows`; returns (pairs, oversize_blocks)."""
    buckets: dict[str, list[int]] = defaultdict(list)
    for col in blocking_cols:
        for i in rows:
            for key in blocking_keys(norm[col][i]):
                buckets[f"{col}#{key}"].append(i)
    pairs: set[tuple[int, int]] = set()
    oversize = 0
    for key in sorted(buckets):
        members = buckets[key]
        if len(members) < 2:
            continue
        if len(members) <= MAX_BLOCK_SIZE:
            pairs.update(combinations(sorted(members), 2))
            continue
        # Giant bucket (e.g. a very common surname): comparing everyone is
        # quadratic, so slide a window over the rows sorted by the column text.
        oversize += 1
        col = key.split("#", 1)[0]
        ordered = sorted(members, key=lambda r: (norm[col][r], r))
        for pos, i in enumerate(ordered):
            for j in ordered[pos + 1: pos + 1 + SORTED_WINDOW]:
                pairs.add((min(i, j), max(i, j)))
    return pairs, oversize


# --- public API -----------------------------------------------------------------

def find_duplicates(df: pd.DataFrame, types: dict[str, str]) -> dict:
    """Find exact and near-duplicate rows and cluster them into review groups.

    Inputs: `df` as returned by parse_csv_bytes (all strings) and `types` from
    infer_types. Output (JSON-safe): {"skipped", "reason", "method", "stats",
    "groups", ...}. Each group has `rows` (0-based data-row indices, sorted),
    `kind` ("exact" = every column identical, "near" otherwise), `confidence`
    (mean similarity over all member pairs), `weakest_pair`, `matched_fields`
    (per-column similarity and values), `differs_in` and plain-English
    `reasons`. Nothing is deleted. Exact full-row duplicates are always
    found; fuzzy matching runs only if the file has name/address-like columns.
    """
    n = len(df)
    empty = {"skipped": True, "reason": None, "method": None,
             "stats": _stats(n, 0, 0, 0, 0), "duplicate_groups": 0,
             "duplicate_rows": 0, "groups": []}
    if n < 2:
        return {**empty, "reason": "need at least 2 rows to look for duplicates"}

    raw_cols = {c: df[c].astype(str).tolist() for c in df.columns}
    norm = {c: [normalise(v) for v in raw_cols[c]] for c in df.columns}
    uf = _UnionFind(n)
    edges: dict[tuple[int, int], float] = {}

    # Exact full-row duplicates: hash the raw row, no similarity needed.
    seen: dict[tuple[str, ...], int] = {}
    for i in range(n):
        key = tuple(raw_cols[c][i] for c in df.columns)
        if key in seen:
            uf.union(seen[key], i)
            edges[(seen[key], i)] = 1.0
        else:
            seen[key] = i

    identity, context = select_columns(df, types, norm)
    method = {"identity_columns": identity, "context_columns": context,
              "match_threshold": MATCH_THRESHOLD, "min_column_similarity": MIN_COLUMN_SIM}
    candidates = oversize = naive = 0
    reason = None
    if identity:
        linker = _Linker(norm, identity, context)
        # Rows identical after normalisation are the same entity by definition:
        # link them directly and let one representative enter blocking.
        sig_rep: dict[tuple[str, ...], int] = {}
        reps: list[int] = []
        for i in range(n):
            sig = tuple(norm[c][i] for c in identity + context)
            if all(not s for s in sig):
                continue
            if sig in sig_rep:
                if uf.find(sig_rep[sig]) != uf.find(i):
                    uf.union(sig_rep[sig], i)
                    edges[(sig_rep[sig], i)] = 1.0
            else:
                sig_rep[sig] = i
                reps.append(i)
        naive = n * (n - 1) // 2  # baseline = comparing every row with every other
        pairs, oversize = candidate_pairs(norm, identity, reps)
        candidates = len(pairs)
        for i, j in sorted(pairs):
            score = linker.score(i, j)
            if score >= MATCH_THRESHOLD:
                uf.union(i, j)
                edges[(i, j)] = score
    else:
        reason = ("no name/address-like columns (text that is mostly distinct); "
                  "only exact full-row duplicates were checked")

    members: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        members[uf.find(i)].append(i)
    clusters = sorted((m for m in members.values() if len(m) > 1),
                      key=lambda m: m[0])
    linker = _Linker(norm, identity, context)
    row_keys = [tuple(raw_cols[c][i] for c in df.columns) for i in range(n)]
    groups = [_describe(m, df, raw_cols, norm, linker, edges, row_keys) for m in clusters]
    groups.sort(key=lambda g: (-g["confidence"], -g["size"], g["rows"][0]))
    for gid, g in enumerate(groups, start=1):
        g["group_id"] = gid
    surplus = sum(len(m) - 1 for m in clusters)
    return {
        "skipped": False, "reason": reason, "method": method,
        "stats": _stats(n, naive, candidates, len(edges), oversize),
        "duplicate_groups": len(groups), "duplicate_rows": surplus,
        "groups": groups[:MAX_GROUPS_RETURNED],
        "groups_truncated": len(groups) > MAX_GROUPS_RETURNED,
    }


def candidate_pair_set(df: pd.DataFrame, types: dict[str, str]) -> set[tuple[int, int]]:
    """Row pairs (i < j) that blocking would hand to the similarity scorer.

    Inputs as for find_duplicates. Exposed so tests and the benchmark can
    measure blocking recall (does it keep the true pairs?) separately from
    scoring quality. Uses every row; find_duplicates additionally collapses
    rows that are identical after normalisation before blocking.
    """
    norm = {c: [normalise(v) for v in df[c].astype(str)] for c in df.columns}
    identity, _ = select_columns(df, types, norm)
    if not identity or len(df) < 2:
        return set()
    return candidate_pairs(norm, identity, list(range(len(df))))[0]


def _stats(rows: int, naive: int, candidates: int, linked: int, oversize: int) -> dict:
    return {"rows": rows, "naive_pairs": naive, "candidate_pairs": candidates,
            "linked_pairs": linked, "oversize_blocks": oversize,
            "blocking_reduction": round(1 - candidates / naive, 4) if naive else None}


def _describe(rows: list[int], df: pd.DataFrame, raw: dict[str, list[str]],
              norm: dict[str, list[str]], linker: _Linker,
              edges: dict[tuple[int, int], float],
              row_keys: list[tuple[str, ...]]) -> dict:
    """Build one review group: confidence, matched fields, differences, reasons."""
    if len(rows) <= MAX_PAIRWISE_CLUSTER:
        pair_scores = [_pair_score(i, j, linker, edges, row_keys)
                       for i, j in combinations(rows, 2)]
    else:
        member_set = set(rows)
        pair_scores = [s for (i, _), s in edges.items() if i in member_set]
    confidence = sum(pair_scores) / len(pair_scores)
    exact = all(len({raw[c][i] for i in rows}) == 1 for c in df.columns)

    fields, differs, reasons = [], [], []
    for col in df.columns:
        raws = [raw[col][i] for i in rows]
        if len(set(raws)) > 1:
            differs.append(col)
        if col not in linker.identity + linker.context:
            continue
        sims = [linker.column_sims(i, j).get(col) for i, j in combinations(rows, 2)]
        sims = [s for s in sims if s is not None]
        if not sims:
            continue
        unique_raw = sorted({v for v in raws if v != ""})
        same_norm = len({norm[col][i] for i in rows if norm[col][i]}) == 1
        fields.append({"column": col, "similarity": round(sum(sims) / len(sims), 4),
                       "identical": len(unique_raw) == 1, "identical_after_cleanup": same_norm,
                       "values": unique_raw})
        if len(unique_raw) == 1:
            reasons.append(f"{col}: identical ({unique_raw[0]!r})")
        elif same_norm:
            reasons.append(f"{col}: same after ignoring case/spacing/punctuation "
                           f"({', '.join(repr(v) for v in unique_raw)})")
        else:
            reasons.append(f"{col}: similar spelling, similarity {sum(sims) / len(sims):.2f} "
                           f"({', '.join(repr(v) for v in unique_raw)})")
    if exact:
        reasons.insert(0, "every column is identical: exact duplicate rows")
    elif differs:
        reasons.append("values still differ in: " + ", ".join(differs))
    return {
        "kind": "exact" if exact else "near",
        "rows": rows, "size": len(rows),
        "confidence": round(confidence, 4),
        "weakest_pair": round(min(pair_scores), 4),
        "matched_fields": fields, "differs_in": differs, "reasons": reasons,
    }


def _pair_score(i: int, j: int, linker: _Linker,
                edges: dict[tuple[int, int], float],
                row_keys: list[tuple[str, ...]]) -> float:
    if (i, j) in edges:
        return edges[(i, j)]
    if row_keys[i] == row_keys[j]:
        return 1.0
    # Not a matched edge: either never a candidate pair or below threshold.
    # Compute directly (ignoring the cut-off floor) so chains lower confidence.
    sims = linker.column_sims(i, j)
    ident = [sims[c] for c in linker.identity if c in sims]
    if not ident:
        return 0.0
    ident_score = sum(ident) / len(ident)
    ctx = [sims[c] for c in linker.context if c in sims]
    return ident_score if not ctx else (
        (1 - CONTEXT_WEIGHT) * ident_score + CONTEXT_WEIGHT * sum(ctx) / len(ctx))
