"""Token-level diff features that separate true matches from near-duplicate distractors (design v2 S3
"Features v2", D-013).

True matches carry *explainable* noise (typos, abbreviations, transliteration, legal-form drift, truncation).
Distractors differ by an *unexplainable* change: one content word swapped for another, a business-type word
added ("Group", "Ventures"), or a house number nudged by a few units. Generic string similarity scores both
kinds of difference the same way, so this module classifies each name token's fate individually and learns,
from labelled training pairs only, which unmatched tokens are distractor signatures.

Every function here is pure and allocation-light so it can run over ~50M candidate pairs in ~200k-row batches
across several worker processes: token-pair classification and skeletons are memoized with ``lru_cache``, and
the only per-pair work is over each name's small token list (2-6 tokens), never the batch as a whole.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from functools import lru_cache

import numpy as np
from rapidfuzz.distance import Levenshtein

# --------------------------------------------------------------------------------------------------
# Token classification vocabulary
# --------------------------------------------------------------------------------------------------

# Legal-form words (hand-written, not looked up): their presence/absence differs between true-match
# records all the time ("Pvt Ltd" vs "Private Limited" vs nothing) and is not distractor evidence on its own.
LEGAL_WORDS = frozenset(
    {
        "pvt",
        "private",
        "ltd",
        "limited",
        "llp",
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "co",
        "company",
        "llc",
        "pllc",
        "pc",
        "sarl",
        "sas",
        "sa",
        "eurl",
        "sasu",
        "sci",
    }
)

# Small connector words that a joined/domain-style name typically drops ("Foot AND Ankle" -> "footankle...").
# Hand-written, language-generic; not tied to any one country.
CONNECTORS = frozenset({"and", "&", "of", "the", "at", "de", "et", "und", "y", "e", "a", "el", "la", "le"})

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]")
_VOWEL_Y_H_RE = re.compile(r"[aeiouyh]")
_REPEAT_RE = re.compile(r"(.)\1+")

NAME_DIFF_FEATURES = [
    "nd_l_unexplained",
    "nd_r_unexplained",
    "nd_typo",
    "nd_abbrev",
    "nd_translit",
    "nd_legal_diff",
    "nd_l_unexp_idf",
    "nd_r_unexp_idf",
    "nd_l_logodds_max",
    "nd_l_logodds_sum",
    "nd_r_logodds_max",
    "nd_r_logodds_sum",
    "nd_l_exact_frac",
    "nd_l_ntok",
    "nd_r_ntok",
]
NUMBER_RELATION_FEATURES = ["nr_class", "nr_min_rel_diff", "nr_first_eq", "nr_n_equal", "nr_both_present"]


@lru_cache(maxsize=200_000)
def _norm_token(t: str) -> str:
    """Lowercase a token and keep only ASCII letters/digits (memoized: tokens repeat heavily across pairs)."""
    return _NON_ALNUM_RE.sub("", t.lower())


@lru_cache(maxsize=131_072)
def _skeleton(token: str) -> str:
    """Consonant skeleton of a normalized token: drop vowels, 'y', and 'h', then collapse repeated letters."""
    return _REPEAT_RE.sub(r"\1", _VOWEL_Y_H_RE.sub("", token))


def _is_typo(x: str, y: str) -> bool:
    """True if x/y look like a typo of each other: normalized Levenshtein similarity >= 0.75, or edit
    distance <= 2 for tokens of length >= 5, <= 1 for length 3-4 (too ambiguous below that)."""
    if not x or not y:
        return False
    if Levenshtein.normalized_similarity(x, y, processor=None) >= 0.75:
        return True
    length = max(len(x), len(y))
    if length >= 5:
        return Levenshtein.distance(x, y, processor=None) <= 2
    if length in (3, 4):
        return Levenshtein.distance(x, y, processor=None) <= 1
    return False


def _is_abbrev_pair(x: str, y: str) -> bool:
    """True if the shorter token is a (non-trivial) prefix of the longer one, e.g. an initial or truncation."""
    if not x or not y:
        return False
    short, long_ = (x, y) if len(x) <= len(y) else (y, x)
    return short != long_ and long_.startswith(short)


def _is_translit_pair(x: str, y: str) -> bool:
    """True if x and y share a consonant skeleton of length >= 2 (transliteration-robust equality)."""
    if not x or not y or x == y:
        return False
    sx, sy = _skeleton(x), _skeleton(y)
    return len(sx) >= 2 and sx == sy


@lru_cache(maxsize=500_000)
def _best_candidate(x: str, y: str) -> tuple[int, float, str] | None:
    """Best (priority, score, class) two normalized tokens could count as beyond exact/legal/joined-form:
    typo (priority 0) > abbreviation/prefix (1) > transliteration (2). None if no acceptable partner."""
    if not x or not y:
        return None
    if _is_typo(x, y):
        return (0, Levenshtein.normalized_similarity(x, y, processor=None), "typo")
    if _is_abbrev_pair(x, y):
        short, long_ = (x, y) if len(x) <= len(y) else (y, x)
        return (1, len(short) / len(long_), "abbrev")
    if _is_translit_pair(x, y):
        return (2, 0.5, "translit")
    return None


def _concat_match(
    x_tokens: list[str],
    other_tokens: list[str],
    cls_x: list[str | None],
    cls_other: list[str | None],
    used_x: set[int],
    used_other: set[int],
) -> None:
    """Mark 'abbrev' on token i of x when it equals the concatenation of a consecutive run (>=2 tokens,
    connector words skipped over) of *other*'s still-unmatched tokens, e.g. "footanklemodern" <-
    "foot" + "ankle" + "modern" (skipping "and"). Mutates cls_x/cls_other/used_x/used_other in place."""
    n_other = len(other_tokens)
    for i in range(len(x_tokens)):
        if i in used_x:
            continue
        target = x_tokens[i]
        if len(target) < 4:
            continue
        for start in range(n_other):
            if start in used_other or other_tokens[start] in CONNECTORS:
                continue
            acc = ""
            covered: list[int] = []
            matched = False
            for j in range(start, n_other):
                if j in used_other:
                    break
                tok = other_tokens[j]
                if tok in CONNECTORS:
                    continue
                acc += tok
                if len(acc) > len(target):
                    break
                covered.append(j)
                if acc == target and len(covered) >= 2:
                    matched = True
                    break
            if matched:
                cls_x[i] = "abbrev"
                used_x.add(i)
                for j in covered:
                    cls_other[j] = "abbrev"
                    used_other.add(j)
                break


def _greedy_match(
    na: list[str], nb: list[str], rem_a: list[int], rem_b: list[int]
) -> tuple[dict[int, str], dict[int, str]]:
    """Greedily pair remaining token indices by best (priority, score) from _best_candidate."""
    candidates = []
    for i in rem_a:
        for j in rem_b:
            c = _best_candidate(na[i], nb[j])
            if c is not None:
                prio, score, cls = c
                candidates.append((prio, -score, i, j, cls))
    candidates.sort()
    cls_a: dict[int, str] = {}
    cls_b: dict[int, str] = {}
    used_a: set[int] = set()
    used_b: set[int] = set()
    for _prio, _negscore, i, j, cls in candidates:
        if i in used_a or j in used_b:
            continue
        cls_a[i] = cls
        cls_b[j] = cls
        used_a.add(i)
        used_b.add(j)
    return cls_a, cls_b


def align_tokens(a: list[str], b: list[str]) -> tuple[list[str], list[str]]:
    """Greedily align two names' tokens (normalized: lowercased, ASCII letters/digits only) and classify
    every token of a and of b as one of:

      exact        - equal after normalization
      typo         - small edit distance (see _is_typo)
      abbrev       - one token is a prefix of the other (initial/truncation), or it is the concatenation of a
                     consecutive run of the other side's tokens (connector words skipped), e.g. a joined
                     domain-style name
      translit     - same consonant skeleton (transliteration-robust)
      legal        - a legal-form word (LEGAL_WORDS) with no exact partner on the other side
      unexplained  - no acceptable partner found

    Matching order (each step only considers tokens not already matched): exact string equality, then
    joined-form concatenation, then legal-form words, then a greedy typo/abbrev/translit pass ranked by
    (class priority, similarity score). Returns (classes_for_a, classes_for_b), same length/order as a, b.
    """
    na = [_norm_token(t) for t in a]
    nb = [_norm_token(t) for t in b]
    cls_a: list[str | None] = [None] * len(a)
    cls_b: list[str | None] = [None] * len(b)
    used_a: set[int] = set()
    used_b: set[int] = set()

    # 1. exact matches
    b_by_value: dict[str, list[int]] = {}
    for j, t in enumerate(nb):
        if t:
            b_by_value.setdefault(t, []).append(j)
    for i, t in enumerate(na):
        bucket = b_by_value.get(t) if t else None
        if bucket:
            j = bucket.pop(0)
            cls_a[i] = cls_b[j] = "exact"
            used_a.add(i)
            used_b.add(j)

    # 2. joined-form concatenation, both directions
    _concat_match(na, nb, cls_a, cls_b, used_a, used_b)
    _concat_match(nb, na, cls_b, cls_a, used_b, used_a)

    # 3. legal-form words, still unmatched
    for i, t in enumerate(na):
        if i not in used_a and t in LEGAL_WORDS:
            cls_a[i] = "legal"
            used_a.add(i)
    for j, t in enumerate(nb):
        if j not in used_b and t in LEGAL_WORDS:
            cls_b[j] = "legal"
            used_b.add(j)

    # 4. greedy typo / abbrev(prefix) / translit over what's left
    rem_a = [i for i in range(len(a)) if i not in used_a]
    rem_b = [j for j in range(len(b)) if j not in used_b]
    m_a, m_b = _greedy_match(na, nb, rem_a, rem_b)
    for i, cls in m_a.items():
        cls_a[i] = cls
    for j, cls in m_b.items():
        cls_b[j] = cls

    # 5. everything left is unexplained
    cls_a = [c if c is not None else "unexplained" for c in cls_a]
    cls_b = [c if c is not None else "unexplained" for c in cls_b]
    return cls_a, cls_b


def name_diff_features(
    l_names: list[str],
    r_names: list[str],
    idf: Mapping[str, float] | None,
    logodds: Mapping[str, float] | None,
) -> dict[str, np.ndarray]:
    """Per-pair token-alignment features for l_names[i] vs r_names[i] (see align_tokens for the class
    taxonomy). idf maps a normalized token to its IDF value (missing token -> 1.0). logodds maps a
    normalized token to its learned distractor log-odds from learn_token_logodds (missing/unseen -> 0.0).
    Returns float32 numpy arrays, one entry per pair:

      nd_l_unexplained, nd_r_unexplained - unexplained token count on each side
      nd_typo, nd_abbrev, nd_translit    - matched counts of that class, both sides combined
      nd_legal_diff                      - count of legal-form tokens with no exact partner, both sides
      nd_l_unexp_idf, nd_r_unexp_idf     - summed IDF of unexplained tokens on each side
      nd_l_logodds_max/_sum, nd_r_...    - max/sum learned log-odds of unexplained tokens on each side
      nd_l_exact_frac                    - fraction of l's tokens classified exact
      nd_l_ntok, nd_r_ntok               - token count of each side
    """
    idf = idf or {}
    logodds = logodds or {}
    n = len(l_names)
    out = {name: np.zeros(n, dtype=np.float32) for name in NAME_DIFF_FEATURES}
    for i in range(n):
        l_toks = l_names[i].split() if l_names[i] else []
        r_toks = r_names[i].split() if r_names[i] else []
        cls_l, cls_r = align_tokens(l_toks, r_toks)

        out["nd_l_ntok"][i] = len(l_toks)
        out["nd_r_ntok"][i] = len(r_toks)
        out["nd_l_exact_frac"][i] = (sum(c == "exact" for c in cls_l) / len(l_toks)) if l_toks else 0.0

        n_typo = n_abbrev = n_translit = n_legal = 0
        n_unexp_l = n_unexp_r = 0
        l_idf_mass = r_idf_mass = 0.0
        l_lo: list[float] = []
        r_lo: list[float] = []
        for side_toks, side_cls, is_left in ((l_toks, cls_l, True), (r_toks, cls_r, False)):
            for tok, c in zip(side_toks, side_cls, strict=True):
                if c == "typo":
                    n_typo += 1
                elif c == "abbrev":
                    n_abbrev += 1
                elif c == "translit":
                    n_translit += 1
                elif c == "legal":
                    n_legal += 1
                elif c == "unexplained":
                    key = _norm_token(tok)
                    lo = logodds.get(key, 0.0)
                    if is_left:
                        n_unexp_l += 1
                        l_idf_mass += idf.get(key, 1.0)
                        l_lo.append(lo)
                    else:
                        n_unexp_r += 1
                        r_idf_mass += idf.get(key, 1.0)
                        r_lo.append(lo)

        out["nd_l_unexplained"][i] = n_unexp_l
        out["nd_r_unexplained"][i] = n_unexp_r
        out["nd_typo"][i] = n_typo
        out["nd_abbrev"][i] = n_abbrev
        out["nd_translit"][i] = n_translit
        out["nd_legal_diff"][i] = n_legal
        out["nd_l_unexp_idf"][i] = l_idf_mass
        out["nd_r_unexp_idf"][i] = r_idf_mass
        out["nd_l_logodds_max"][i] = max(l_lo) if l_lo else 0.0
        out["nd_l_logodds_sum"][i] = sum(l_lo)
        out["nd_r_logodds_max"][i] = max(r_lo) if r_lo else 0.0
        out["nd_r_logodds_sum"][i] = sum(r_lo)
    return out


def _classify_number_pair(a: str, b: str) -> int:
    """Relation class for two numeric strings: 1 equal, 2 digit-drop (one is the other with one digit
    removed), 3 one-digit substitution, 4 adjacent-digit transposition, 5 small numeric difference (<=10),
    6 far apart / not comparable. Digit-drop and "small" are checked before generic substitution/
    transposition so a numerically close pair (e.g. 5124 vs 5129) reads as "small", not "substitution"."""
    if a == b:
        return 1
    if not (a.isdigit() and b.isdigit()):
        return 6
    la, lb = len(a), len(b)
    if abs(la - lb) == 1:
        longer, shorter = (a, b) if la > lb else (b, a)
        if any(longer[:k] + longer[k + 1 :] == shorter for k in range(len(longer))):
            return 2
    if abs(int(a) - int(b)) <= 10:
        return 5
    if la == lb:
        mismatches = [k for k in range(la) if a[k] != b[k]]
        if len(mismatches) == 1:
            return 3
        if len(mismatches) == 2:
            i, j = mismatches
            if j == i + 1 and a[i] == b[j] and a[j] == b[i]:
                return 4
    return 6


def number_relation_features(l_nums: list[str], r_nums: list[str]) -> dict[str, np.ndarray]:
    """Per-pair house-number relation features. l_nums[i]/r_nums[i] are space-joined numeric tokens for one
    pair (e.g. "10756 1226"); missing numbers are "". Picks the best-matching number pair (lowest, i.e. most
    explainable, relation code from _classify_number_pair, tie-broken by smallest relative difference).
    Returns float32 numpy arrays:

      nr_class         - relation class code of the best-matching pair (0 if either side has no numbers)
      nr_min_rel_diff   - |a-b| / max(a,b) of the best-matching pair (1.0 if not comparable/missing)
      nr_first_eq       - whether the first numbers on each side are equal
      nr_n_equal        - count of numbers shared between the two sides
      nr_both_present   - whether both sides have at least one number
    """
    n = len(l_nums)
    nr_class = np.zeros(n, dtype=np.float32)
    nr_min_rel_diff = np.ones(n, dtype=np.float32)
    nr_first_eq = np.zeros(n, dtype=np.float32)
    nr_n_equal = np.zeros(n, dtype=np.float32)
    nr_both_present = np.zeros(n, dtype=np.float32)
    for i in range(n):
        la = l_nums[i].split() if l_nums[i] else []
        lb = r_nums[i].split() if r_nums[i] else []
        if not (la and lb):
            nr_class[i] = 0.0
            continue
        nr_both_present[i] = 1.0
        nr_first_eq[i] = float(la[0] == lb[0])
        nr_n_equal[i] = float(len(set(la) & set(lb)))
        best_code = 7
        best_rd = 1.0
        for x in la:
            for y in lb:
                code = _classify_number_pair(x, y)
                if x.isdigit() and y.isdigit():
                    rd = abs(int(x) - int(y)) / max(int(x), int(y), 1)
                else:
                    rd = 0.0 if x == y else 1.0
                if code < best_code or (code == best_code and rd < best_rd):
                    best_code, best_rd = code, rd
        nr_class[i] = best_code
        nr_min_rel_diff[i] = best_rd
    return {
        "nr_class": nr_class,
        "nr_min_rel_diff": nr_min_rel_diff,
        "nr_first_eq": nr_first_eq,
        "nr_n_equal": nr_n_equal,
        "nr_both_present": nr_both_present,
    }


def learn_token_logodds(
    pairs: Iterable[tuple[str, str, int]], min_count: int = 20, alpha: float = 1.0
) -> dict[str, float]:
    """Learn per-token log-odds of being 'unexplained' (align_tokens) in negative vs positive pairs, i.e.
    distractor signatures like "Group" or "Ventures" that show up unmatched far more often in non-matches.

    LEAKAGE (D-004): pairs must be fit-group pairs only, and the fit group itself must be scored with a
    model fitted out-of-fold (e.g. 2-fold within fit) -- never fit and apply this on the same pairs, since a
    token's own pair would then leak its own label. Tune/holdout/test always use an out-of-fold fit-group
    table as-is.

    Returns log(P(unexplained|neg) / P(unexplained|pos)), Laplace-smoothed by alpha, for tokens occurring at
    least min_count times total across both classes; callers should default an absent token to 0.0.
    """
    pos_unexp: dict[str, int] = {}
    pos_total: dict[str, int] = {}
    neg_unexp: dict[str, int] = {}
    neg_total: dict[str, int] = {}
    for l_name, r_name, label in pairs:
        l_toks = l_name.split() if l_name else []
        r_toks = r_name.split() if r_name else []
        cls_l, cls_r = align_tokens(l_toks, r_toks)
        unexp, total = (pos_unexp, pos_total) if label else (neg_unexp, neg_total)
        for tok, c in zip(l_toks + r_toks, cls_l + cls_r, strict=True):
            t = _norm_token(tok)
            if not t:
                continue
            total[t] = total.get(t, 0) + 1
            if c == "unexplained":
                unexp[t] = unexp.get(t, 0) + 1
    out: dict[str, float] = {}
    for t in set(pos_total) | set(neg_total):
        n_pos = pos_total.get(t, 0)
        n_neg = neg_total.get(t, 0)
        if n_pos + n_neg < min_count:
            continue
        p_pos = (pos_unexp.get(t, 0) + alpha) / (n_pos + 2 * alpha)
        p_neg = (neg_unexp.get(t, 0) + alpha) / (n_neg + 2 * alpha)
        out[t] = math.log(p_neg / p_pos)
    return out


def idf_table(names: Iterable[str]) -> dict[str, float]:
    """Token IDF (log(N / document_frequency)) over normalized name tokens; N = number of names seen."""
    df: dict[str, int] = {}
    n = 0
    for name in names:
        n += 1
        for t in {_norm_token(tok) for tok in name.split()}:
            if t:
                df[t] = df.get(t, 0) + 1
    return {t: math.log(n / c) for t, c in df.items()} if n else {}
