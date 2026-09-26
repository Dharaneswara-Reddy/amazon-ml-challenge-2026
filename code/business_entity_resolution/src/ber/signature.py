"""Distractor-signature features: which edit operations turn an S1 record into a right record (D-021, v3).

The dataset's distractors were generated from a real S1 "parent" by (a) a house/address number nudged UP by an offset
from a small fixed set and (b) a name edit drawn from small vocabularies (legal-form swap, inserted word), and then
the same copy noise as true matches was applied (research_notes/v3/F_generator_signature.md). True copies instead
show typos, abbreviations, truncation, dropped words and digit drops. This module diffs a pair at token level and
turns the operations into features; the edit-token vocabularies are learned from labelled FIT-group candidate pairs
only (out-of-fold for the fit folds). All rules are hand-written; no external data.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache

import numpy as np
from rapidfuzz.distance import Levenshtein
from rapidfuzz.fuzz import ratio

TOK = re.compile(r"[a-z0-9]+")
DIG = re.compile(r"\d+")
SWAP = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s", "6": "g", "8": "b", "4": "a", "7": "t"})
LEGAL = {"llc", "inc", "ltd", "limited", "pvt", "private", "corp", "corporation", "co", "company", "lp", "llp",
         "pllc", "pc", "plc", "pa", "lllp", "incorporated", "public", "l", "p", "c", "sarl", "sas", "sa", "eurl", "sci"}


@lru_cache(maxsize=1_000_000)
def fold(s: str) -> str:
    """Lowercase, accent-stripped text; non-Latin letters are kept (they are dropped by TOK later)."""
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch)).lower()


def nonlatin(s: str) -> bool:
    """True if the string contains a letter outside Latin/ASCII blocks (Indic script etc.)."""
    return any(ord(ch) > 0x24F and ch.isalpha() for ch in s)


def name_tokens(s: str) -> list[str]:
    """Folded name tokens with letter/digit swaps repaired on mixed tokens ('br0thers' -> 'brothers')."""
    out = []
    for t in TOK.findall(fold(s)):
        if not t.isdigit() and any(c.isdigit() for c in t) and any(c.isalpha() for c in t):
            t = t.translate(SWAP)
        out.append(t)
    return out


def is_abbrev(short: str, long: str) -> bool:
    """short is an abbreviation of long: same first letter and an in-order subsequence (pvt/private, st/street)."""
    if not short or not long or len(short) >= len(long) or short[0] != long[0]:
        return False
    it = iter(long)
    return all(c in it for c in short)


def classify_tok(x: str, y: str) -> str | None:
    """Relation of a deleted L token x and an inserted R token y: typo / abbrev / None."""
    if is_abbrev(y, x) or is_abbrev(x, y):
        return "abbrev"
    if min(len(x), len(y)) >= 3 and ratio(x, y) >= 75:
        return "typo"
    if len(x) >= 4 and len(y) >= 4 and (x.startswith(y) or y.startswith(x)):
        return "trunc"
    return None


def token_diff(a: list[str], b: list[str]) -> dict:
    """Multiset diff of token lists a (L) -> b (R), pairing leftovers into typo/abbrev/trunc/replaced.

    Returns counts plus the lists of inserted / deleted / replaced tokens (with position of inserts in b).
    """
    ca, cb = Counter(a), Counter(b)
    common = ca & cb
    dels = list((ca - common).elements())
    ins = list((cb - common).elements())
    res = {"typo": 0, "abbrev": 0, "trunc": 0, "replaced": [], "inserted": [], "deleted": []}
    # greedy pairing of fuzzy relations
    for x in list(dels):
        best, bc = None, None
        for y in ins:
            c = classify_tok(x, y)
            if c and (best is None or ratio(x, y) > ratio(x, best)):
                best, bc = y, c
        if best is not None:
            res[bc] += 1
            dels.remove(x)
            ins.remove(best)
    # concatenation: an inserted token equal to the join of >=2 L tokens
    joined = "".join(a)
    for y in list(ins):
        if len(y) >= 6 and (y in joined or y.replace("com", "") in joined) and len(dels) >= 1:
            parts = [x for x in dels if x in y]
            if parts:
                for x in parts:
                    dels.remove(x)
                ins.remove(y)
                res["concat"] = res.get("concat", 0) + 1
    # remaining pairs in order -> replaced; leftovers -> inserted / deleted
    k = min(len(dels), len(ins))
    res["replaced"] = list(zip(dels[:k], ins[:k]))
    rest_ins, rest_del = ins[k:], dels[k:]
    for y in rest_ins:
        pos = b.index(y)
        res["inserted"].append((y, 0 if pos == 0 else (2 if pos == len(b) - 1 else 1), ca[y] > 0))
    res["deleted"] = rest_del
    # reordering of common tokens
    ka = [t for t in a if common[t]]
    kb = [t for t in b if common[t]]
    res["reorder"] = int(ka != kb and sorted(ka) == sorted(kb))
    res["n_common"] = sum(common.values())
    return res


def num_rel(l: str, r: str) -> tuple[str, int]:
    """Relation of an unmatched L digit run l and R digit run r: (class, signed diff r-l or 0)."""
    ls, rs = l.lstrip("0") or "0", r.lstrip("0") or "0"
    if ls == rs:
        return "pad", 0
    if len(ls) == len(rs) + 1 and ls.startswith(rs):
        return "trunc_last", 0
    if len(ls) == len(rs) + 1 and ls[1:] == rs:
        return "drop_first", 0
    if len(ls) == len(rs) + 1 and any(ls[:i] + ls[i + 1:] == rs for i in range(len(ls))):
        return "drop_inner", 0
    if len(rs) == len(ls) + 1 and any(rs[:i] + rs[i + 1:] == ls for i in range(len(rs))):
        return "add_digit", int(rs) - int(ls) if len(rs) < 12 else 0
    if len(ls) == len(rs) and len(ls) > 1 and sorted(ls) == sorted(rs) and Levenshtein.distance(ls, rs) == 2:
        return "transpose", int(rs) - int(ls) if len(rs) < 12 else 0
    if len(ls) < 12 and len(rs) < 12:
        return "offset", int(rs) - int(ls)
    return "other", 0


def number_diff(la: str, ra: str) -> dict:
    """Digit-run diff of two addresses: equal runs removed, the rest paired in order and classified."""
    ln = DIG.findall(la)
    rn = DIG.findall(ra)
    lk = [x.lstrip("0") or "0" for x in ln]
    rk = [x.lstrip("0") or "0" for x in rn]
    cl, cr = Counter(lk), Counter(rk)
    com = cl & cr
    lrest = list((cl - com).elements())
    rrest = list((cr - com).elements())
    # keep L order for pairing
    lrest = [x for x in dict.fromkeys(lk) for _ in range((cl - com)[x])]
    rrest = [x for x in dict.fromkeys(rk) for _ in range((cr - com)[x])]
    pairs = []
    rr = list(rrest)
    for x in lrest:
        if not rr:
            break
        # prefer same-length partner with the smallest |diff|, else the first remaining
        best = min(rr, key=lambda y: (abs(len(y) - len(x)), abs(int(y[:11]) - int(x[:11]))))
        pairs.append((x, best, lk.index(x), rk.index(best)))
        rr.remove(best)
    rels = [(num_rel(x, y), i, j) for x, y, i, j in pairs]
    return {"nl": len(ln), "nr": len(rn), "n_eq": sum(com.values()), "pairs": rels,
            "l_only": len(lrest) - len(pairs), "r_only": len(rr), "pad": sum(1 for x in rn if x.startswith("0") and len(x) > 1)}


def pair_ops(ln: str, la: str, rn: str, ra: str) -> dict:
    """All diff information for one (L, R) pair of raw names and addresses."""
    rn = rn or ""
    ra = ra or ""
    d = {"r_name_nonlatin": nonlatin(rn), "r_addr_empty": ra.strip() == "", "name_case_only": False}
    d["name_exact"] = rn == ln
    d["name_case_only"] = rn != ln and fold(rn) == fold(ln)
    a, b = name_tokens(ln), name_tokens(rn)
    d["name"] = token_diff(a, b) if not d["r_name_nonlatin"] else None
    d["name_ntok"] = (len(a), len(b))
    if d["r_addr_empty"]:
        d["addr"] = None
        d["num"] = None
    else:
        fa, fr = fold(la), fold(ra)
        wa = [t for t in TOK.findall(fa) if not t.isdigit()]
        wr = [t for t in TOK.findall(fr) if not t.isdigit()]
        d["addr"] = token_diff(wa, wr)
        d["num"] = number_diff(fa, fr)
    return d


# Distractor number offsets measured on train (F finding 1: uniform draw from {+1..+5,+7,+9,+11,+13,+21});
# +1/+2 are kept apart because true copies also make +-1/+-2 typos.
S_OFF = frozenset({3, 4, 5, 7, 9, 11, 13, 21})
VOCAB_MIN_COUNT = 10

SIG_FEATURES = [
    "sg_r_nonlatin", "sg_name_exact", "sg_case_only", "sg_r_addr_empty", "sg_l_ntok", "sg_r_ntok",
    "sg_n_typo", "sg_n_abbrev", "sg_n_trunc", "sg_n_concat", "sg_n_reorder", "sg_n_repl", "sg_n_ins", "sg_n_del",
    "sg_n_common", "sg_ins_lo_max", "sg_ins_lo_sum", "sg_repl_lo_max", "sg_repl_lo_sum", "sg_rtgt_lo_max",
    "sg_del_lo_min", "sg_edit_lo_max", "sg_a_typo", "sg_a_abbrev", "sg_a_repl", "sg_a_ins", "sg_a_del",
    "sg_a_ins_lo_max", "sg_a_del_lo_max", "sg_num_nl", "sg_num_nr", "sg_num_eq", "sg_num_l_only", "sg_num_r_only",
    "sg_num_changed", "sg_off_in_s", "sg_off_p12", "sg_off_m12", "sg_off_other", "sg_num_trunc", "sg_num_add_digit",
    "sg_num_transpose", "sg_first_off", "sg_first_off_idx", "sg_num_pad", "sg_sole_off_in_s",
]


def edit_tokens(d: dict) -> dict[str, list[str]]:
    """Edit tokens of one pair_ops result, keyed by vocabulary: ins, del, repl ('x>y'), rtgt, ains, adel."""
    out: dict[str, list[str]] = {k: [] for k in ("ins", "del", "repl", "rtgt", "ains", "adel")}
    n, a = d["name"], d["addr"]
    if n is not None:
        out["ins"] = [w for w, _, _ in n["inserted"]]
        out["del"] = list(n["deleted"])
        out["repl"] = [f"{x}>{y}" for x, y in n["replaced"]]
        out["rtgt"] = [y for _, y in n["replaced"]]
    if a is not None:
        out["ains"] = [w for w, _, _ in a["inserted"]]
        out["adel"] = list(a["deleted"])
    return out


def learn_vocab(rows: Iterable[tuple[str, str, str, str, int]], min_count: int = VOCAB_MIN_COUNT) -> dict:
    """Smoothed log-odds (non-match vs match) of edit tokens from labelled candidate pairs.

    rows: (l_name, l_addr, r_name, r_addr, label) of FIT-group candidate pairs. Returns
    {vocab: {token: log((c_nonmatch + 1) / (c_match + 1)) + log(n_match / n_nonmatch)}} for tokens seen >= min_count.
    """
    cnt = {k: (Counter(), Counter()) for k in ("ins", "del", "repl", "rtgt", "ains", "adel")}
    n_pos = n_neg = 0
    for ln, la, rn, ra, label in rows:
        toks = edit_tokens(pair_ops(ln or "", la or "", rn or "", ra or ""))
        side = 1 if label else 0
        if label:
            n_pos += 1
        else:
            n_neg += 1
        for k, ws in toks.items():
            cnt[k][side].update(ws)
    prior = math.log(max(n_pos, 1) / max(n_neg, 1))
    vocab = {}
    for k, (neg, pos) in cnt.items():
        vocab[k] = {
            w: round(math.log((neg[w] + 1) / (pos[w] + 1)) + prior, 3)
            for w in set(neg) | set(pos)
            if neg[w] + pos[w] >= min_count
        }
    return vocab


def _row_features(ln: str, la: str, rn: str, ra: str, voc: Mapping[str, Mapping[str, float]]) -> list[float]:
    """Feature vector (SIG_FEATURES order) of one pair of names and addresses."""
    d = pair_ops(ln or "", la or "", rn or "", ra or "")
    nan = float("nan")
    n = d["name"]
    if n is None:
        nf = [nan] * 16
    else:
        ins = [voc["ins"].get(w, 0.0) for w, _, _ in n["inserted"]]
        rep = [voc["repl"].get(f"{x}>{y}", voc["rtgt"].get(y, 0.0)) for x, y in n["replaced"]]
        rtg = [voc["rtgt"].get(y, 0.0) for _, y in n["replaced"]]
        dl = [voc["del"].get(w, 0.0) for w in n["deleted"]]
        nf = [n["typo"], n["abbrev"], n["trunc"], n.get("concat", 0), n["reorder"], len(n["replaced"]),
              len(n["inserted"]), len(n["deleted"]), n["n_common"], max(ins, default=0.0), sum(ins),
              max(rep, default=0.0), sum(rep), max(rtg, default=0.0), min(dl, default=0.0),
              max(ins + rep, default=0.0)]
    a = d["addr"]
    if a is None:
        af = [nan] * 7
        mf = [nan] * 17
    else:
        ai = [voc["ains"].get(w, 0.0) for w, _, _ in a["inserted"]]
        ad = [voc["adel"].get(w, 0.0) for w in a["deleted"]]
        af = [a["typo"], a["abbrev"], len(a["replaced"]), len(a["inserted"]), len(a["deleted"]),
              max(ai, default=0.0), max(ad, default=0.0)]
        m = d["num"]
        rel = [c for (c, _), _, _ in m["pairs"]]
        off = [v for (c, v), _, _ in m["pairs"] if c == "offset"]
        first = next(((v, i) for (c, v), i, _ in m["pairs"] if c in ("offset", "add_digit", "transpose")), (nan, -1))
        mf = [m["nl"], m["nr"], m["n_eq"], m["l_only"], m["r_only"], len(rel),
              sum(v in S_OFF for v in off), sum(v in (1, 2) for v in off), sum(v in (-1, -2) for v in off),
              sum(v not in S_OFF and v not in (1, 2, -1, -2) for v in off),
              rel.count("trunc_last") + rel.count("drop_first") + rel.count("drop_inner"), rel.count("add_digit"),
              rel.count("transpose"), max(-1000.0, min(1000.0, first[0])) if first[0] == first[0] else nan, first[1],
              m["pad"], float(len(rel) == 1 and len(off) == 1 and off[0] in S_OFF)]
    return [float(d["r_name_nonlatin"]), float(d["name_exact"]), float(d["name_case_only"]),
            float(d["r_addr_empty"]), d["name_ntok"][0], d["name_ntok"][1], *nf, *af, *mf]


def signature_features(
    l_name: Sequence[str],
    l_addr: Sequence[str],
    r_name: Sequence[str],
    r_addr: Sequence[str],
    vocab: Mapping[str, Mapping[str, float]] | None,
) -> dict[str, np.ndarray]:
    """SIG_FEATURES for aligned lists of left/right names and addresses; an empty vocab gives 0 log-odds."""
    voc = vocab or {}
    voc = {k: voc.get(k, {}) for k in ("ins", "del", "repl", "rtgt", "ains", "adel")}
    rows = [_row_features(*x, voc) for x in zip(l_name, l_addr, r_name, r_addr, strict=True)]
    arr = np.asarray(rows, dtype=np.float32).reshape(len(rows), len(SIG_FEATURES))
    return {name: arr[:, j] for j, name in enumerate(SIG_FEATURES)}
