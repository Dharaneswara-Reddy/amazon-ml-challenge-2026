"""Raw-text diff between an S1 record (L) and a right record (R): which edit operations turn L into R.

Used both for the generator analysis (op tables, vocabularies) and for the pair features of the holdout test.
All rules are hand-written (no external data): accent folding, letter/digit swap repair, abbreviation = the short
token's letters are an in-order subsequence of the long token starting with the same letter, typo = rapidfuzz
ratio >= 75 on tokens of length >= 3.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from functools import lru_cache

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
