"""Text normalization for business names and addresses.

Every rule here was derived from, and measured on, samples of the training data (see
docs/normalization_findings.md). No external data is used: all maps are hand-written.

Design goals:
- script-agnostic: any non-Latin script is romanized with ``anyascii`` (ISC license), with one fix for
  Indic anusvara; Latin accents are folded, so French/US/Indian text share one code path;
- precision-safe: only tokens proven to be noise (legal forms, honorifics, alias markers, ID tags) are
  removed, and only in ``name_core``; ``name_norm`` keeps every content word;
- country-agnostic: country is never used to choose a rule; state maps are keyed by the component text.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import re
import unicodedata
from collections import Counter, defaultdict, deque
from collections.abc import Iterable
from functools import lru_cache

import pyarrow as pa
import pyarrow.parquet as pq
from anyascii import anyascii
from rapidfuzz.distance import JaroWinkler

# --------------------------------------------------------------------------------------------------
# Transliteration
# --------------------------------------------------------------------------------------------------

# Anusvara / candrabindu signs of every Indic block. anyascii renders them as "m" (e.g. "kansaltents"
# becomes "kmsltemts"); a nasal before a non-labial consonant is pronounced "n" in the English words
# that dominate these names, which raised mean token_set_ratio on Indic<->Latin pairs (findings doc).
_ANUSVARA = "ँंঁংਁਂੰઁંଁଂஂఁంಁಂം"
_NASAL_MARK = "\x01"
_ANUSVARA_TABLE = str.maketrans({c: _NASAL_MARK for c in _ANUSVARA})
_NASAL_RE = re.compile(_NASAL_MARK + r"(?=[pbm])")
# Zero-width joiners appear inside Indic words ("ఎల్‌ఎల్‌పీ"); drop them before romanizing.
_ZW_TABLE = str.maketrans({"​": None, "‌": None, "‍": None, "﻿": None})


@lru_cache(maxsize=262_144)
def _romanize_word(word: str) -> str:
    """Romanize one whitespace-free non-ASCII chunk to lowercase ASCII (cached: words repeat a lot)."""
    t = word.translate(_ANUSVARA_TABLE)
    t = anyascii(t).lower()
    t = _NASAL_RE.sub("m", t).replace(_NASAL_MARK, "n")
    return t.replace("'", "")  # anyascii marks nukta/aspiration with an apostrophe ("ph'aundesn")


def to_ascii(raw: str | None) -> str:
    """NFKC-normalize, romanize any script and fold accents; returns lowercase ASCII ('' for None)."""
    if not raw:
        return ""
    t = unicodedata.normalize("NFKC", raw)
    if t.isascii():
        return t.lower()
    t = t.translate(_ZW_TABLE).replace("°", "o")  # "N°" -> "No" (anyascii would give "deg")
    return " ".join(w.lower() if w.isascii() else _romanize_word(w) for w in t.split())


# Unicode ranges used by detect_script. Latin covers ASCII letters plus Latin-1/Extended accents.
def _char_script(ch: str) -> str | None:
    """Classify one alphabetic character as 'latin', 'devanagari' or 'other'; None for non-letters."""
    if not ch.isalpha():
        return None
    o = ord(ch)
    if o < 0x0250 or 0x1E00 <= o <= 0x1EFF:
        return "latin"
    if 0x0900 <= o <= 0x097F:
        return "devanagari"
    return "other"


def detect_script(raw: str | None) -> str:
    """Return the script of a raw string: 'latin' | 'devanagari' | 'other' | 'mixed'.

    Strings without letters (or empty) count as 'latin'.
    """
    if not raw or raw.isascii():
        return "latin"
    found = set()
    for ch in raw:
        s = _char_script(ch)
        if s:
            found.add(s)
            if len(found) > 1:
                return "mixed"
    return found.pop() if found else "latin"


# --------------------------------------------------------------------------------------------------
# Shared cleaning helpers
# --------------------------------------------------------------------------------------------------

_DOTTED_ACRONYM_RE = re.compile(r"\b(?:[a-z]\.){2,}(?:[a-z]\b)?")  # l.l.c. / p.c. / s.a.r.l / m.d
_AMP_RE = re.compile(r"\s*[&+]\s*")
_POSSESSIVE_RE = re.compile(r"(?<=[a-z])'s\b")  # "lisi's" -> "lisis" (keeps the word whole)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_SINGLE_LETTER_RUN_RE = re.compile(r"\b(?:[a-z] ){1,}[a-z]\b")  # "j k enterprises" -> "jk enterprises"
_ORDINAL_RE = re.compile(r"^\d+(?:st|nd|rd|th|er|e|eme|bis|ter)$")
# Digit-for-letter swaps seen in S2/S3 names ("br0thers", "techno1ogy", "5ervice", "8allard"), applied
# only to tokens that mix letters and digits, never to ordinals like "3rd" or pure numbers.
_LEET_TABLE = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"})


def _undot(m: re.Match) -> str:
    """Regex callback: 'l.l.c.' -> 'llc '."""
    return m.group(0).replace(".", "") + " "


def _join_single_letters(text: str) -> str:
    """Join runs of single-letter tokens into one token ("l l c" -> "llc", "j k" -> "jk")."""
    return _SINGLE_LETTER_RUN_RE.sub(lambda m: m.group(0).replace(" ", ""), text)


def _dedupe_consecutive(tokens: list[str]) -> list[str]:
    """Drop immediately repeated tokens ("corp corp", "p.c. p.c." injected by the noise generator)."""
    out: list[str] = []
    for t in tokens:
        if not out or out[-1] != t:
            out.append(t)
    return out


# --------------------------------------------------------------------------------------------------
# Business names
# --------------------------------------------------------------------------------------------------

# Alias markers: in 764/764 sampled true pairs with such a marker the S1 name was the part AFTER it
# ("Arcvio DBA: Hanson Bitwise Inc"); the part before is a random trade name, so it is discarded.
_ALIAS_RE = re.compile(
    r"^.*?\b(?:d\s*/\s*b\s*/\s*a|d\.b\.a\.?|dba|a\s*/\s*k\s*/\s*a|a\.k\.a\.?|aka|f\s*/\s*k\s*/\s*a|fka|"
    r"t\s*/\s*a|trading\s+as|doing\s+business\s+as|also\s+known\s+as|operating\s+as|"
    r"formerly(?:\s+known\s+as)?)\s*:?\s+",
    re.IGNORECASE,
)
_ID_TAG_RE = re.compile(r"\(\s*id\s*:?\s*\d+\s*\)", re.IGNORECASE)  # "(ID: 40888)"
_PIPE_SPLIT_RE = re.compile(r"\s*\|\s*")
_URL_PREFIX_RE = re.compile(r"\b(?:https?://)?www\.", re.IGNORECASE)
_TLD_RE = re.compile(r"\.(?:c0m|com|co\.in|in|net|org|co|fr|biz|info|us)\b", re.IGNORECASE)
_GLUED_COM_RE = re.compile(r"(?<=[a-z])Com\b")  # "DennisfamilypracticeCom"

# Canonical forms (short) for legal/generic words. Includes romanized Indic spellings measured on
# Indic<->Latin true pairs (praivet/piraivet/praibhet -> private, limitet/limirrd -> limited, elelpi -> llp)
# and French legal forms (the test set adds France).
NAME_TOKEN_MAP: dict[str, str] = {
    "private": "pvt",
    "pvt": "pvt",
    "praivet": "pvt",
    "piraivet": "pvt",
    "praibhet": "pvt",
    "praivrr": "pvt",
    "prvt": "pvt",
    "pte": "pvt",
    "limited": "ltd",
    "ltd": "ltd",
    "limitet": "ltd",
    "limirrd": "ltd",
    "lintid": "ltd",
    "ltda": "ltd",
    "incorporated": "inc",
    "inc": "inc",
    "incorporation": "inc",
    "corporation": "corp",
    "corp": "corp",
    "corpn": "corp",
    "company": "co",
    "co": "co",
    "cos": "co",
    "companies": "co",
    "elelpi": "llp",
    "ailailpi": "llp",
    "elelsi": "llc",
    "and": "and",
    "et": "and",
    "und": "and",
    "centre": "center",
    "ctr": "center",
    "intl": "international",
    "bros": "brothers",
    "bro": "brothers",
    "mfg": "manufacturing",
    "mfrs": "manufacturers",
    "assoc": "associates",
    "assocs": "associates",
    "svc": "service",
    "svcs": "services",
    "tech": "tech",
    "technologies": "technologies",
    "dept": "department",
    "univ": "university",
    "hosp": "hospital",
    # French
    "cie": "compagnie",
    "ets": "etablissements",
    "etab": "etablissements",
}
# Multi-token phrases (applied on the space-joined string, word bounded).
_NAME_PHRASES = {
    "pra li": "pvt ltd",  # Devanagari "प्रा. लि."
    "m s": "ms",  # "M/s" after punctuation removal
    "pr ivate": "pvt",
    "li mited": "ltd",
}
_NAME_PHRASE_RE = re.compile(r"\b(?:" + "|".join(map(re.escape, _NAME_PHRASES)) + r")\b")
# Legal forms removed by name_core (canonical spellings, i.e. after NAME_TOKEN_MAP).
LEGAL_TOKENS = frozenset(
    "pvt ltd inc corp llc llp lp pllc pc pa plc co lllp ltda gmbh ag bv nv "
    # French legal forms
    "sarl sas sa eurl sasu sci snc ei eirl selarl scop scm sca gie scs sel".split()
)
# Honorific / filler tokens that the noise generator prepends ("Smt ...", "Dr ...", "M/s ...", "The ...").
# Removed by name_core only when they lead the name.
LEADING_FILLER = frozenset("the ms mr mrs smt sri shri shree dr m s messrs".split())


def _fix_leet(tok: str) -> str:
    """Undo digit-for-letter swaps inside mixed alphanumeric tokens; leave numbers/ordinals alone."""
    if tok.isalpha() or tok.isdigit() or _ORDINAL_RE.match(tok):
        return tok
    if not any(c.isalpha() for c in tok):
        return tok
    return tok.translate(_LEET_TABLE)


def normalize_name(raw: str | None) -> str:
    """Normalize a business name to lowercase ASCII tokens with canonical legal forms.

    Steps: strip "(ID: n)" tags, keep the first "|"-separated part, keep the text after alias markers
    (dba/aka/t/a/formerly...), romanize + fold accents, drop URL scaffolding (www., .com), collapse dotted
    acronyms, '&'/'+' -> 'and', punctuation -> space, join single-letter runs, fix digit swaps, map
    abbreviations, drop consecutive duplicates. Input: raw string or None. Output: str ('' if empty).
    Results are memoized per distinct raw string.
    """
    return _normalize_name_cached(raw) if raw else ""


@lru_cache(maxsize=131_072)
def _normalize_name_cached(raw: str) -> str:
    """Uncached body of normalize_name (see its docstring)."""
    t = raw
    if "(" in t:
        t = _ID_TAG_RE.sub(" ", t)
    if "|" in t:
        parts = [p for p in _PIPE_SPLIT_RE.split(t) if p.strip()]
        t = parts[0] if parts else ""
    if "Com" in t:
        t = _GLUED_COM_RE.sub("", t)
    t = _ALIAS_RE.sub("", t, count=1) or t
    t = to_ascii(t)
    if "." in t:
        if "www." in t:
            t = _URL_PREFIX_RE.sub(" ", t)
        t = _TLD_RE.sub(" ", t)
        t = _DOTTED_ACRONYM_RE.sub(_undot, t)
    if "&" in t or "+" in t:
        t = _AMP_RE.sub(" and ", t)
    if "'" in t:
        t = _POSSESSIVE_RE.sub("s", t)
    t = _NON_ALNUM_RE.sub(" ", t).strip()
    if not t:
        return ""
    t = _join_single_letters(t)
    t = _NAME_PHRASE_RE.sub(lambda m: _NAME_PHRASES[m.group(0)], t)
    toks = [NAME_TOKEN_MAP.get(w, w) for w in (_fix_leet(w) for w in t.split())]
    return " ".join(_dedupe_consecutive(toks))


def name_core(name_norm: str) -> str:
    """Remove legal-form tokens anywhere and filler/honorific tokens at the start of a normalized name.

    Keeps token order. Never returns '' for a non-empty input: falls back to ``name_norm``.
    """
    toks = name_norm.split()
    i = 0
    while i < len(toks) - 1 and toks[i] in LEADING_FILLER:
        i += 1
    core = [t for t in toks[i:] if t not in LEGAL_TOKENS]
    return " ".join(core) if core else name_norm


# Phonetic consonant skeleton. Indic names are English words spelled phonetically ("ड्रीम पावर" ->
# "drim pavr" vs "dream power"); vowels and c/k, w/v, ph/f, z/j spellings are exactly where romanized and
# Latin spellings disagree. On 2,711 Indic<->Latin true pairs mean ratio of name_core rose 73.8 -> 92.1
# with the skeleton, while hard negatives moved only 58.4 -> 59.2 (findings doc).
_SKEL_RULES = [
    (re.compile(r"ph"), "f"),
    (re.compile(r"c(?=[eiy])"), "s"),
    (re.compile(r"[cq]"), "k"),
    (re.compile(r"w"), "v"),
    (re.compile(r"z"), "j"),
    (re.compile(r"x"), "ks"),
    (re.compile(r"(?<=[a-z])h"), ""),
    (re.compile(r"(?<=\w)[aeiouy]"), ""),
    (re.compile(r"(.)\1+"), r"\1"),
]


@lru_cache(maxsize=262_144)
def _skeleton_word(w: str) -> str:
    """Consonant skeleton of one lowercase token (numbers unchanged)."""
    if w.isdigit():
        return w
    for rx, rep in _SKEL_RULES:
        w = rx.sub(rep, w)
    return w


def name_skeleton(name: str) -> str:
    """Phonetic consonant skeleton of a normalized name (use on name_core): keeps first letter of each
    token, drops later vowels and non-initial 'h', unifies c/k/q, w/v, ph/f, z/j, collapses doubles."""
    return " ".join(_skeleton_word(w) for w in name.split())


# --------------------------------------------------------------------------------------------------
# Name decomposition v2: legal form / type words / learned transliteration map (DESIGN_V2 S1; D-012, D-013)
# --------------------------------------------------------------------------------------------------

# Canonical legal-form string for a *set* of LEGAL_TOKENS found in a name, so word-order shuffles
# ("ltd pvt" vs "pvt ltd") and single-token forms both resolve to one label. Keys are frozensets of the
# canonical spellings already produced by NAME_TOKEN_MAP (see normalize_name), so this runs on name_norm.
LEGAL_FORM_CANON: dict[frozenset[str], str] = {
    frozenset({"pvt", "ltd"}): "private limited",
    frozenset({"ltd"}): "limited",
    frozenset({"pvt"}): "private",
    frozenset({"llp"}): "llp",
    frozenset({"lp"}): "lp",
    frozenset({"inc"}): "inc",
    frozenset({"llc"}): "llc",
    frozenset({"corp"}): "corp",
    frozenset({"pllc"}): "pllc",
    frozenset({"pc"}): "pc",
    frozenset({"pa"}): "pa",
    frozenset({"plc"}): "plc",
    frozenset({"co"}): "co",
    frozenset({"lllp"}): "lllp",
    frozenset({"ltda"}): "ltda",
    frozenset({"gmbh"}): "gmbh",
    frozenset({"ag"}): "ag",
    frozenset({"bv"}): "bv",
    frozenset({"nv"}): "nv",
    # French
    frozenset({"sarl"}): "sarl",
    frozenset({"sas"}): "sas",
    frozenset({"sa"}): "sa",
    frozenset({"eurl"}): "eurl",
    frozenset({"sasu"}): "sasu",
    frozenset({"sci"}): "sci",
    frozenset({"snc"}): "snc",
    frozenset({"ei"}): "ei",
    frozenset({"eirl"}): "eirl",
    frozenset({"selarl"}): "selarl",
    frozenset({"scop"}): "scop",
    frozenset({"scm"}): "scm",
    frozenset({"sca"}): "sca",
    frozenset({"gie"}): "gie",
    frozenset({"scs"}): "scs",
    frozenset({"sel"}): "sel",
}


def split_legal_form(name_norm: str) -> tuple[str, str]:
    """Split a normalized name into (name without legal-form tokens, canonical legal form).

    Scans every token against LEGAL_TOKENS (already canonical spellings, since normalize_name ran
    NAME_TOKEN_MAP first) and removes them regardless of position, so a shuffled suffix like "ltd pvt"
    resolves the same as "pvt ltd". The set of legal tokens found is mapped to one label via
    LEGAL_FORM_CANON (e.g. {"pvt","ltd"} -> "private limited"); an unlisted combination falls back to its
    tokens sorted and space-joined, so the function never raises on an unseen mixture. Returns
    (name_norm, '') when no legal token is present. Input/output: str.
    """
    toks = name_norm.split()
    found = frozenset(t for t in toks if t in LEGAL_TOKENS)
    if not found:
        return name_norm, ""
    kept = [t for t in toks if t not in LEGAL_TOKENS]
    legal = LEGAL_FORM_CANON.get(found, " ".join(sorted(found)))
    return " ".join(kept), legal


# Generic business-type words: a hand-written seed (measured on true pairs, docs/normalization_findings.md
# notes distractors add exactly these words). Kept as a *parameter default* on type_words_of, not baked
# in, so a data-learned vocabulary (tokens over-represented on one side of negative pairs) can replace it
# later without touching call sites.
TYPE_WORDS: frozenset[str] = frozenset(
    """
    group ventures enterprise enterprises company services service solutions overseas industries industry
    holdings holding trading traders international associates consultancy consultants technologies tech
    systems global agency agencies partners network networks resources logistics exports imports
    distributors distribution marketing management consulting innovations works
    """.split()
)


def type_words_of(name_core: str, vocab: frozenset[str] = TYPE_WORDS) -> str:
    """Return the generic business-type words present in ``name_core``, sorted and space-joined.

    ``vocab`` defaults to the hand-written TYPE_WORDS seed but is a parameter so a data-learned vocabulary
    can be substituted. Does not remove the words from name_core (D-013 / findings.md: they stay, since
    stripping them hurt precision) -- this only labels which ones are present, as a separate feature.
    Input: a name_core string. Output: '' if none of ``vocab`` appears.
    """
    if not name_core:
        return ""
    toks = set(name_core.split())
    return " ".join(sorted(toks & vocab))


def _ngram_units(tokens: list[str]) -> list[str]:
    """Unigrams plus adjacent bigrams of a token list, so a one-word abbreviation on one side (e.g. "pra
    li" transliterated as two tokens) can align to a two-word canonical phrase on the other ("private
    limited") and vice versa."""
    units = list(tokens)
    units += [f"{a} {b}" for a, b in zip(tokens, tokens[1:], strict=False)]
    return units


def _compile_phrase_pattern(token_map: dict[str, str]) -> re.Pattern:
    """Compile one longest-key-first, word-bounded alternation regex for apply_token_map.

    Built once (per token map) and reused across a whole batch -- token_map can have thousands of
    entries, and compiling per row would blow the throughput budget.
    """
    keys = sorted(token_map, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(k) for k in keys) + r")\b")


def learn_token_map(
    pairs: Iterable[tuple[str, str]],
    min_support: int = 50,
    min_precision: float = 0.9,
    align_threshold: float = 0.55,
) -> dict[str, str]:
    """Learn a right-name-token/bigram -> left-name-phrase map from true pairs (D-012).

    Input: (s1_name_norm, r_name_norm) pairs for TRUE matches where the right name came from a non-Latin
    script and was transliterated (e.g. right="pra li", left="private limited"). For every pair, each
    unigram/bigram of the right name is aligned to whichever unigram/bigram of the left name has the
    highest rapidfuzz JaroWinkler similarity between their phonetic skeletons (``name_skeleton``, which is
    what survives transliteration drift -- see D-016); an alignment below ``align_threshold`` is dropped.
    Counts, per right unit, how often it aligns to each left unit across all pairs. Keeps a mapping only
    when its support (count) is >= min_support and its precision (share of that right unit's aligned
    occurrences going to this particular left unit) is >= min_precision, so rare or inconsistent
    alignments are not kept. No external dictionary is used -- only the provided train pairs (D-015).
    Returns a JSON-serializable {right_unit: left_unit} dict, suitable for apply_token_map / normalize_parquet.
    """
    counts: dict[str, Counter] = defaultdict(Counter)
    totals: Counter = Counter()
    for left, right in pairs:
        if not left or not right:
            continue
        l_units = _ngram_units(left.split())
        r_units = _ngram_units(right.split())
        if not l_units or not r_units:
            continue
        l_skel = {u: name_skeleton(u) for u in l_units}
        for r in r_units:
            r_skel = name_skeleton(r)
            best_unit, best_sim = "", 0.0
            for u, u_skel in l_skel.items():
                sim = JaroWinkler.normalized_similarity(r_skel, u_skel)
                if sim > best_sim:
                    best_unit, best_sim = u, sim
            if best_sim >= align_threshold:
                counts[r][best_unit] += 1
                totals[r] += 1
    result: dict[str, str] = {}
    for r, cnts in counts.items():
        left_unit, support = cnts.most_common(1)[0]
        if support >= min_support and support / totals[r] >= min_precision:
            result[r] = left_unit
    return result


def apply_token_map(name_norm: str, token_map: dict[str, str] | None, _pattern: re.Pattern | None = None) -> str:
    """Replace learned right-side phrases with their canonical left-side form (D-012).

    Longest-match-first, whole-word replacement (a multi-word key wins over a shorter key that is its
    prefix). ``_pattern`` lets a caller reuse a pre-compiled pattern across many rows (normalize_batch
    does this); when omitted, one is compiled for this call. Returns name_norm unchanged if token_map is
    empty/None or name_norm is empty.
    """
    if not token_map or not name_norm:
        return name_norm
    pattern = _pattern or _compile_phrase_pattern(token_map)
    return pattern.sub(lambda m: token_map[m.group(0)], name_norm)


# --------------------------------------------------------------------------------------------------
# Addresses
# --------------------------------------------------------------------------------------------------

# Placeholder components as they look after cleaning ("<NULL>" -> "null", "N/A" -> "n a").
_PLACEHOLDER_COMPONENTS = frozenset({"null", "n a", "na", "none", "unknown", "nan", "nil", ""})
_HOUSE_MARK_RE = re.compile(r"(?:(?<=\s)|^)(?:n[o°º]\.?|#+|no\s*#+)\s*(?=\d)", re.IGNORECASE)

# Street-type and landmark abbreviations -> full word. Ambiguous ones (st, ste, dr, r) are handled
# positionally in _canon_address_tokens.
ADDR_TOKEN_MAP: dict[str, str] = {
    "rd": "road",
    "ave": "avenue",
    "av": "avenue",
    "avn": "avenue",
    "blvd": "boulevard",
    "bd": "boulevard",
    "boul": "boulevard",
    "ln": "lane",
    "ct": "court",
    "cir": "circle",
    "pl": "place",
    "sq": "square",
    "pkwy": "parkway",
    "hwy": "highway",
    "ter": "terrace",
    "terr": "terrace",
    "trl": "trail",
    "cres": "crescent",
    "expy": "expressway",
    "fwy": "freeway",
    "tpke": "turnpike",
    "ctr": "center",
    "centre": "center",
    "apt": "apartment",
    "bldg": "building",
    "fl": "floor",
    "flr": "floor",
    "rm": "room",
    "mkt": "market",
    "opp": "opposite",
    "opps": "opposite",
    "nr": "near",
    "bh": "behind",
    "bhd": "behind",
    "ngr": "nagar",
    "sec": "sector",
    "sect": "sector",
    "stn": "station",
    "rly": "railway",
    "govt": "government",
    "clny": "colony",
    "col": "colony",
    "extn": "extension",
    "ext": "extension",
    "indl": "industrial",
    "estt": "estate",
    "mg": "mg",
    "mahatma": "mahatma",
    "hno": "house",
    "tq": "taluka",
    "tal": "taluka",
    "dist": "district",
    "distt": "district",
    "vill": "village",
    "vil": "village",
    "po": "post",
    "mt": "mount",
    "ft": "fort",
    "pt": "point",
    "hts": "heights",
    "jn": "junction",
    "jct": "junction",
    # French
    "all": "allee",
    "ch": "chemin",
    "che": "chemin",
    "imp": "impasse",
    "rte": "route",
    "fbg": "faubourg",
    "qu": "quai",
    "crs": "cours",
    "pass": "passage",
    "sq.": "square",
    "res": "residence",
    "zi": "zi",
    "za": "za",
    "bat": "batiment",
    "chem": "chemin",
    "prom": "promenade",
    "lot": "lotissement",
    # "no"/"number" markers carry no information once the number itself is kept
    "no": "",
    "number": "",
    "num": "",
}
_ADDR_PHRASES = {"h no": "house", "kh no": "khasra", "s no": "survey", "next to": "near"}
_ADDR_PHRASE_RE = re.compile(r"\b(?:" + "|".join(map(re.escape, _ADDR_PHRASES)) + r")\b")
_NON_ALNUM_KEEP_COMMA_RE = re.compile(r"[^a-z0-9,]+")

# State/province component -> canonical short code. Keyed by the full component text so no country
# lookup is needed; codes that collide across countries (TN = Tennessee / Tamil Nadu) are harmless
# because every sampled true pair shares its country. Romanized Indic spellings were measured on
# true pairs (e.g. "mharastr" -> maharashtra). France: regions/departments are left as words.
_US_STATES = {
    "alabama": "al",
    "alaska": "ak",
    "arizona": "az",
    "arkansas": "ar",
    "california": "ca",
    "colorado": "co",
    "connecticut": "ct",
    "delaware": "de",
    "district of columbia": "dc",
    "florida": "fl",
    "georgia": "ga",
    "hawaii": "hi",
    "idaho": "id",
    "illinois": "il",
    "indiana": "in",
    "iowa": "ia",
    "kansas": "ks",
    "kentucky": "ky",
    "louisiana": "la",
    "maine": "me",
    "maryland": "md",
    "massachusetts": "ma",
    "michigan": "mi",
    "minnesota": "mn",
    "mississippi": "ms",
    "missouri": "mo",
    "montana": "mt",
    "nebraska": "ne",
    "nevada": "nv",
    "new hampshire": "nh",
    "new jersey": "nj",
    "new mexico": "nm",
    "new york": "ny",
    "north carolina": "nc",
    "north dakota": "nd",
    "ohio": "oh",
    "oklahoma": "ok",
    "oregon": "or",
    "pennsylvania": "pa",
    "rhode island": "ri",
    "south carolina": "sc",
    "south dakota": "sd",
    "tennessee": "tn",
    "texas": "tx",
    "utah": "ut",
    "vermont": "vt",
    "virginia": "va",
    "washington": "wa",
    "west virginia": "wv",
    "wisconsin": "wi",
    "wyoming": "wy",
    "puerto rico": "pr",
}
_IN_STATES = {
    "maharashtra": "mh",
    "mharastr": "mh",
    "delhi": "dl",
    "dilli": "dl",
    "uttar pradesh": "up",
    "uttr prdes": "up",
    "karnataka": "ka",
    "krnatk": "ka",
    "tamil nadu": "tn",
    "tamilnadu": "tn",
    "tmilnatu": "tn",
    "gujarat": "gj",
    "gujrat": "gj",
    "west bengal": "wb",
    "pscimbng": "wb",
    "telangana": "tg",
    "telngan": "tg",
    "ts": "tg",
    "haryana": "hr",
    "hriyana": "hr",
    "kerala": "kl",
    "kerlm": "kl",
    "keralam": "kl",
    "rajasthan": "rj",
    "rajsthan": "rj",
    "raj": "rj",
    "madhya pradesh": "mp",
    "mdhy prdes": "mp",
    "bihar": "br",
    "punjab": "pb",
    "pnjab": "pb",
    "andhra pradesh": "ap",
    "andhrprdes": "ap",
    "orissa": "od",
    "odisha": "od",
    "odisa": "od",
    "or": "or",
    "goa": "ga",
    "assam": "as",
    "jharkhand": "jh",
    "chhattisgarh": "cg",
    "uttarakhand": "uk",
    "himachal pradesh": "hp",
    "jammu and kashmir": "jk",
    "chandigarh": "ch",
    "puducherry": "py",
    "pondicherry": "py",
}
STATE_CANON: dict[str, str] = {**_US_STATES, **_IN_STATES}
# Well-known city renames seen on true pairs (Poona<->Pune 68x in the pair sample).
CITY_ALIASES = {
    "poona": "pune",
    "bombay": "mumbai",
    "calcutta": "kolkata",
    "madras": "chennai",
    "bengaluru": "bangalore",
    "gurugram": "gurgaon",
    "mysuru": "mysore",
    "trivandrum": "thiruvananthapuram",
    "baroda": "vadodara",
}


def _canon_address_tokens(toks: list[str]) -> list[str]:
    """Expand address abbreviations in one comma component, resolving st/ste/dr/r by position."""
    out: list[str] = []
    n = len(toks)
    for i, w in enumerate(toks):
        nxt = toks[i + 1] if i + 1 < n else ""
        prev = toks[i - 1] if i else ""
        if w == "st":
            # "105 elm st" (component end / before a number) = street; "st herblain", "st louis" = saint
            w = "street" if (not nxt or nxt.isdigit()) else "saint"
        elif w == "ste":
            w = "suite" if (not nxt or nxt[0].isdigit()) else "sainte"
        elif w == "dr":
            w = "drive" if (not nxt or nxt.isdigit()) else "doctor"
        elif w == "r":
            # French "R. Jules Renard" / "28 R De ..." = rue; single-letter initials elsewhere stay
            w = "rue" if (i == 0 or prev.isdigit()) and nxt and not nxt.isdigit() else w
        else:
            w = ADDR_TOKEN_MAP.get(w, w)
            w = CITY_ALIASES.get(w, w)
        if w:
            if w.isdigit():
                w = w.lstrip("0") or "0"
            out.append(w)
    return out


def normalize_address(raw: str | None) -> str:
    """Normalize an address to lowercase ASCII; comma components are kept (as ' , ') in original order.

    Drops placeholder components (NULL, N/A...), romanizes scripts, folds accents, removes "No."/"#"
    house markers, expands street/landmark abbreviations (English and French), canonicalizes state
    components to short codes and strips leading zeros from numbers. Output: str ('' if empty).
    Results are memoized per distinct raw string.
    """
    return _normalize_address_cached(raw) if raw else ""


@lru_cache(maxsize=131_072)
def _normalize_address_cached(raw: str) -> str:
    """Uncached body of normalize_address (see its docstring)."""
    t = _HOUSE_MARK_RE.sub(" ", raw)
    t = to_ascii(t)
    if "." in t:
        t = _DOTTED_ACRONYM_RE.sub(_undot, t)
    if "&" in t or "+" in t:
        t = _AMP_RE.sub(" and ", t)
    t = _NON_ALNUM_KEEP_COMMA_RE.sub(" ", t)
    t = _ADDR_PHRASE_RE.sub(lambda m: _ADDR_PHRASES[m.group(0)], t)
    comps_out: list[str] = []
    for comp in t.split(","):
        c = comp.strip()
        if c in _PLACEHOLDER_COMPONENTS:
            continue
        st_code = STATE_CANON.get(c)
        if st_code:
            comps_out.append(st_code)
            continue
        toks = _canon_address_tokens(c.split())
        if toks:
            comps_out.append(" ".join(toks))
    return " , ".join(comps_out)


# Postal codes: a standalone 5- or 6-digit number (US ZIP / French code postal = 5, Indian PIN = 6).
# House numbers are also 3-5 digits and usually START a component ("10103 kirsch rd"), so a candidate
# must not be followed by a word in its component; the last qualifying candidate wins.
_PIN_KEYWORD_RE = re.compile(r"\b(?:pin|pincode|pin code|zip|cp|code postal)\s*[:\-]?\s*(\d{3}\s?\d{3}|\d{5})\b", re.I)


def extract_postal(raw_address: str | None) -> str:
    """Extract a postal code from a raw address: '' if none.

    Accepts "PIN 560001" / "PIN 560 001" / "zip 12345" after a keyword. Otherwise takes the last 5-6 digit
    number that is the LAST token of its comma component and follows at least one word or stands alone
    ("IL 62293", "Mumbai 400069", "Pune411011", "... , 44000"). Numbers followed by anything
    ("10103 Kirsch Rd", "21100 68") are house/route numbers, and numbers with a leading '0' are
    zero-padded house numbers ("002160 Muren Blvd") in this data, so both are rejected. Digits only.
    """
    if not raw_address:
        return ""
    m = _PIN_KEYWORD_RE.search(raw_address)
    if m:
        return m.group(1).replace(" ", "")
    best = ""
    for comp in raw_address.split(","):
        mm = _POSTAL_TAIL_RE.search(comp)
        if mm and mm.group(1)[0] != "0":
            best = mm.group(1)
    return best


_POSTAL_TAIL_RE = re.compile(r"(?:^|[^\d/-])(\d{5,6})\s*$")
_DIGITS_RE = re.compile(r"\d+")


def address_numbers(addr_norm: str, postal: str = "") -> str:
    """Return the sorted unique numeric tokens of a normalized address, excluding the postal code."""
    nums = {d.lstrip("0") or "0" for d in _DIGITS_RE.findall(addr_norm)}
    if postal:
        nums.discard(postal.lstrip("0") or "0")
    return " ".join(sorted(nums, key=lambda x: (len(x), x)))


# --------------------------------------------------------------------------------------------------
# Address decomposition v2 (DESIGN_V2 S1; D-011): house / unit / street / city / admin
# --------------------------------------------------------------------------------------------------

# Unit-marker words as they survive normalize_address (ADDR_TOKEN_MAP already expands apt->apartment,
# fl/flr->floor; a bare "#"/"No." marker is stripped upstream by _HOUSE_MARK_RE and so is NOT
# distinguishable here from a house number -- "# 1" ends up in `house`, not `unit`; this is a known,
# accepted limitation, exercised by a test).
UNIT_WORDS: frozenset[str] = frozenset({"unit", "suite", "apartment", "flat", "shop", "pmb", "floor", "room"})
# Street-type / landmark words that mark a component as a street (canonical spellings, i.e. what
# ADDR_TOKEN_MAP maps TO, plus words that already arrive spelled out).
STREET_WORDS: frozenset[str] = frozenset(
    """
    road street avenue boulevard lane drive court circle place square parkway highway terrace trail
    crescent expressway freeway turnpike way walk close crossing nagar marg colony sector chemin allee
    route quai cours passage residence rue faubourg impasse
    """.split()
)
# Admin (state/région/département) codes derived only from the hand-written STATE_CANON map already used
# by normalize_address (no new external list: D-015). By the time addr_norm exists, a matched state
# component has already been rewritten to its short code, so membership here is a plain string check.
ADMIN_CODES: frozenset[str] = frozenset(STATE_CANON.values())
# House-number suffixes attached to a digit ("30 bis", "46 a"): kept with the number, not city/street.
_NUM_SUFFIX: frozenset[str] = frozenset({"a", "b", "c", "bis", "ter", "quater"})
# Building/plot qualifier words that only ever precede a house number ("Plot 72", "House 12", the
# "khasra"/"survey" phrases _ADDR_PHRASES already expands "kh no"/"s no" to): dropped rather than kept in
# `city`, since they carry no locality information (findings.md: "house/building/plot numbers").
_HOUSE_QUALIFIERS: frozenset[str] = frozenset({"plot", "house", "khasra", "survey", "block"})


def _split_addr_components(addr_norm: str) -> list[str]:
    """Split addr_norm on the ' , ' component separator normalize_address emits; drops empty pieces."""
    return [c for c in addr_norm.split(" , ") if c]


def parse_address(addr_norm: str, country_norm: str, admin_vocab: frozenset[str] | None = None) -> dict[str, str]:
    """Decompose a normalized address into house/unit/street/city/admin (DESIGN_V2 S1, D-011).

    Classifies each comma component of ``addr_norm`` by its CONTENT, never its position, because true
    pairs reorder components between sources ("IA, Iowa City, 1064 Newton Rd" vs the US order). Per
    component: (1) if the whole component matches an admin code -- ADMIN_CODES (US/Indian state codes,
    hand-written, derived from STATE_CANON) unioned with the caller-supplied, data-learned
    ``admin_vocab`` (see learn_admin_vocab) -- it's `admin`; (2) else if it contains a unit-marker word
    (unit/suite/apartment/flat/shop/pmb/floor/room, or the phrase "post box"), it's `unit`; (3) else every
    digit token (plus an attached suffix like "bis"/"a") is pulled into `house`, and the remaining words
    go to `street` if they contain a street-type word (road/avenue/rue/nagar/marg/sector/...), else to
    `city`. Multiple components can land in the same class (e.g. "Bangalore North" and "Bangalore" both
    landing in `city`); their text is space-joined, in the order seen. ``country_norm`` is accepted for a
    future per-country refinement, but classification itself stays country-agnostic on purpose: no
    hand-written France région/département map is used (D-011, D-015), which is what lets an unseen
    country degrade gracefully -- a French région with no matching admin_vocab entry simply falls into
    `city` rather than raising or silently dropping data. Deterministic; '' addr_norm gives all-'' fields.
    """
    vocab = ADMIN_CODES if not admin_vocab else ADMIN_CODES | admin_vocab
    house: list[str] = []
    unit: list[str] = []
    street: list[str] = []
    city: list[str] = []
    admin: list[str] = []
    for comp in _split_addr_components(addr_norm):
        if comp in vocab:
            admin.append(comp)
            continue
        toks = comp.split()
        if any(t in UNIT_WORDS for t in toks) or ("post" in toks and "box" in toks):
            unit.append(comp)
            continue
        has_digit = any(t.isdigit() for t in toks)
        rest: list[str] = []
        for i, t in enumerate(toks):
            if t.isdigit():
                house.append(t)
            elif t in _NUM_SUFFIX and house and i > 0 and toks[i - 1].isdigit():
                house.append(t)
            elif t in _HOUSE_QUALIFIERS and has_digit:
                continue  # "Plot 72" / "House 12": the qualifier carries no locality signal
            else:
                rest.append(t)
        if not rest:
            continue
        (street if any(t in STREET_WORDS for t in rest) else city).append(" ".join(rest))
    return {
        "house": " ".join(house),
        "unit": " ".join(unit),
        "street": " ".join(street),
        "city": " ".join(city),
        "admin": " ".join(admin),
    }


def learn_admin_vocab(addr_norms: Iterable[str], min_count: int = 200, max_tokens: int = 3) -> set[str]:
    """Learn an admin-unit (state/région/département) vocabulary from address data alone (D-011, D-015).

    Counts every non-numeric comma component with at most ``max_tokens`` tokens across ``addr_norms``, and
    separately counts how often each occurs specifically as the FIRST or LAST component of its address --
    the typical position of a state/région/département. A component is kept when its total count is >=
    min_count and at least half of its occurrences are at an edge position. No hand-written French
    région/département table is used (that's explicitly rejected in D-011): this is how such names get
    discovered at all. Deterministic for a given input order. Returns a plain set (JSON-serializable via
    `list(...)`), meant to be passed as ``admin_vocab`` to parse_address / normalize_batch.
    """
    total: Counter = Counter()
    edge: Counter = Counter()
    for addr in addr_norms:
        if not addr:
            continue
        comps = _split_addr_components(addr)
        n = len(comps)
        for i, c in enumerate(comps):
            toks = c.split()
            if not toks or len(toks) > max_tokens or any(t.isdigit() for t in toks):
                continue
            total[c] += 1
            if i == 0 or i == n - 1:
                edge[c] += 1
    return {c for c, cnt in total.items() if cnt >= min_count and edge[c] * 2 >= cnt}


# --------------------------------------------------------------------------------------------------
# Batch / file processing
# --------------------------------------------------------------------------------------------------

OUTPUT_SCHEMA = pa.schema(
    [
        ("entity_id", pa.string()),
        ("name_norm", pa.string()),
        ("name_core", pa.string()),
        ("name_skel", pa.string()),
        ("addr_norm", pa.string()),
        ("postal", pa.string()),
        ("addr_nums", pa.string()),
        ("script", pa.string()),
        ("country_norm", pa.string()),
        ("legal_form", pa.string()),
        ("type_words", pa.string()),
        ("addr_house", pa.string()),
        ("addr_unit", pa.string()),
        ("addr_street", pa.string()),
        ("addr_city", pa.string()),
        ("addr_admin", pa.string()),
    ]
)


def normalize_batch(
    table: pa.Table | pa.RecordBatch,
    token_map: dict[str, str] | None = None,
    admin_vocab: frozenset[str] | None = None,
) -> pa.Table:
    """Normalize a table with entity_id, business_name, business_address, country columns.

    ``token_map`` (learn_token_map output, D-012) is applied to the normalized name -- before name_core /
    name_skel are computed, so a right-side abbreviation like "pra li" resolves to "private limited" first
    -- but name_norm itself is left unmapped, same as name_core already keeps name_norm unstripped.
    ``admin_vocab`` (learn_admin_vocab output, D-011) extends parse_address's built-in state-code
    vocabulary. Both are looked up / compiled once per batch, not per row, to hold throughput.

    Returns a table with OUTPUT_SCHEMA: entity_id, name_norm, name_core, name_skel, addr_norm, postal,
    addr_nums, script (of the raw name), country_norm (lowercased, stripped), legal_form, type_words,
    addr_house, addr_unit, addr_street, addr_city, addr_admin.
    """
    ids = table.column("entity_id").to_pylist()
    names = table.column("business_name").to_pylist()
    addrs = table.column("business_address").to_pylist()
    countries = table.column("country").to_pylist()
    pattern = _compile_phrase_pattern(token_map) if token_map else None
    nn, nc, nk, an, po, anum, sc, cn = [], [], [], [], [], [], [], []
    lf, tw, ah, au, ast, acy, aad = [], [], [], [], [], [], []
    for name, addr, ctry in zip(names, addrs, countries, strict=True):
        n = normalize_name(name)
        a = normalize_address(addr)
        p = extract_postal(addr)
        c_norm = (ctry or "").strip().lower()
        nn.append(n)
        mapped = apply_token_map(n, token_map, pattern) if token_map else n
        core = name_core(mapped)
        nc.append(core)
        nk.append(name_skeleton(core))
        an.append(a)
        po.append(p)
        anum.append(address_numbers(a, p))
        sc.append(detect_script(name))
        cn.append(c_norm)
        _, legal = split_legal_form(mapped)
        lf.append(legal)
        tw.append(type_words_of(core))
        addr_parts = parse_address(a, c_norm, admin_vocab)
        ah.append(addr_parts["house"])
        au.append(addr_parts["unit"])
        ast.append(addr_parts["street"])
        acy.append(addr_parts["city"])
        aad.append(addr_parts["admin"])
    cols = (nn, nc, nk, an, po, anum, sc, cn, lf, tw, ah, au, ast, acy, aad)
    return pa.Table.from_arrays(
        [pa.array(ids, pa.string()), *(pa.array(c, pa.string()) for c in cols)],
        schema=OUTPUT_SCHEMA,
    )


_IN_COLS = ["entity_id", "business_name", "business_address", "country"]


def _iter_batches(in_path: str, batch_rows: int):
    """Yield record batches (only the 4 input columns) from a Parquet file."""
    pf = pq.ParquetFile(in_path)
    yield from pf.iter_batches(batch_size=batch_rows, columns=_IN_COLS)


def _load_token_map(path: str | None) -> dict[str, str] | None:
    """Load a learn_token_map JSON file, or return None if no path was given."""
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_admin_vocab(path: str | None) -> frozenset[str] | None:
    """Load a learn_admin_vocab JSON file (a list of strings), or return None if no path was given."""
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        return frozenset(json.load(f))


def normalize_parquet(
    in_path: str,
    out_path: str,
    workers: int = 2,
    batch_rows: int = 50_000,
    token_map_path: str | None = None,
    admin_vocab_path: str | None = None,
) -> int:
    """Stream ``in_path`` through normalize_batch and write zstd Parquet to ``out_path``.

    Uses a process pool fed through a bounded window of ``2*workers`` in-flight batches (Pool.imap would
    drain the reader eagerly), so memory stays bounded; output order equals input order. ``token_map_path``
    / ``admin_vocab_path`` (JSON files produced by learn_token_map / learn_admin_vocab) are loaded once
    here and passed to every worker call, so the learned maps are computed exactly once regardless of the
    number of batches or workers. Returns the number of rows written.
    """
    token_map = _load_token_map(token_map_path)
    admin_vocab = _load_admin_vocab(admin_vocab_path)
    writer = pq.ParquetWriter(out_path, OUTPUT_SCHEMA, compression="zstd")
    n = 0
    try:
        if workers <= 1:
            for b in _iter_batches(in_path, batch_rows):
                out = normalize_batch(b, token_map, admin_vocab)
                writer.write_table(out)
                n += out.num_rows
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(workers) as pool:
                window: deque = deque()
                for b in _iter_batches(in_path, batch_rows):
                    window.append(pool.apply_async(normalize_batch, (b, token_map, admin_vocab)))
                    if len(window) >= 2 * workers:
                        out = window.popleft().get()
                        writer.write_table(out)
                        n += out.num_rows
                while window:
                    out = window.popleft().get()
                    writer.write_table(out)
                    n += out.num_rows
    finally:
        writer.close()
    return n
