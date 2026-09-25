"""Tests for ber.textdiff on tiny synthetic name/number pairs: true matches (explainable noise) must score
better than near-duplicate distractors (unexplainable word swaps/additions, nudged house numbers)."""

from __future__ import annotations

import numpy as np

from ber.textdiff import (
    align_tokens,
    idf_table,
    learn_token_logodds,
    name_diff_features,
    number_relation_features,
)

TRUE_PAIRS = [
    ("bonanza travels", "bonanza travles"),
    ("win hydro", "dr win hydro"),
    ("foot and ankle modern clinic", "footanklemodern"),
    ("ram marketing", "raam maarketing"),
]
DISTRACTOR_PAIRS = [
    ("bonanza real", "bonanza travels"),
    ("inland appliances", "inland aphpliance group"),
    ("peak ethereum", "peak ethereum ventures"),
]


def test_align_tokens_true_typo_examples() -> None:
    """Pure typo noise: every token finds a same-side partner, nothing is left unexplained."""
    cls_l, cls_r = align_tokens("bonanza travels".split(), "bonanza travles".split())
    assert cls_l == ["exact", "typo"]
    assert cls_r == ["exact", "typo"]

    cls_l, cls_r = align_tokens("ram marketing".split(), "raam maarketing".split())
    assert cls_l == ["typo", "typo"]
    assert cls_r == ["typo", "typo"]


def test_align_tokens_true_added_title_word() -> None:
    """A junk prefix ("dr") on the noisy side is the only unexplained token; the content words match exactly."""
    cls_l, cls_r = align_tokens("win hydro".split(), "dr win hydro".split())
    assert cls_l == ["exact", "exact"]
    assert cls_r == ["unexplained", "exact", "exact"]


def test_align_tokens_joined_form_abbrev() -> None:
    """A domain-style joined name is explained as concatenation of the other side's content tokens, skipping
    the connector word "and"; only "and" and the dropped "clinic" are left unexplained."""
    a = "foot and ankle modern clinic".split()
    b = "footanklemodern".split()
    cls_a, cls_b = align_tokens(a, b)
    assert cls_b == ["abbrev"]
    assert cls_a[0] == "abbrev"  # foot
    assert cls_a[2] == "abbrev"  # ankle
    assert cls_a[3] == "abbrev"  # modern
    assert cls_a[1] == "unexplained"  # and
    assert cls_a[4] == "unexplained"  # clinic


def test_align_tokens_distractor_swapped_word() -> None:
    """A swapped content word has no acceptable partner on either side."""
    cls_l, cls_r = align_tokens("bonanza real".split(), "bonanza travels".split())
    assert cls_l == ["exact", "unexplained"]
    assert cls_r == ["exact", "unexplained"]


def test_align_tokens_distractor_added_type_word() -> None:
    """An added business-type word ("ventures") is unexplained; the shared words match exactly."""
    cls_l, cls_r = align_tokens("peak ethereum".split(), "peak ethereum ventures".split())
    assert cls_l == ["exact", "exact"]
    assert cls_r == ["exact", "exact", "unexplained"]


def test_align_tokens_distractor_added_group_word() -> None:
    """An added "group" survives as unexplained even with an unrelated typo elsewhere in the name."""
    _cls_l, cls_r = align_tokens("inland appliances".split(), "inland aphpliance group".split())
    assert cls_r[-1] == "unexplained"


def test_align_tokens_legal_form_drift() -> None:
    """A legal-form suffix present on only one side is classified 'legal', not 'unexplained'."""
    cls_l, cls_r = align_tokens("ram marketing pvt ltd".split(), "ram marketing".split())
    assert cls_l[:2] == ["exact", "exact"]
    assert set(cls_l[2:]) == {"legal"}


def test_align_tokens_translit_skeleton() -> None:
    """Two spellings that share a consonant skeleton but differ in every vowel (too far apart to be a typo:
    4 substitutions in an 8-letter token) align as transliteration."""
    cls_l, cls_r = align_tokens(["kanawana"], ["konowono"])
    assert cls_l == ["translit"]
    assert cls_r == ["translit"]


def test_align_tokens_empty_inputs() -> None:
    """Empty token lists align to empty classifications without error."""
    assert align_tokens([], []) == ([], [])
    cls_l, cls_r = align_tokens([], ["a"])
    assert cls_l == []
    assert cls_r == ["unexplained"]


def test_align_tokens_deterministic() -> None:
    """Repeated calls on the same input give the same classification."""
    a, b = "peak ethereum".split(), "peak ethereum ventures".split()
    assert align_tokens(a, b) == align_tokens(a, b)


def _train_logodds() -> dict[str, float]:
    """A synthetic fit-group table where 'group' and 'ventures' are added, unexplained, only in negatives."""
    pairs: list[tuple[str, str, int]] = []
    bases = [
        ("acme partners", "acme partnrs"),
        ("blue star traders", "blu star tradrs"),
        ("summit rentals", "sumit rentals"),
        ("harbor logistics", "harbour logistcs"),
        ("delta foods", "delta foods pvt ltd"),
    ]
    for l_name, r_name in bases:
        for _ in range(6):  # >= 20 occurrences per base word across positives, no group/ventures here
            pairs.append((l_name, r_name, 1))
    neg_bases = ["acme partners", "blue star traders", "summit rentals", "harbor logistics", "delta foods"]
    for base in neg_bases:
        for _ in range(6):
            pairs.append((base, base + " group", 0))
            pairs.append((base, base + " ventures", 0))
    return learn_token_logodds(pairs, min_count=20, alpha=1.0)


def test_learn_token_logodds_flags_distractor_words() -> None:
    """'group' and 'ventures' are unexplained almost only in negative pairs, so they get positive log-odds."""
    logodds = _train_logodds()
    assert "group" in logodds
    assert "ventures" in logodds
    assert logodds["group"] > 0
    assert logodds["ventures"] > 0


def test_learn_token_logodds_ignores_rare_tokens() -> None:
    """A token seen fewer than min_count times is dropped, not returned as (falsely confident) 0."""
    logodds = learn_token_logodds([("acme partners", "acme partners group", 0)], min_count=20)
    assert "group" not in logodds
    assert "acme" not in logodds


def test_name_diff_features_separates_true_from_distractor() -> None:
    """With trained log-odds, the distractor's unexplained added word scores far higher than any true pair's,
    even though the joined-form true example also leaves some tokens unexplained."""
    logodds = _train_logodds()
    all_names = [n for pair in TRUE_PAIRS + DISTRACTOR_PAIRS for n in pair]
    idf = idf_table(all_names)

    l_names = [p[0] for p in TRUE_PAIRS + DISTRACTOR_PAIRS]
    r_names = [p[1] for p in TRUE_PAIRS + DISTRACTOR_PAIRS]
    feats = name_diff_features(l_names, r_names, idf, logodds)

    for col in (
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
    ):
        assert col in feats
        assert feats[col].dtype == np.float32
        assert len(feats[col]) == len(l_names)

    n_true = len(TRUE_PAIRS)
    true_max_logodds = np.maximum(feats["nd_l_logodds_max"][:n_true], feats["nd_r_logodds_max"][:n_true])
    distractor_max_logodds = np.maximum(feats["nd_l_logodds_max"][n_true:], feats["nd_r_logodds_max"][n_true:])
    # "peak ethereum ventures" and "inland ... group" carry a trained distractor signature; no true pair does.
    assert distractor_max_logodds.max() > 0
    assert true_max_logodds.max() == 0

    # the pure-typo true pairs have zero unexplained tokens on both sides.
    assert feats["nd_l_unexplained"][0] == 0  # bonanza travels/travles
    assert feats["nd_r_unexplained"][0] == 0
    assert feats["nd_typo"][0] == 2
    assert feats["nd_typo"][3] == 4  # ram marketing / raam maarketing


def test_name_diff_features_defaults_and_empty() -> None:
    """idf/logodds default sensibly when None, and empty names don't crash the feature computation."""
    l_names = ["", "corp", "acme corp"]
    r_names = ["acme", "", "corp acme"]
    feats = name_diff_features(l_names, r_names, None, None)

    assert feats["nd_l_ntok"][0] == 0
    assert feats["nd_r_ntok"][0] == 1
    assert feats["nd_l_exact_frac"][0] == 0.0
    assert feats["nd_r_unexplained"][0] == 1  # "acme" has no counterpart on the empty left side

    assert feats["nd_r_ntok"][1] == 0
    # "corp" with no counterpart is a legal-form word, not unexplained.
    assert feats["nd_l_unexplained"][1] == 0
    assert feats["nd_legal_diff"][1] == 1

    # exact matching is position-independent: reordered tokens still match exactly, not as legal-form drift.
    assert feats["nd_l_exact_frac"][2] == 1.0
    assert feats["nd_legal_diff"][2] == 0


def test_number_relation_digit_drop() -> None:
    """7619 vs 619: removing the leading digit gives an exact match."""
    feats = number_relation_features(["7619"], ["619"])
    assert feats["nr_class"][0] == 2
    assert feats["nr_both_present"][0] == 1


def test_number_relation_substitution() -> None:
    """10756 vs 10796: a single non-trivial digit changed, numerically far apart (diff 40 > 10)."""
    feats = number_relation_features(["10756"], ["10796"])
    assert feats["nr_class"][0] == 3


def test_number_relation_small_difference() -> None:
    """5124 vs 5129: same length, but the numeric difference (5) is small -- a nudged house number."""
    feats = number_relation_features(["5124"], ["5129"])
    assert feats["nr_class"][0] == 5
    assert feats["nr_min_rel_diff"][0] < 0.01


def test_number_relation_equal_present_among_several() -> None:
    """One side has a single number, the other has two; the shared "20" is picked as the best match."""
    feats = number_relation_features(["20"], ["20 22"])
    assert feats["nr_class"][0] == 1
    assert feats["nr_first_eq"][0] == 1
    assert feats["nr_n_equal"][0] == 1
    assert feats["nr_both_present"][0] == 1


def test_number_relation_missing_side() -> None:
    """No numbers on one (or both) side gives class 0, not a false "far" reading."""
    feats = number_relation_features(["", "123"], ["45", ""])
    assert feats["nr_class"][0] == 0
    assert feats["nr_both_present"][0] == 0
    assert feats["nr_class"][1] == 0
    assert feats["nr_both_present"][1] == 0


def test_number_relation_deterministic() -> None:
    """Repeated calls give bit-identical arrays."""
    a = number_relation_features(["7619", "5124"], ["619", "5129"])
    b = number_relation_features(["7619", "5124"], ["619", "5129"])
    for k in a:
        assert np.array_equal(a[k], b[k])


def test_idf_table_basic() -> None:
    """log(N / document_frequency), computed over normalized (lowercased, ASCII) tokens."""
    idf = idf_table(["acme corp", "acme foods", "beta corp"])
    assert idf["acme"] == np.log(3 / 2)
    assert idf["corp"] == np.log(3 / 2)
    assert idf["foods"] == np.log(3 / 1)


def test_idf_table_empty() -> None:
    """No names in -> empty table out, no crash."""
    assert idf_table([]) == {}
