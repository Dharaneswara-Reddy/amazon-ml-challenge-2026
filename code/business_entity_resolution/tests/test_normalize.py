"""Table-driven tests for ber.normalize, built from patterns observed in the training/test samples."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rapidfuzz import fuzz

from ber.normalize import (
    OUTPUT_SCHEMA,
    TYPE_WORDS,
    address_numbers,
    apply_token_map,
    detect_script,
    extract_postal,
    learn_admin_vocab,
    learn_token_map,
    name_core,
    name_skeleton,
    normalize_address,
    normalize_batch,
    normalize_name,
    normalize_parquet,
    parse_address,
    split_legal_form,
    type_words_of,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("-- Holloway Peak Inc Seafood", "holloway peak inc seafood"),  # junk prefix
        ("<< Team Ecole", "team ecole"),
        ("*** Moon Traders Private Ltd", "moon traders pvt ltd"),
        ("Rajd0ot Traders-Public [Limited]", "rajdoot traders public ltd"),  # digit swap, brackets
        ("Rajani + Br0thers Pvt (Ltd)", "rajani and brothers pvt ltd"),
        ("Vickers & Rább Indigo LLC", "vickers and rabb indigo llc"),  # injected accent
        ("West Siddhi Córp", "west siddhi corp"),
        ("Continental Freight Technologies L.L.C.", "continental freight technologies llc"),
        ("Arcvio DBA: Hanson Bitwise Inc", "hanson bitwise inc"),  # alias marker -> keep tail
        ("Novigild formerly known as Valley Pacific Outdoor LLC", "valley pacific outdoor llc"),
        ("Lumyuma F/K/A Ballard Agricultural", "ballard agricultural"),
        ("Stewart  Academy (ID: 69587)", "stewart academy"),
        ("ca1dial cola | www.caldialco.com", "caldial cola"),
        ("DennisfamilypracticeCom", "dennisfamilypractice"),
        ("urbanmassage.c0m", "urbanmassage"),
        ("Freshlight Chit Corp Corp", "freshlight chit corp"),  # consecutive duplicate
        ("Copains + Frères Groupe S.A.R.L.", "copains and freres groupe sarl"),  # French
        ("Lecho et Fils SAS", "lecho and fils sas"),
        ("Ets Transmettre SAS", "etablissements transmettre sas"),
        ("Fete & Cie S.A.S", "fete and compagnie sas"),
        ("", ""),
        (None, ""),
        ("***", ""),
    ],
)
def test_normalize_name(raw, expected):
    """normalize_name maps observed noise patterns to the expected canonical string."""
    assert normalize_name(raw) == expected


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Shivam Infotech Pvt Ltd", "Shivam Infotech Private Limited"),
        ("Sterling Rate Inc.", "Zephvera trading as Sterling Rate Inc."),
        ("Vickers and Rabb Indigo LLC", "VICKERS & RÁBB INDIGO L.L.C."),
        ("Nou Elementaire SARL", "Nou Élémentaire S.A.R.L."),
    ],
)
def test_equivalent_names_normalize_equal(a, b):
    """Spelling variants of the same business normalize to the identical string."""
    assert normalize_name(a) == normalize_name(b)


def _skel(raw):
    """Full name pipeline down to the phonetic skeleton."""
    return name_skeleton(name_core(normalize_name(raw)))


@pytest.mark.parametrize(
    ("indic", "latin", "min_ratio"),
    [
        ("राम मार्केटिंग प्राइवेट लिमिटेड", "Ram Marketing Private Limited", 100),
        ("রাম কনসালটেন্সি প্রাইভেট লিমিটেড", "Ram Consultancy Private Limited", 100),  # Bengali
        ("ड्रीम पावर प्राइवेट लिमिटेड", "Dream Power Private Limited", 100),
        ("ગોલ્ડ સુપ્રીમ હોસ્પિટાલિટી પ્રાઇવેટ લિમિટેડ", "Gold Supreme Hospitality Private Limited", 85),
        ("शिवम इंफोटेक प्रा. लि.", "Shivam Infotech Pvt Ltd", 85),
        ("జైన్ హరి టెక్ ఎల్‌ఎల్‌పీ", "Jain Hari Tech LLP", 100),  # Telugu, zero-width joiners
    ],
)
def test_indic_transliteration_close_to_latin(indic, latin, min_ratio):
    """Indic-script names written phonetically reach their Latin S1 counterpart at skeleton level."""
    assert normalize_name("राम मार्केटिंग प्राइवेट लिमिटेड") == "ram marketing pvt ltd"
    assert fuzz.ratio(_skel(indic), _skel(latin)) >= min_ratio


def test_skeleton_keeps_distinct_names_apart():
    """The phonetic skeleton must not merge different trade words."""
    assert _skel("Sharma Traders") != _skel("Sharma Textiles")
    assert name_skeleton("dream power") == name_skeleton("drim pavr") == "drm pvr"


@pytest.mark.parametrize(
    ("norm", "core"),
    [
        ("ram marketing pvt ltd", "ram marketing"),
        ("dr win hydro ltd pvt", "win hydro"),  # leading honorific + shuffled legal suffix
        ("inc jordanas blue diagnostics", "jordanas blue diagnostics"),
        ("the little barbershop lp", "little barbershop"),
        ("sharma traders", "sharma traders"),  # content words are never stripped
        ("sharma textiles pvt ltd", "sharma textiles"),
        ("pvt ltd", "pvt ltd"),  # never empty: falls back to the full name
        ("the", "the"),
        ("", ""),
    ],
)
def test_name_core(norm, core):
    """name_core strips legal forms and leading fillers only, and never returns empty."""
    assert name_core(norm) == core


def test_name_core_keeps_distinct_businesses_apart():
    """Different trades must stay different after core extraction (precision guard)."""
    a = name_core(normalize_name("Sharma Traders Pvt Ltd"))
    b = name_core(normalize_name("Sharma Textiles Private Limited"))
    assert a != b


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("105 ELM ST, MORGANTON, NC", "105 elm street , morganton , nc"),
        ("3227 Redding Rd, Columbus, Ohio", "3227 redding road , columbus , oh"),
        ("2853 Ontario Rd, <NULL>, Washinton, District of Columbia", "2853 ontario road , washinton , dc"),
        ("001217 GLENWOOD, TX, GRAND PRAIRIE", "1217 glenwood , tx , grand prairie"),
        ("NO #23 , 3RD CR, BEHIND NEW", "23 , 3rd cr , behind new"),
        ("Nr. Narmada Canal, Adalaj, Gandhinagar, ગુજરાત", "near narmada canal , adalaj , gandhinagar , gj"),
        ("Tola - Ibrahimpur, Patna, बिहार", "tola ibrahimpur , patna , br"),
        ("Indore, मध्य प्रदेश", "indore , mp"),
        ("Pune, Market Yard, Poona, MH", "pune , market yard , pune , mh"),
        ("N° 7 R. JULES RENARD, MÉRIGNAC", "7 rue jules renard , merignac"),
        ("12 ALLÉE PROSPER MÉRIMÉE, ST.-HERBLAIN", "12 allee prosper merimee , saint herblain"),
        ("194 AV ANTOINE DE SAINT EXUPÉRY, CALAIS", "194 avenue antoine de saint exupery , calais"),
        ("", ""),
        (None, ""),
    ],
)
def test_normalize_address(raw, expected):
    """normalize_address canonicalizes abbreviations, states, scripts and placeholders."""
    assert normalize_address(raw) == expected


def test_reordered_address_same_tokens():
    """Reordered components give the same token set (order is left to the similarity function)."""
    a = normalize_address("1064 Newton Road, Unit 11, Iowa City, IA")
    b = normalize_address("IA, Iowa City, 1064 Newton Rd, Unit 11")
    assert sorted(a.replace(",", " ").split()) == sorted(b.replace(",", " ").split())


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("10103 Kirsch Road, Saint Jacob, IL 62293", "62293"),
        ("Andheri East, Mumbai 400069", "400069"),
        ("Koramangala, PIN 560 001", "560001"),
        ("Somwar Peth, Pune411011, Pune", "411011"),
        ("Rue Nationale, Lille 59000", "59000"),
        ("10103 Kirsch Road, Saint Jacob, IL", ""),  # 5-digit house number is not a ZIP
        ("21100 68, Robertsdale, AL", ""),  # house + route number
        ("002160 MUREN BOULEVARD, BELLEVILLE, IL", ""),  # zero-padded house number
        ("KH NO. -570/13, NEW DELHI", ""),
        ("", ""),
        (None, ""),
    ],
)
def test_extract_postal(raw, expected):
    """extract_postal finds real postal codes and rejects house numbers."""
    assert extract_postal(raw) == expected


def test_address_numbers_excludes_postal():
    """address_numbers returns sorted unique numbers without the postal code."""
    raw = "No 44, Ward No. 1, 44 Main Road, Bangalore 560001"
    assert address_numbers(normalize_address(raw), extract_postal(raw)) == "1 44"
    assert address_numbers("") == ""


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Ram Marketing", "latin"),
        ("Établissements Boule", "latin"),
        ("राम मार्केटिंग", "devanagari"),
        ("রাম কনসালটেন্সি", "other"),
        ("Arihant টেকনোলজিস Private", "mixed"),
        ("", "latin"),
        (None, "latin"),
        ("123 --", "latin"),
    ],
)
def test_detect_script(raw, expected):
    """detect_script classifies the letters of a raw string."""
    assert detect_script(raw) == expected


def _raw_table():
    """Small input table in the raw-cache schema."""
    return pa.table(
        {
            "entity_id": ["S1-1", "S2-2", "S3-3", "S2-4"],
            "business_name": ["Ram Marketing Pvt Ltd", "राम मार्केटिंग प्राइवेट लिमिटेड", "Lecho & Fils SAS", None],
            "business_address": ["KH NO. -570/13, Delhi 110018", "Delhi", None, "N/A"],
            "country": ["India", " India ", "France", "US"],
        }
    )


def test_normalize_batch_schema_and_values():
    """normalize_batch returns OUTPUT_SCHEMA with one row per input and sensible values."""
    out = normalize_batch(_raw_table())
    assert out.schema == OUTPUT_SCHEMA
    rows = out.to_pylist()
    assert [r["entity_id"] for r in rows] == ["S1-1", "S2-2", "S3-3", "S2-4"]
    assert rows[0]["name_core"] == rows[1]["name_core"] == "ram marketing"
    assert rows[0]["name_skel"] == rows[1]["name_skel"] == "rm mrktng"
    assert rows[0]["postal"] == "110018"
    assert rows[0]["addr_nums"] == "13 570"
    assert rows[1]["script"] == "devanagari"
    assert rows[1]["country_norm"] == "india"
    assert rows[2]["country_norm"] == "france"
    assert rows[3]["name_norm"] == "" and rows[3]["addr_norm"] == ""


def test_determinism():
    """Normalization is deterministic across repeated calls."""
    t = _raw_table()
    assert normalize_batch(t).equals(normalize_batch(t))


@pytest.mark.parametrize("workers", [1, 2])
def test_normalize_parquet_roundtrip(tmp_path, workers):
    """normalize_parquet streams batches and preserves row order, with and without a pool."""
    t = pa.concat_tables([_raw_table()] * 5)
    src, dst = tmp_path / "in.parquet", tmp_path / "out.parquet"
    pq.write_table(t, src)
    n = normalize_parquet(str(src), str(dst), workers=workers, batch_rows=3)
    out = pq.read_table(dst)
    assert n == out.num_rows == t.num_rows
    assert out.column("entity_id").to_pylist() == t.column("entity_id").to_pylist()
    assert out.equals(normalize_batch(t))


# --------------------------------------------------------------------------------------------------
# v2: legal form / type words (DESIGN_V2 S1, D-013)
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name_norm", "expected"),
    [
        ("ram marketing pvt ltd", ("ram marketing", "private limited")),
        ("dr win hydro ltd pvt", ("dr win hydro", "private limited")),  # shuffled suffix
        ("win hydro pvt ltd", ("win hydro", "private limited")),
        ("jordanas blue diagnostics inc", ("jordanas blue diagnostics", "inc")),
        ("continental freight technologies llc", ("continental freight technologies", "llc")),
        ("fete and compagnie sas", ("fete and compagnie", "sas")),
        ("copains and freres groupe sarl", ("copains and freres groupe", "sarl")),
        ("lecho and fils sas", ("lecho and fils", "sas")),
        ("etablissements transmettre sas", ("etablissements transmettre", "sas")),
        ("sharma traders", ("sharma traders", "")),  # no legal form
        ("", ("", "")),
    ],
)
def test_split_legal_form(name_norm, expected):
    """split_legal_form strips legal-form tokens (any order) and canonicalizes their combination."""
    assert split_legal_form(name_norm) == expected


def test_split_legal_form_word_order_shuffle_agrees():
    """A shuffled legal suffix resolves to the same canonical form as the unshuffled one."""
    assert split_legal_form("acme ltd pvt")[1] == split_legal_form("acme pvt ltd")[1] == "private limited"


@pytest.mark.parametrize(
    ("name_core_str", "expected"),
    [
        ("acme ventures group", "group ventures"),
        ("sharma traders", "traders"),
        ("jordanas blue diagnostics", ""),
        ("global tech solutions holdings", "global holdings solutions tech"),
        ("", ""),
    ],
)
def test_type_words_of(name_core_str, expected):
    """type_words_of returns the TYPE_WORDS present in name_core, sorted and space-joined."""
    assert type_words_of(name_core_str) == expected


def test_type_words_of_custom_vocab():
    """A caller-supplied vocab (e.g. a learned one) replaces the TYPE_WORDS default."""
    assert type_words_of("acme foo bar", vocab=frozenset({"foo"})) == "foo"
    assert type_words_of("acme group", vocab=frozenset({"foo"})) == ""  # "group" not in the custom vocab
    assert "group" in TYPE_WORDS and "services" in TYPE_WORDS


# --------------------------------------------------------------------------------------------------
# v2: address decomposition (DESIGN_V2 S1, D-011)
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "country", "expected"),
    [
        (
            "#563/564, Bangalore North, Bangalore, KA",
            "india",
            {"house": "563 564", "unit": "", "street": "", "city": "bangalore north bangalore", "admin": "ka"},
        ),
        (
            "Plot No. 72, Varanasi, UP",
            "india",
            {"house": "72", "unit": "", "street": "", "city": "varanasi", "admin": "up"},
        ),
        (
            "NY, ROCHESTER, 145 KLINK ROAD",
            "us",
            {"house": "145", "unit": "", "street": "klink road", "city": "rochester", "admin": "ny"},
        ),
        (
            "123 Main St, Unit 16, Springfield, IL",
            "us",
            {"house": "123", "unit": "unit 16", "street": "main street", "city": "springfield", "admin": "il"},
        ),
        ("", "us", {"house": "", "unit": "", "street": "", "city": "", "admin": ""}),
    ],
)
def test_parse_address(raw, country, expected):
    """parse_address classifies each comma component of addr_norm by content, not position."""
    assert parse_address(normalize_address(raw), country) == expected


def test_parse_address_unit_marker_word_vs_stripped_hash():
    """A word unit marker ('Unit 16') is kept as `unit`; a bare '#' marker is already stripped by
    normalize_address's house-marker cleanup, so it surfaces in `house`, not `unit` (documented limit)."""
    with_word = parse_address(normalize_address("123 Main St, Unit 16, Springfield, IL"), "us")
    with_hash = parse_address(normalize_address("123 Main St, # 1, Springfield, IL"), "us")
    assert with_word["unit"] == "unit 16"
    assert with_hash["unit"] == ""
    assert with_hash["house"] == "123 1"


@pytest.mark.parametrize(
    ("a_raw", "b_raw", "country"),
    [
        ("1064 Newton Road, Unit 11, Iowa City, IA", "IA, Iowa City, 1064 Newton Rd, Unit 11", "us"),
        ("Plot No. 72, Varanasi, UP", "UP, Varanasi, Plot No. 72", "india"),
    ],
)
def test_parse_address_reordered_components_agree(a_raw, b_raw, country):
    """Reordered comma components (as seen between sources) still classify the same way."""
    a = parse_address(normalize_address(a_raw), country)
    b = parse_address(normalize_address(b_raw), country)
    assert a == b


def test_parse_address_france_degrades_gracefully():
    """An unseen-in-training country (France) never crashes; without a learned admin_vocab the région
    falls back into `city` (no hand-written département/région map: D-011, D-015). Passing a learned
    admin_vocab (as learn_admin_vocab would produce) correctly separates it into `admin`."""
    addr = "30 bis rue rene delissen , dunkerque , hauts de france"
    no_vocab = parse_address(addr, "france")
    assert no_vocab["house"] == "30 bis"
    assert no_vocab["street"] == "rue rene delissen"
    assert no_vocab["admin"] == ""
    assert "hauts de france" in no_vocab["city"]

    with_vocab = parse_address(addr, "france", frozenset({"hauts de france"}))
    assert with_vocab["admin"] == "hauts de france"
    assert with_vocab["city"] == "dunkerque"


def test_parse_address_determinism_and_empty():
    """parse_address is deterministic and handles '' cleanly."""
    addr = normalize_address("105 Elm St, Morganton, NC")
    assert parse_address(addr, "us") == parse_address(addr, "us")
    assert parse_address("", "") == {"house": "", "unit": "", "street": "", "city": "", "admin": ""}


def test_learn_admin_vocab_synthetic():
    """learn_admin_vocab discovers a recurring, mostly-edge-position component (a French région) from
    address data alone, with no hand-written région/département table (D-011, D-015)."""
    addrs = (
        ["123 main street , springfield , hauts de france"] * 250
        + ["45 rue de la paix , paris , hauts de france"] * 10
        + ["30 bis rue rene delissen , dunkerque , hauts de france"] * 5
        + ["7 impasse des fleurs , lille , nord"] * 5  # below min_count: must not be kept
    )
    vocab = learn_admin_vocab(addrs, min_count=200, max_tokens=3)
    assert vocab == {"hauts de france"}


def test_learn_admin_vocab_ignores_numeric_and_mid_position():
    """Numeric-only components and components that never sit at an edge are not admin candidates."""
    addrs = ["1 2 3 , middle town , north zone , 44000"] * 500
    vocab = learn_admin_vocab(addrs, min_count=100, max_tokens=3)
    assert "middle town" not in vocab  # a middle component, not first/last
    assert "1 2 3" not in vocab  # numeric
    assert "44000" not in vocab  # numeric


def test_learn_admin_vocab_empty():
    """learn_admin_vocab handles an empty/blank iterable without error."""
    assert learn_admin_vocab([], min_count=1) == set()
    assert learn_admin_vocab(["", None, ""], min_count=1) == set()


# --------------------------------------------------------------------------------------------------
# v2: learned transliteration token map (D-012)
# --------------------------------------------------------------------------------------------------


def test_learn_token_map_synthetic_indic_abbreviation():
    """learn_token_map recovers an unmapped transliterated Indic legal-form abbreviation from repeated
    true pairs (D-012). "prai li" stands in for a Hindi/Devanagari abbreviation like 'प्रा. लि.' romanized
    slightly differently from the one form ('pra li') already hand-mapped, so this exercises the learned
    path rather than the existing hard-coded phrase."""
    pairs = [("shivam infotech pvt ltd", "shivam infotech prai li")] * 60
    token_map = learn_token_map(pairs, min_support=50, min_precision=0.9)
    assert token_map["prai li"] == "pvt ltd"
    assert apply_token_map("shivam infotech prai li", token_map) == "shivam infotech pvt ltd"


def test_learn_token_map_realistic_devanagari_input():
    """A real (phonetically transliterated) Devanagari name aligned against its Latin S1 counterpart:
    learn_token_map recovers not just the legal-form abbreviation ("li" -> "ltd") but the phonetic content
    words too ("sivm" -> "shivam"), and applying it reconstructs the S1 spelling exactly."""
    left = normalize_name("Shivam Infotech Pvt Ltd")
    right = normalize_name("शिवम इंफोटेक प्रा. लि.")
    assert right.endswith("ltd")  # already hits the existing "pra li" -> "pvt ltd" hard-coded phrase
    token_map = learn_token_map([(left, right)] * 60, min_support=50, min_precision=0.9)
    assert apply_token_map(right, token_map) == left


def test_learn_token_map_respects_min_support_and_precision():
    """A mapping seen fewer than min_support times, or an inconsistent one, is dropped."""
    rare = [("acme pvt ltd", "acme prai li")] * 10  # below default min_support=50
    assert learn_token_map(rare) == {}

    inconsistent = [("acme pvt ltd", "acme xyz")] * 30 + [("beta corp", "beta xyz")] * 30
    # "xyz" aligns inconsistently across pairs with unrelated left contexts; support per single mapping
    # stays below 50 for at least one of the two targets, so it must not pass min_support=50.
    token_map = learn_token_map(inconsistent, min_support=50, min_precision=0.9)
    assert "xyz" not in token_map


def test_learn_token_map_empty():
    """learn_token_map handles an empty iterable and blank names without error."""
    assert learn_token_map([]) == {}
    assert learn_token_map([("", ""), ("acme pvt ltd", "")]) == {}


def test_apply_token_map_longest_match_first():
    """apply_token_map prefers a multi-word key over a shorter key that is its prefix, and is a no-op for
    an empty map or empty name."""
    token_map = {"pra": "pvt", "pra li": "pvt ltd"}
    assert apply_token_map("acme pra li", token_map) == "acme pvt ltd"
    assert apply_token_map("acme pra co", token_map) == "acme pvt co"
    assert apply_token_map("acme pra li", None) == "acme pra li"
    assert apply_token_map("acme pra li", {}) == "acme pra li"
    assert apply_token_map("", token_map) == ""


# --------------------------------------------------------------------------------------------------
# v2: wiring into normalize_batch / normalize_parquet
# --------------------------------------------------------------------------------------------------


def test_normalize_batch_v2_columns():
    """normalize_batch adds legal_form/type_words/addr_* columns and applies a token_map before name_core."""
    table = pa.table(
        {
            "entity_id": ["S1-1", "S2-2"],
            "business_name": ["Shivam Infotech Pvt Ltd", "Shivam Infotech Prai Li Ventures"],
            "business_address": ["Plot No. 72, Varanasi, UP", "#563/564, Bangalore North, Bangalore, KA"],
            "country": ["India", "India"],
        }
    )
    token_map = {"prai li": "pvt ltd"}
    out = normalize_batch(table, token_map=token_map).to_pylist()
    assert out[0]["legal_form"] == "private limited"
    assert out[0]["addr_house"] == "72"
    assert out[0]["addr_city"] == "varanasi"
    assert out[0]["addr_admin"] == "up"
    # the token map is applied before name_core/name_skel, so "prai li" resolves to "pvt ltd" and is then
    # stripped from name_core by the existing legal-form logic, same as a literal "pvt ltd" would be
    assert out[1]["name_core"] == "shivam infotech ventures"
    assert out[1]["type_words"] == "ventures"
    assert out[1]["legal_form"] == "private limited"
    # name_norm itself is left unmapped
    assert "prai li" in out[1]["name_norm"]
    assert out[1]["addr_house"] == "563 564"
    assert out[1]["addr_admin"] == "ka"


def test_normalize_batch_v2_columns_default_empty():
    """Without a token_map/admin_vocab, the new columns are still populated from the hand-written maps."""
    out = normalize_batch(_raw_table()).to_pylist()
    assert out[0]["legal_form"] == "private limited"  # "Ram Marketing Pvt Ltd"
    assert out[1]["addr_admin"] == "dl"  # address is just "Delhi" -> normalize_address -> "dl"
    assert out[3]["legal_form"] == "" and out[3]["addr_house"] == ""  # all-empty input row


@pytest.mark.parametrize("workers", [1, 2])
def test_normalize_parquet_with_token_map_and_admin_vocab(tmp_path, workers):
    """normalize_parquet loads JSON token_map / admin_vocab files and passes them through to every worker."""
    t = pa.concat_tables([_raw_table()] * 3)
    src, dst = tmp_path / "in.parquet", tmp_path / "out.parquet"
    pq.write_table(t, src)
    tm_path = tmp_path / "token_map.json"
    tm_path.write_text(json.dumps({"prai li": "pvt ltd"}))
    av_path = tmp_path / "admin_vocab.json"
    av_path.write_text(json.dumps(["hauts de france"]))
    n = normalize_parquet(
        str(src), str(dst), workers=workers, batch_rows=3, token_map_path=str(tm_path), admin_vocab_path=str(av_path)
    )
    out = pq.read_table(dst)
    assert n == out.num_rows == t.num_rows
    expected = normalize_batch(t, {"prai li": "pvt ltd"}, frozenset({"hauts de france"}))
    assert out.equals(expected)


def test_normalize_parquet_without_maps_matches_normalize_batch(tmp_path):
    """normalize_parquet with no token_map_path/admin_vocab_path behaves exactly like before (backward compat)."""
    t = pa.concat_tables([_raw_table()] * 2)
    src, dst = tmp_path / "in.parquet", tmp_path / "out.parquet"
    pq.write_table(t, src)
    n = normalize_parquet(str(src), str(dst), workers=1, batch_rows=5)
    out = pq.read_table(dst)
    assert n == out.num_rows
    assert out.equals(normalize_batch(t))
