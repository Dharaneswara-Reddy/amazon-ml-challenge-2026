"""Table-driven tests for ber.normalize, built from patterns observed in the training/test samples."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rapidfuzz import fuzz

from ber.normalize import (
    OUTPUT_SCHEMA,
    address_numbers,
    detect_script,
    extract_postal,
    name_core,
    name_skeleton,
    normalize_address,
    normalize_batch,
    normalize_name,
    normalize_parquet,
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
