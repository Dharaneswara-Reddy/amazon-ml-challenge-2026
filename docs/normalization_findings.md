# Normalization findings (ber.normalize)

All numbers come from training/test samples only. No external data, word lists or services were used;
every map in `normalize.py` is hand-written and was checked against these samples.

## Samples

| Sample | Size | How it was built |
|---|---|---|
| True pairs | 69,542 pairs (20,000 random GT rows, exploded) | S1 record joined to each matched S2/S3 record |
| Hard negatives | 54,678 pairs (18,576 S1) | S2/S3 records in the same country that share the S1 name's first token (≥3 chars) but are not matched; 30% Bernoulli sample of S2/S3, ≤3 per S1 |
| Raw rows | 60,000 per source per split (train + test) | reservoir sample; the test samples cover France |

Location: `artifacts/tmp/norm/*.parquet`.

## Noise catalogue (60k random rows per source, train)

| Pattern | S1 | S2 | S3 |
|---|---|---|---|
| Indic script in name (any) | 0% | ≥9.1% (Devanagari 5.3, Telugu 0.85, Kannada 0.74, Tamil 0.65, Gujarati 0.64, Bengali 0.59, Malayalam 0.34) | ≥5.1% (Devanagari 3.0, others 2.1) |
| Indic script in address | 0% | ≥9.3% (Devanagari 5.4; also Oriya, Gurmukhi…), mostly the state component ("महाराष्ट्र") | ≥8.8% |
| Leading junk (`@ # -- >> ... *** ( [ &`) | 0.15% | 2.2% | 2.2% |
| Brackets in name | 2.1% | 7.7% | 8.1% |
| Domain-style name (`x.com`, `…Com`) | 0% | 4.2% | 3.9% |
| Letter/digit swaps (`Br0thers`, `Techno1ogy`, `8allard`) | 0.03% | 2.8% | 2.8% |
| All-caps name / address | 0% / 0% | 27.9% / 63.2% | 7.6% / 0% |
| Address placeholder component (`N/A`, `NULL`, `null`, `<NULL>`) | ~0.07% | 3.5% | 3.2% |
| Empty address | 0% | 3.5% | 3.4% |
| Landmark words (near/opp/behind/beside) | 5.3% | 4.7% | 3.5% |

Seen on true pairs:
- Alias markers (dba, d/b/a, aka, a/k/a, t/a, trading as, doing business as, formerly [known as], f/k/a): in
  764 of 764 checked cases the S1 name is the text after the marker. The text before it is a made-up brand.
- Tokens added on the S2/S3 side (count of pairs): center 2124, services 1634, service 689, the 593,
  mr 372, smt 363, dr 360, sri 356, shri 337, formerly 284, `(ID: n)` tags, and duplicated suffixes
  ("Corp Corp").
- Legal forms dropped or swapped: limited 5872, llc 3660, inc 2953, private 2781, ltd 1556.
- Which digits replace letters: 0 (914 tokens), 1 (565), 5 (321), 8 (163), 6 (116), 3 (27).
- State written differently on each side. India: S1 uses full names and S2/S3 use codes (MH 1864,
  DL 1100, UP 644, KA 631, TN 589, GJ 503, WB 489, TG 402…). US: S1 uses codes and S2/S3 use full names
  (Texas 1978, Ohio 1267…). City renames also appear (Poona↔Pune, 68 pairs).
- France (test only): S2 addresses are upper case, with abbreviations `R`/`R.`→rue, `AV`, `All.`, `Ch.`, `St-`,
  and `N°`/`Nº`/`#`. Legal forms are SARL/SAS/SASU/EURL/SCI/SA/EI/SNC, sometimes dotted (`S.A.R.L.`).
  `(France)` is injected noise, and `Cie`/`Ets` are abbreviations. Region and département are
  interchanged (Hauts-de-France vs Nord).

## Transliteration (2,711 distinct Indic↔Latin true name pairs, 9 scripts)

Score: mean rapidfuzz `token_set_ratio` / `ratio` after lowercasing and replacing punctuation with spaces.

| Method | tsr | ratio | Tamil | Devanagari |
|---|---|---|---|---|
| raw (no transliteration) | 13.0 | 8.2 | 10 | 14 |
| indic_transliteration ITRANS | 67.9 | 72.4 | 46 | 69 |
| HK | 67.6 | 72.1 | 47 | 69 |
| IAST + accent fold | 68.1 | 72.6 | 48 | 69 |
| OPTITRANS | 68.6 | 72.4 | 46 | 71 |
| IAST + simplification (schwa drop, ṃ→n, aa→a, ph→f, w→v) | 74.2 | 77.2 | 48 | 77 |
| IAST + medial-schwa deletion | 72.5 | 76.4 | 47 | 75 |
| anyascii | 74.1 | 77.3 | 63 | 75 |
| **anyascii + anusvara→n (chosen)** | **75.4** | **78.5** | 63 | 77 |

Chosen method: anyascii (ISC license). It is script-generic, including Tamil, it already deletes the
inherent schwa, and it is the fastest (0.1 s for all pairs vs 0.2–0.3 s). The anusvara fix adds +1.6
tsr inside the full pipeline (78.6 → 80.2). Full `normalize_name` on these pairs scores tsr 80.2.

The remaining gap is phonetic ("ड्रीम पावर" → "drim pavr" vs "dream power"). `name_skeleton` handles it
with a consonant skeleton: keep each token's first letter, drop later vowels and non-initial h, and
merge c/k/q, w/v, ph/f, z/j and doubled letters. The table compares `name_core` with and without it.

| Pairs | name_core ratio (≥90 share) | skeleton(name_core) ratio (≥90 share) |
|---|---|---|
| Indic↔Latin positives | 73.8 (0.087) | **92.1 (0.659)** |
| Latin positives (15k) | 91.1 (0.725) | 91.0 (0.721) |
| Hard negatives (15k) | 58.4 (0.031) | 59.2 (0.032) |

## Before and after: positives vs hard negatives (20k pairs each)

Names, `token_set_ratio`:

| Variant | pos | neg | gap | pos ≥90 | neg ≥90 |
|---|---|---|---|---|---|
| baseline (lowercase + punctuation) | 86.8 | 63.1 | 23.8 | 0.718 | 0.061 |
| normalize_name | 92.5 | 64.4 | 28.1 | 0.767 | 0.062 |
| name_core | 92.0 | 64.6 | 27.4 | 0.771 | 0.086 |
| name_core + generic words (center/services/group/holdings/partners) | 93.7 | 68.0 | 25.6 | 0.830 | **0.166** |

Names, `ratio` (order-sensitive):

| Variant | pos | neg | gap | pos ≥90 | neg ≥90 |
|---|---|---|---|---|---|
| baseline | 79.8 | 57.3 | 22.5 | 0.441 | 0.014 |
| normalize_name | 86.8 | 58.3 | 28.5 | 0.536 | 0.017 |
| name_core | 90.0 | 58.5 | 31.5 | 0.683 | 0.031 |
| − token map (legal/abbrev canonicalization) | 87.9 | 57.8 | 30.0 | 0.585 | 0.026 |
| − alias handling | 89.3 | 58.5 | 30.9 | 0.664 | 0.031 |
| − leading honorific removal | 89.6 | 58.6 | 31.0 | 0.666 | 0.031 |
| − digit-swap fix | 89.8 | 58.4 | 31.4 | 0.681 | 0.031 |

Decision: generic words stay in `name_core`. Removing them doubles the share of hard negatives scoring
≥90 (0.086 → 0.166), which would hurt precision under F0.5. With `token_set_ratio`, `name_core` also
raises that share (0.062 → 0.086), because a shorter name more often becomes a subset of the other.
Use `name_core` with order-sensitive or set-size-aware scorers, and keep `name_norm` as a separate
feature.

Addresses, `token_set_ratio`:

| Variant | pos | neg | gap | pos ≥90 | neg ≥90 |
|---|---|---|---|---|---|
| baseline | 87.5 | 36.5 | 51.0 | 0.638 | 0.005 |
| normalize_address | 93.4 | 39.9 | 53.5 | 0.869 | 0.011 |
| − state canonicalization | 90.4 | 39.9 | 50.5 | 0.770 | 0.008 |
| − street/landmark abbreviation map | 92.5 | 39.7 | 52.8 | 0.852 | 0.011 |
| − city aliases | 93.3 | 39.9 | 53.4 | 0.867 | 0.011 |

## Postal codes

Postal codes are almost absent from this data. 5-digit tokens occur in 6.7% of S1 addresses, but they
are house numbers ("10103 Kirsch Rd", "21100 68"). 6-digit tokens are mostly zero-padded house numbers
("002160 MUREN BLVD"), and "PIN" appears in 7 of 50k S1 rows. `extract_postal` accepts only a keyword
("PIN 560 001") or a 5–6 digit number that ends its comma component and has no leading 0.

| Split | S1 | S2 | S3 |
|---|---|---|---|
| train US / India | 0.2% / 0.1% | 0.0% / 0.1% | 0.1% / 0.1% |
| test US / India / France | 0.1% / 0.1% / 0.2% | 0.0% / 0.1% / 0.2% | 0.1% / 0.1% / 0.2% |

On true pairs where both sides have a code, the codes agree in 70 of 70 cases. A looser first version
(any standalone 5–6 digit number not followed by a word) agreed in only 184 of 191 cases, because it
picked up house numbers. Treat `postal` as a rare bonus feature. House numbers are perturbed between
true matches ("7619" vs "619"), so `addr_nums` conflicts are soft evidence.

## Throughput

Single process, cold cache, 50k-row samples: 34k rows/s (train S1), 28k rows/s (train S2), 30k rows/s
(test S3). `normalize_parquet` with 2 workers and 10k-row batches: 39k rows/s on 50k rows, including
process spawn. Whole-string regexes guarded by cheap `in` checks gave about a 4× speedup over
per-component regexes. Word-level transliteration and whole-string results are cached with LRU caches.
The pool uses a bounded window of 2×workers in-flight batches, because `Pool.imap` would read the whole
input eagerly.

## Output columns

`entity_id, name_norm, name_core, name_skel, addr_norm, postal, addr_nums, script, country_norm`.
`addr_norm` keeps the comma components in their original order, joined by " , ".
