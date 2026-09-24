# Data profile (measured)

Produced by `scripts/profile_data.py` (DuckDB, 700 MB cap). Numbers are from the full train set unless noted.

## Size

| Split | Source 1 | Source 2 | Source 3 | S2+S3 per S1 |
|---|---|---|---|---|
| Train | 2,206,821 | 5,034,616 | 5,285,603 | 4.68 |
| Test | 1,732,544 | 4,887,273 | 5,082,316 | 5.76 |

~2.4 GB of TSV. Test has ~23% more right-side records per Source 1 entity than train — either higher match
multiplicity or more unmatched distractors. Thresholds tuned on train may be optimistic on precision.

## Ground truth structure

- 2,206,821 GT rows (one per train S1), 7,638,365 true pairs.
- **Every S2/S3 record matches at most one S1 entity** (0 of 7.64 M right-side IDs appear under two S1s).
  → the task is "attach each right-side record to its best S1 or to none"; one-to-one assignment on the right side is exact.
- Singleton rate: **5.6%**. Matches per non-singleton S1: 1 → 5.7%, 2 → 18%, 3 → 25%, 4 → 23%, 5 → 15%, 6 → 8%, 7+ → 4% (max 11).
- Typical composition: 1–3 from S2 and 1–4 from S3; ~11% of S1s match only one source.
- 73.4% of S2 and 74.6% of S3 records are matched; ~26% are unmatched distractors.

## True-pair similarity (200k sampled pairs)

- Same country: **100%** → country is a safe blocking partition.
- Case-insensitive identical names: only 10.7%. Mean Jaro-Winkler (lowercased raw): name 0.868, address 0.840.

## Noise observed in true pairs

- Names: legal-suffix drift and reordering ("Win Hydro Pvt Ltd" ↔ "Dr Win Hydro Ltd Pvt"), injected accents
  ("Córp", "Nétwork"), digit-for-letter ("C0"), brackets ("Perry's [Fitness]"), truncation ("Primary Care Pacific Ci"),
  concatenation ("DennisfamilypracticeCom"), prefixes ("Dr", "Service"), person-name reordering ("King, Pearla D.O."),
  fully different trade names ("Ram Services Private Limited" ↔ "Lumlum").
- Scripts: Devanagari and **Bengali** names; Devanagari state names inside addresses ("मध्य प्रदेश").
- Addresses: case, component reordering, state abbreviation vs full name (TN/Tennessee, MH, KA, WB, "Raj."),
  street-type abbreviations and typos ("Mian St", "OAK VALLY DR"), added unit/PMB/PO Box, missing house number,
  **perturbed house numbers between true matches** ("7619" vs "619", "10756" vs "10796"), empty address on one side.
