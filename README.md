# SKAS – Seed-Keyed Attribute Splitting (Family 1: storage-only)

SKAS hides a sensitive column inside ordinary-looking columns.
With the right seed the original values are rebuilt exactly; with a wrong
seed you get realistic but fake values (honey), so an attacker cannot tell
a correct guess from a wrong one.

## How Family 1 works
1. **Encode** – the value is turned into a code with a histogram encoder
   (common values get more codes), so random codes decode to realistic values.
2. **Mask** – a secret per-row mask is added: `w = (code + mask) mod 2^24`,
   with `mask = AES(key from seed, version ‖ RowKey)`.
3. **Split** – `w` is cut into 3 pieces of 8 bits each.
4. **Disguise** – each piece becomes a realistic offset from a real column:
   * Retail (hidden: Price) → `PickedAt`, `PackedAt`, `ScannedAt` (whole-minute warehouse times)
   * Taxi (hidden: end latitude) → `Waypoint_LAT`, `Waypoint_LON`, `DispatchTime`

We tested 2 vs 3 columns; 3 columns gave better honey and more natural disguises.

**Safety features**
* **Versioned masks** – an updated value is re-protected with version+1, so old and
  new rows never share a mask.
* **Integrity tags** – a per-row AES-CBC-MAC under a *separate* random key covers RowKey,
  version, the disguised columns and their base columns. It detects any edited cell and,
  because it never uses the seed, cannot be used to test seed guesses (honey stays intact).
  Tags are kept in a side table, not in the disguised table.
* **Stable keys** – masks use a permanent `RowKey`, not the row position, so reordering the
  table is harmless. (Taxi `TRIP_ID` is not used: 11 IDs appear twice.)

## Requirements
* Python 3.10 or newer
* `pip install -r requirements.txt`

## Datasets (not included – download them into this folder)
| Dataset | File | Source |
|---|---|---|
| Online Retail II | `online_retail_II.xlsx` | https://archive.ics.uci.edu/dataset/502/online+retail+ii |
| Porto Taxi (ECML/PKDD 2015) | `train.csv` | https://archive.ics.uci.edu/dataset/339/taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015 |

On the first taxi run, `train.csv` (1.9 GB) is shrunk to `taxi_small.csv.gz`
(GPS lists dropped, start/end points kept). This happens once and takes a few minutes.

## Run
```
python family1_storage_only.py retail taxi   # ~1–3 min: rebuild, honey, updates, integrity
python stealth_test.py                       # ~8–15 min: can an attacker find the hidden columns?
```
The first retail run is slower (reading the Excel file); later runs use a cache.
`stealth_test.py` options: `--seeds 0 1 2 3 4` (default) and `--reps 10` (default).
The same seeds and reps give an identical table on the same machine; on another machine
(other OS or library versions) a few taxi numbers can differ by about ±0.05.

## Expected output
`family1_storage_only.py` – for each dataset:
* `[1] Exact rebuild with the right seed: True (100.0000% of rows)`
* `[3]` the share of each common value under wrong seeds matches the real share (honey)
* `[6]` with versioned masks, an attacker who knew the old values guesses new ones at chance level
* `[7]` `detected 100.0%` for both tampering tests
* `[8]` `rebuild by RowKey exact = True`

Timings and the exact honey percentages change slightly from run to run (new random keys).

`stealth_test.py` – the tables below (with the default seeds; tiny differences across machines are normal).

## Results

**Family 1** – exact rebuild 100% on both datasets; ~0.1 µs/row split, ~0.2 µs/row rebuild.
Honey: most common real price is 12.16% of rows; under wrong seeds 12.10–12.20%.

| Check (20,000 updated rows / 1,000 tampered rows) | Retail | Taxi |
|---|---|---|
| Attacker who knew old values guesses new ones – old design (same mask) | 64% | 66% |
| Same attack – versioned masks | 5% (chance) | 0% |
| Tampered rows detected by integrity tag | 100% | 100% |
| Rebuild exact after shuffling rows | yes | yes |

**Stealth** – share of the attacker's guesses that are correct (lower = better hidden).
ML classifier trained on the other dataset; 5 seeds × 10 samples of 5,000 rows;
mean, with the range over seeds in brackets.

| Scheme | Retail (chance 0.27*) | Taxi (chance 0.21*) |
|---|---|---|
| **SKAS** | **0.60** (0.53–0.67) | **0.41–0.43** (0.33–0.63) |
| Rahmani 2023a | 0.80 | 0.75 |
| Rahmani 2022 | 1.00 | 0.92 |
| SRSOSE | 1.00 | 0.99 |
| AES | 1.00 | 1.00 |

\*Chance is for SKAS; it differs per scheme because each adds a different number of columns.

Rule-based detectors (entropy, independence) find none of the SKAS columns on retail,
while SRSOSE and AES are caught 100% by the independence test.

**Why results depend on the seed:** each seed trains a new ML attacker, and how well it
transfers from one dataset to the other varies. That is why results are averaged over seeds.

**Known limitation:** SKAS values are almost all unique, while real per-order times repeat,
so a trained classifier still beats chance on retail.

Baselines are our re-implementations from the papers (no public code exists).
