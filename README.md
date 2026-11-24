# SKAS – Seed-Keyed Attribute Splitting (Family 1: storage-only)

SKAS hides a sensitive column inside ordinary-looking columns.
With the right seed the original values are rebuilt exactly; with a wrong
seed you get realistic but fake values (honey), so an attacker cannot tell
a correct guess from a wrong one.

## How Family 1 works
1. **Encode** – the value is turned into a code with a histogram encoder
   (common values get more codes), so random codes decode to realistic values.
2. **Mask** – a secret per-row mask from the seed is added: `w = (code + mask) mod 2^24`.
3. **Split** – `w` is cut into 3 pieces of 8 bits each.
4. **Disguise** – each piece becomes a realistic offset from a real column:
   * Retail (hidden: Price) → `PickedAt`, `PackedAt`, `ScannedAt` (whole-minute warehouse times)
   * Taxi (hidden: end latitude) → `Waypoint_LAT`, `Waypoint_LON`, `DispatchTime`

We tested 2 vs 3 columns; 3 columns gave better honey and more natural disguises.

**Safety features**
* **Versioned masks** – mask = PRF(seed, version ‖ RowKey). An updated value is re-protected
  with version+1, so old and new rows never share a mask.
* **Integrity tags** – a per-row AES-CBC-MAC under a *separate* random key covers RowKey,
  version, the disguised columns and their base columns. It detects any edited cell and,
  because it never uses the seed, cannot be used to test seed guesses (honey stays intact).
  Tags are kept in a side table, not in the disguised table.
* **Stable keys** – masks use a permanent `RowKey`, not the row position, so reordering the
  table is harmless. (Taxi `TRIP_ID` is not used: 11 IDs appear twice.)

## Datasets (not included – download them into this folder)
| Dataset | File | Source |
|---|---|---|
| Online Retail II | `online_retail_II.xlsx` | https://archive.ics.uci.edu/dataset/502/online+retail+ii |
| Porto Taxi (ECML/PKDD 2015) | `train.csv` | https://archive.ics.uci.edu/dataset/339/taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015 |

On the first taxi run, `train.csv` (1.9 GB) is shrunk to `taxi_small.csv.gz`
(GPS lists dropped, start/end points kept). This happens once.

## Run
```
pip install -r requirements.txt
python family1_storage_only.py retail     # rebuild, timing, honey, realism checks
python family1_storage_only.py taxi
python stealth_test.py                    # can an attacker find the hidden columns?
```

## Results (10 samples × 5,000 rows)

**Family 1** – exact rebuild 100% on both datasets; ~0.1 µs/row split, ~0.2 µs/row rebuild.
Honey: most common real price is 12.16% of rows; under wrong seeds 12.16–12.20%.

| Check (20,000 updated rows / 1,000 tampered rows) | Retail | Taxi |
|---|---|---|
| Attacker who knew old values guesses new ones – old design (same mask) | 63.8% | 65.9% |
| Same attack – versioned masks | 5.2% (chance) | 0.0% |
| Tampered rows detected by integrity tag | 100% | 100% |
| Rebuild exact after shuffling rows | yes | yes |

**Stealth** – share of the attacker's guesses that are correct (lower = better hidden), ML classifier:

| Scheme | Retail (chance) | Retail | Taxi (chance) | Taxi |
|---|---|---|---|---|
| **SKAS** | 0.27 | **0.60** | 0.21 | **0.33** |
| Rahmani 2023a | 0.42 | 0.80 | 0.33 | 0.72 |
| Rahmani 2022 | 0.20 | 0.95 | 0.15 | 0.55 |
| SRSOSE | 0.43 | 1.00 | 0.35 | 0.93 |
| AES | 0.11 | 1.00 | 0.08 | 0.70 |

Rule-based detectors (entropy, independence) find none of the SKAS columns on retail,
while SRSOSE and AES are caught 100% by the independence test.

**Known limitation:** SKAS values are almost all unique, while real per-order times repeat,
so a trained classifier still beats chance on retail.

Baselines are our re-implementations from the papers (no public code exists).
