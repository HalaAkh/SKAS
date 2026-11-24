"""SKAS - Family 1 (storage-only), standalone version.

What it does
  1. Loads a dataset and picks one sensitive column (retail: Price, taxi: END_LAT).
  2. Hides that column in three ordinary-looking columns, made from a secret
     seed and the row number (retail: PickedAt / PackedAt / ScannedAt
     warehouse times for each order line; taxi: a waypoint latitude and longitude near the pickup point and a
     dispatch time shortly before the trip starts).
  3. Rebuilds the column with the right seed and checks it is 100% exact.
  4. Rebuilds it with wrong seeds and checks the result looks realistic (honey).
  5. Prints timings, checks that the disguised columns have natural offsets,
     and saves the first rows of the protected table to CSV.

Setup (once):
    pip install numpy pandas cryptography openpyxl

Run (in the folder that contains the data files):
    python family1_storage_only.py retail     # needs online_retail_II.xlsx
    python family1_storage_only.py taxi       # needs train.csv (or taxi_small.csv.gz)
"""
from __future__ import annotations

import hashlib
import hmac
import os
import sys
import time

import numpy as np
import pandas as pd
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

PIECES = 3              # each secret value is stored as 3 disguised columns
BITS = 8                # each piece carries 8 bits, i.e. 256 possible values
B = 1 << BITS           # values per piece (256)
M = 1 << (PIECES * BITS)  # size of the code space (16,777,216)


# =============================================================== 1. secret tools
def new_seed() -> bytes:
    """The administrator's secret seed (32 random bytes)."""
    return os.urandom(32)


def derive(seed: bytes, label: str) -> bytes:
    """A separate 16-byte key for one purpose, made from the seed."""
    return hmac.new(seed, b"skas-kdf|" + label.encode(), hashlib.sha256).digest()[:16]


def row_masks(key: bytes, row_ids, modulus: int) -> np.ndarray:
    """One secret number per row: AES(key, row id) reduced to 0..modulus-1."""
    ids = np.asarray(row_ids, dtype=np.uint64)
    blocks = np.zeros((len(ids), 2), dtype=np.uint64)
    blocks[:, 1] = ids
    enc = Cipher(algorithms.AES(key), modes.ECB()).encryptor()
    out = np.frombuffer(enc.update(blocks.tobytes()) + enc.finalize(), dtype=np.uint64)
    return (out.reshape(-1, 2)[:, 0] % np.uint64(modulus)).astype(np.int64)


def fresh_random(n: int, modulus: int) -> np.ndarray:
    """Truly random numbers (not from the seed), used for the first share."""
    return (np.frombuffer(os.urandom(8 * n), dtype=np.uint64) % np.uint64(modulus)).astype(np.int64)


# =============================================================== 2. value <-> code
class HistogramDTE:
    """Turns a value into a code in 0..M-1. Common values own more codes, so a
    random code turns back into a realistic value (this gives the honey effect)."""

    def __init__(self, values, M: int):
        vals, counts = np.unique(np.asarray(values), return_counts=True)
        raw = counts / counts.sum() * M
        slots = np.maximum(1, np.floor(raw).astype(np.int64))
        extra = M - slots.sum()
        if extra > 0:
            slots[np.argsort(-(raw - np.floor(raw)))[:extra]] += 1
        while slots.sum() > M:
            slots[np.argmax(slots)] -= 1
        self.vals, self.slots, self.M = vals, slots, M
        self.starts = np.concatenate([[0], np.cumsum(slots)[:-1]])

    def encode(self, values, rnd):
        idx = np.searchsorted(self.vals, values)
        return self.starts[idx] + (np.asarray(rnd) % self.slots[idx])

    def decode(self, codes):
        idx = np.searchsorted(self.starts, np.asarray(codes) % self.M, side="right") - 1
        return self.vals[idx]


# =============================================================== 3. disguise
class OffsetDecoy:
    """Turns a number 0..M-1 into 'base column + a realistic offset', for example
    ShippedAt = InvoiceDate + delay. Every number gets its own distinct offset."""

    def __init__(self, M: int, example_offsets, step=1):
        q = (np.arange(M) + 0.5) / M
        d = np.round(np.quantile(np.asarray(example_offsets, float), q) / step) * step
        ramp = np.arange(M) * step
        self.deltas = np.maximum.accumulate(d - ramp) + ramp  # strictly increasing
        self.M, self.step = M, step

    def encode(self, x, base):
        return np.asarray(base) + self.deltas[np.asarray(x) % self.M]

    def decode(self, stored, base):
        d = np.asarray(stored, float) - np.asarray(base, float)
        return np.clip(np.searchsorted(self.deltas, d - self.step / 2), 0, self.M - 1)


# =============================================================== 4. Family 1
class StorageOnly:
    """Hide a value in PIECES columns.

    1. value -> code u (HistogramDTE)
    2. add this row's secret mask:  w = u + mask(row)   (mod M)
       Without the seed, w is just a random-looking number.
    3. cut w into PIECES small pieces of BITS bits each (like digits of a number)
    4. disguise each piece as an ordinary column (base column + realistic offset).
    Because each piece has only 256 possible values, its disguise can follow a
    natural, realistic shape instead of being spread out evenly.
    """

    def __init__(self, name, dte, decoys, bases):
        self.name, self.dte, self.decoys, self.bases = name, dte, decoys, bases
        # bases[j] = name of a real column, or "prev" = the previous disguised column

    def _base(self, j, table, cols):
        return cols[j - 1] if self.bases[j] == "prev" else table[self.bases[j]]

    def split(self, seed, row_ids, values, table):
        k = row_masks(derive(seed, self.name + "|mask"), row_ids, M)     # secret mask per row
        u = self.dte.encode(values, fresh_random(len(values), 1 << 30))  # value -> code
        w = (u + k) % M                                                  # masked code
        cols = []
        for j in range(PIECES):
            piece = (w >> (BITS * j)) & (B - 1)                          # j-th piece
            cols.append(self.decoys[j].encode(piece, self._base(j, table, cols)))
        return cols

    def recover(self, seed, row_ids, cols, table):
        w = np.zeros(len(row_ids), dtype=np.int64)
        for j in range(PIECES):
            piece = self.decoys[j].decode(cols[j], self._base(j, table, cols))
            w |= piece.astype(np.int64) << (BITS * j)
        k = row_masks(derive(seed, self.name + "|mask"), row_ids, M)
        return self.dte.decode((w - k) % M)


# =============================================================== 5. datasets
def shrink_train_csv(src="train.csv", out="taxi_small.csv.gz"):
    """First run only: turn the 1.9 GB train.csv into a small file (drops the GPS
    list, keeps the number of GPS points and the start/end coordinates)."""
    import json
    print("  taxi_small.csv.gz not found; building it from train.csv (a few minutes, once) ...")
    first, total = True, 0
    for chunk in pd.read_csv(src, chunksize=100_000):
        pts = [json.loads(p) if isinstance(p, str) else [] for p in chunk["POLYLINE"]]
        chunk["N_POINTS"] = [len(p) for p in pts]
        chunk["START_LAT"] = [p[0][1] if p else None for p in pts]
        chunk["START_LON"] = [p[0][0] if p else None for p in pts]
        chunk["END_LAT"] = [p[-1][1] if p else None for p in pts]
        chunk["END_LON"] = [p[-1][0] if p else None for p in pts]
        chunk.drop(columns=["POLYLINE"]).to_csv(out, mode="w" if first else "a", header=first,
                                                 index=False, compression="gzip")
        first, total = False, total + len(chunk)
        print(f"    {total:,} rows", end="\r")
    print(f"    done: {total:,} rows")



def load(name):
    rng = np.random.default_rng(7)
    if name == "retail":
        d = pd.concat(pd.read_excel("online_retail_II.xlsx", sheet_name=None).values(), ignore_index=True)
        d = d[d["Customer ID"].notna() & (d.Quantity > 0) & (d.Price > 0)].reset_index(drop=True)
        table = pd.DataFrame({"Invoice": d.Invoice.astype(str),
                              "InvoiceDate": d.InvoiceDate.astype("datetime64[s]").astype(np.int64),
                              "Price_mills": np.round(d.Price * 1000).astype(np.int64)})
        secret = "Price_mills"
        # One shared "order picking starts" time per invoice (public, carries no secret):
        # InvoiceDate + a delay of ~2 hours derived from a hash of the invoice number.
        h = pd.util.hash_pandas_object(table["Invoice"], index=False).values
        u01 = (h % np.uint64(1_000_000)).astype(float) / 1_000_000
        from statistics import NormalDist
        z = np.array([NormalDist().inv_cdf(min(max(x, 1e-6), 1 - 1e-6)) for x in np.unique(u01)])
        zmap = dict(zip(np.unique(u01), z))
        # rounded to whole minutes, like the real InvoiceDate (which never has seconds)
        delay_min = np.round(np.exp(np.log(30) + 0.5 * np.vectorize(zmap.get)(u01))).astype(np.int64)
        table["OrderStart"] = table["InvoiceDate"] + delay_min * 60
        # item-level warehouse events in whole minutes (batch-wave fulfilment): every line of an order is picked, packed and
        # scanned separately, so their times naturally differ within one invoice
        decoys = [OffsetDecoy(B, rng.lognormal(np.log(3 * 3600), 0.8, 200000), step=60),   # PickedAt  = OrderStart + ~3 h (picking waves)
                  OffsetDecoy(B, rng.lognormal(np.log(4 * 3600), 0.7, 200000), step=60),   # PackedAt  = PickedAt + ~4 h (packing queue)
                  OffsetDecoy(B, rng.lognormal(np.log(3 * 3600), 0.8, 200000), step=60)]   # ScannedAt = PackedAt + ~3 h (dispatch scan)
        fam = StorageOnly("price", HistogramDTE(table[secret].values, M), decoys, ["OrderStart", "prev", "prev"])
        names = ("PickedAt", "PackedAt", "ScannedAt")
        offsets = [("PickedAt - InvoiceDate (minutes)", "InvoiceDate", 0, 60),
                   ("PackedAt - PickedAt (minutes)", None, 1, 60),
                   ("ScannedAt - PackedAt (minutes)", None, 2, 60)]
    else:
        if not os.path.exists("taxi_small.csv.gz"):
            shrink_train_csv()
        d = pd.read_csv("taxi_small.csv.gz")
        d = d[(d.N_POINTS > 1) & d.END_LAT.notna()].reset_index(drop=True)
        table = pd.DataFrame({"TIMESTAMP": d.TIMESTAMP.astype(np.int64),
                              "START_LAT_u": np.round(d.START_LAT * 1e6).astype(np.int64),
                              "START_LON_u": np.round(d.START_LON * 1e6).astype(np.int64),
                              "END_LAT_u": np.round(d.END_LAT * 1e6).astype(np.int64)})
        secret = "END_LAT_u"
        decoys = [OffsetDecoy(B, rng.normal(0, 9000, 200000)),            # Waypoint_LAT = START_LAT + ~1 km noise
                  OffsetDecoy(B, rng.normal(0, 12000, 200000)),           # Waypoint_LON = START_LON + ~1 km noise
                  OffsetDecoy(B, -rng.lognormal(np.log(240), 0.8, 200000))]  # DispatchTime = TIMESTAMP - ~4 min
        fam = StorageOnly("end_lat", HistogramDTE(table[secret].values, M), decoys,
                          ["START_LAT_u", "START_LON_u", "TIMESTAMP"])
        names = ("Waypoint_LAT", "Waypoint_LON", "DispatchTime")
        offsets = [("Waypoint_LAT - START_LAT (metres)", "START_LAT_u", 0, 1 / 0.111),
                   ("Waypoint_LON - START_LON (metres)", "START_LON_u", 1, 1 / 0.084),
                   ("TIMESTAMP - DispatchTime (minutes)", "TIMESTAMP", 2, -60)]
    return table, secret, fam, names, offsets


# =============================================================== 6. run
def main(name):
    print(f"Loading {name} ...")
    table, secret, fam, names, offsets = load(name)
    n = len(table)
    ids = np.arange(n)
    values = table[secret].values
    print(f"  rows: {n:,} | distinct secret values: {len(np.unique(values)):,} | "
          f"{PIECES} pieces x {B} values, code space {M:,}")

    seed = new_seed()
    t = time.perf_counter()
    cols = fam.split(seed, ids, values, table)
    t_split = time.perf_counter() - t
    t = time.perf_counter()
    back = fam.recover(seed, ids, cols, table)
    t_rec = time.perf_counter() - t

    print(f"\n[1] Exact rebuild with the right seed: {np.array_equal(back, values)}  "
          f"({np.mean(back == values):.4%} of rows)")
    print(f"[2] Time: split {1e6 * t_split / n:.3f} us/row, rebuild {1e6 * t_rec / n:.3f} us/row")

    print("\n[3] Wrong-seed rebuilds (honey check):")
    top = pd.Series(values).value_counts(normalize=True).head(5)
    shares = []
    for i in range(3):
        wrong = fam.recover(new_seed(), ids, cols, table)
        shares.append(pd.Series(wrong).value_counts(normalize=True))
        print(f"    wrong seed {i + 1}: same value as truth in {np.mean(wrong == values):.4%} of rows")
    comp = pd.DataFrame({"value": top.index, "real share": top.values,
                         **{f"wrong seed {i + 1}": [sh.get(v, 0.0) for v in top.index] for i, sh in enumerate(shares)}})
    print("    Most common real values and how often they appear under wrong seeds:")
    print(comp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n[4] Do the disguised columns look natural? Offset from their base column:")
    print(f"    {'offset':38s} {'5%':>9s} {'25%':>9s} {'median':>9s} {'75%':>9s} {'95%':>9s}")
    for label, base, j, unit in offsets:
        b = table[base].values if base else cols[j - 1]
        q = np.percentile((np.asarray(cols[j], float) - b) / unit, [5, 25, 50, 75, 95])
        print(f"    {label:38s} " + " ".join(f"{v:9.1f}" for v in q))

    if name == "retail":
        inv = table["Invoice"].values
        same = inv[1:] == inv[:-1]
        gap = np.abs(np.diff(np.asarray(cols[0], float)))[same] / 60
        print(f"\n[4b] Lines of the same order: median gap between their PickedAt = {np.median(gap):.0f} min "
              f"(90% within {np.percentile(gap, 90):.0f} min)")

    out = table.drop(columns=[secret] + (["OrderStart"] if "OrderStart" in table else [])).copy()
    for nm, c in zip(names, cols):
        out[nm] = np.asarray(c).astype(np.int64)
    if name == "retail":
        for c in ["InvoiceDate", *names]:
            out[c] = pd.to_datetime(out[c], unit="s")
    else:
        for c in ["TIMESTAMP", "DispatchTime"]:
            out[c] = pd.to_datetime(out[c], unit="s")
    out.head(1000).to_csv(f"protected_{name}_sample.csv", index=False)
    print(f"\n[5] Saved the first 1,000 protected rows to protected_{name}_sample.csv")
    print("    (the secret column is gone; only the disguised columns remain)")


if __name__ == "__main__":
    for ds in sys.argv[1:] or ["retail"]:
        main(ds)
        print("\n" + "=" * 70 + "\n")
