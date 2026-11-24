"""SKAS - Family 1 (storage-only), standalone version.

What it does
  1. Loads a dataset and picks one sensitive column (retail: Price, taxi: END_LAT).
  2. Hides that column in three ordinary-looking columns, made from a secret
     seed, the row's permanent key (RowKey) and its version number (retail: PickedAt / PackedAt / ScannedAt
     warehouse times for each order line; taxi: a waypoint latitude and longitude near the pickup point and a
     dispatch time shortly before the trip starts).
  3. Rebuilds the column with the right seed and checks it is 100% exact.
  4. Rebuilds it with wrong seeds and checks the result looks realistic (honey).
  5. Prints timings, checks that the disguised columns have natural offsets,
     and saves the first rows of the protected table to CSV.
  6. Updates: a changed value is re-protected with version+1, so the old and
     new rows use different masks (an attacker cannot subtract them).
  7. Integrity: a per-row tag under a SEPARATE key detects any edited cell
     (it never touches the honey seed, so it cannot be used to test seed guesses).
  8. Stable keys: rows are rebuilt by RowKey, so reordering the table is harmless.

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


def row_masks(key: bytes, row_ids, modulus: int, versions=None) -> np.ndarray:
    """One secret number per row: AES(key, version || row key) reduced to 0..modulus-1.
    A new version gives a completely new mask, so updated rows never reuse a mask."""
    ids = np.asarray(row_ids, dtype=np.uint64)
    blocks = np.zeros((len(ids), 2), dtype=np.uint64)
    if versions is not None:
        blocks[:, 0] = np.asarray(versions, dtype=np.uint64)
    blocks[:, 1] = ids
    enc = Cipher(algorithms.AES(key), modes.ECB()).encryptor()
    out = np.frombuffer(enc.update(blocks.tobytes()) + enc.finalize(), dtype=np.uint64)
    return (out.reshape(-1, 2)[:, 0] % np.uint64(modulus)).astype(np.int64)


def as_u64(x) -> np.ndarray:
    return np.asarray(np.round(np.asarray(x, dtype=np.float64)) if np.asarray(x).dtype.kind == "f" else x,
                      dtype=np.int64).view(np.uint64)


def integrity_tags(mac_key: bytes, fields) -> np.ndarray:
    """Per-row integrity tag: AES-CBC-MAC (fixed length) over the row key, version,
    the disguised columns and their base columns. Uses its own random key, not the seed."""
    cols = [as_u64(f) for f in fields]
    if len(cols) % 2:
        cols.append(np.zeros(len(cols[0]), dtype=np.uint64))
    enc = Cipher(algorithms.AES(mac_key), modes.ECB()).encryptor()
    state = np.zeros((len(cols[0]), 2), dtype=np.uint64)
    for a in range(0, len(cols), 2):
        state ^= np.stack([cols[a], cols[a + 1]], axis=1)
        state = np.frombuffer(enc.update(state.tobytes()), dtype=np.uint64).reshape(-1, 2).copy()
    return state[:, 0]


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

    def split(self, seed, row_ids, values, table, versions=None):
        k = row_masks(derive(seed, self.name + "|mask"), row_ids, M, versions)  # secret mask per row
        u = self.dte.encode(values, fresh_random(len(values), 1 << 30))  # value -> code
        w = (u + k) % M                                                  # masked code
        cols = []
        for j in range(PIECES):
            piece = (w >> (BITS * j)) & (B - 1)                          # j-th piece
            cols.append(self.decoys[j].encode(piece, self._base(j, table, cols)))
        return cols

    def recover(self, seed, row_ids, cols, table, versions=None):
        w = np.zeros(len(row_ids), dtype=np.int64)
        for j in range(PIECES):
            piece = self.decoys[j].decode(cols[j], self._base(j, table, cols))
            w |= piece.astype(np.int64) << (BITS * j)
        k = row_masks(derive(seed, self.name + "|mask"), row_ids, M, versions)
        return self.dte.decode((w - k) % M)

    def tags(self, mac_key, row_ids, versions, cols, table):
        return integrity_tags(mac_key, [row_ids, versions, *cols, *[table[c].values for c in self.mac_bases]])


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
        table["InvoiceHash"] = h.view(np.int64)
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
        fam.mac_bases = ["InvoiceHash", "InvoiceDate"]
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
        fam.mac_bases = ["START_LAT_u", "START_LON_u", "TIMESTAMP"]
        names = ("Waypoint_LAT", "Waypoint_LON", "DispatchTime")
        offsets = [("Waypoint_LAT - START_LAT (metres)", "START_LAT_u", 0, 1 / 0.111),
                   ("Waypoint_LON - START_LON (metres)", "START_LON_u", 1, 1 / 0.084),
                   ("TIMESTAMP - DispatchTime (minutes)", "TIMESTAMP", 2, -60)]
    # permanent primary key, assigned once when the table is protected and never reused
    # (taxi TRIP_ID is not safe for this: 11 TRIP_IDs appear twice)
    table.insert(0, "RowKey", np.arange(len(table), dtype=np.int64))
    return table, secret, fam, names, offsets


# =============================================================== 6. run
def main(name):
    print(f"Loading {name} ...")
    table, secret, fam, names, offsets = load(name)
    n = len(table)
    ids = table["RowKey"].values
    ver = np.zeros(n, dtype=np.int64)
    values = table[secret].values
    print(f"  rows: {n:,} | distinct secret values: {len(np.unique(values)):,} | "
          f"{PIECES} pieces x {B} values, code space {M:,}")

    seed = new_seed()
    t = time.perf_counter()
    cols = fam.split(seed, ids, values, table, ver)
    t_split = time.perf_counter() - t
    t = time.perf_counter()
    back = fam.recover(seed, ids, cols, table, ver)
    t_rec = time.perf_counter() - t

    print(f"\n[1] Exact rebuild with the right seed: {np.array_equal(back, values)}  "
          f"({np.mean(back == values):.4%} of rows)")
    print(f"[2] Time: split {1e6 * t_split / n:.3f} us/row, rebuild {1e6 * t_rec / n:.3f} us/row")

    print("\n[3] Wrong-seed rebuilds (honey check):")
    top = pd.Series(values).value_counts(normalize=True).head(5)
    shares = []
    for i in range(3):
        wrong = fam.recover(new_seed(), ids, cols, table, ver)
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

    rng = np.random.default_rng(1)

    # ---------------------------------------------------------- [6] updates
    print("\n[6] Updates: 20,000 rows get a new value and are re-protected")
    upd = rng.choice(n, 20_000, replace=False)
    new_vals = values.copy()
    new_vals[upd] = rng.choice(values, len(upd))
    sub = table.iloc[upd].reset_index(drop=True)

    def w_of(c, t):  # what an attacker can compute: the masked code, from the public disguise
        w = np.zeros(len(t), dtype=np.int64)
        for j in range(PIECES):
            base = c[j - 1] if fam.bases[j] == "prev" else t[fam.bases[j]].values
            w |= fam.decoys[j].decode(c[j], base).astype(np.int64) << (BITS * j)
        return w
    old_cols = [np.asarray(c)[upd] for c in cols]
    w_old = w_of(old_cols, sub)
    for label, v_new in [("same version (old design)", ver[upd]), ("version + 1 (new design)", ver[upd] + 1)]:
        new_cols = fam.split(seed, ids[upd], new_vals[upd], sub, v_new)
        ok = np.array_equal(fam.recover(seed, ids[upd], new_cols, sub, v_new), new_vals[upd])
        w_new = w_of(new_cols, sub)
        # attacker who knows the OLD values guesses the NEW ones from w_new - w_old
        u_old_guess = fam.dte.starts[np.searchsorted(fam.dte.vals, values[upd])] + \
            fam.dte.slots[np.searchsorted(fam.dte.vals, values[upd])] // 2
        guess = fam.dte.decode((u_old_guess + w_new - w_old) % M)
        print(f"    {label:27s}: rebuild exact {ok} | attacker who knew the old values "
              f"guesses the new value right in {np.mean(guess == new_vals[upd]):.1%} of rows")
    print("    (chance level = how often a random honey value is right, about the wrong-seed match rate above)")

    # ---------------------------------------------------------- [7] integrity
    mac_key = os.urandom(16)  # separate random key, NOT derived from the seed
    t = time.perf_counter()
    tags = fam.tags(mac_key, ids, ver, cols, table)
    t_tag = time.perf_counter() - t
    print(f"\n[7] Integrity tags (separate key, {1e6 * t_tag / n:.3f} us/row):")
    print(f"    untouched table: {np.mean(fam.tags(mac_key, ids, ver, cols, table) == tags):.4%} of rows pass")
    bad = rng.choice(n, 1000, replace=False)
    for label, edit in [("PickedAt-type column +1 step", "col"), ("base column edited", "base")]:
        c2 = [np.asarray(c).copy() for c in cols]
        t2 = table.copy()
        if edit == "col":
            c2[1][bad] = c2[1][bad] + (60 if name == "retail" else 1)
        else:
            bcol = fam.mac_bases[-1]
            t2.loc[bad, bcol] = t2.loc[bad, bcol] + 60
        changed = fam.recover(seed, ids[bad], [c[bad] for c in c2], t2.iloc[bad].reset_index(drop=True), ver[bad])
        caught = fam.tags(mac_key, ids, ver, c2, t2)[bad] != tags[bad]
        print(f"    1,000 rows tampered ({label:28s}): detected {caught.mean():.1%} | "
              f"without the tag, {np.mean(changed != values[bad]):.1%} would silently decode to a different value")
    print("    The tag uses only stored columns and its own key, so checking it reveals nothing about the seed.")

    # ---------------------------------------------------------- [8] stable keys
    perm = rng.permutation(n)
    tp = table.iloc[perm].reset_index(drop=True)
    cp = [np.asarray(c)[perm] for c in cols]
    back_p = fam.recover(seed, tp["RowKey"].values, cp, tp, ver[perm])
    print(f"\n[8] Table rows shuffled (as a database may do): rebuild by RowKey exact = "
          f"{np.array_equal(back_p, values[perm])}")

    out = table.drop(columns=[secret] + [c for c in ["OrderStart", "InvoiceHash"] if c in table]).copy()
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
