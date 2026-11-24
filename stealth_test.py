"""RQ1 - Stealth test: can an attacker who steals the whole table find the hidden columns?

For each dataset, the sensitive attribute (retail: price, taxi: end latitude) is
protected by each scheme. The attacker gets the stored table with ALL the other
real columns (names removed), ranks every column by how suspicious it looks, and
flags the top k, where k = number of protected columns.

Schemes compared
  SKAS            our Family 1 (3 disguised columns) - imported from family1_storage_only.py
  Rahmani 2023a   4 share columns + 1 order-preserving index (cover = quantity / number of GPS points)
  Rahmani 2022    2 share columns
  SRSOSE          6 share columns
  AES             1 ciphertext column

Detectors (the attacker's tools)
  entropy scanner     flags random-looking or text-blob columns
  independence test   flags columns unrelated to every other column
  linear-link test    flags columns almost perfectly tied to another column
  ML classifier       random forest trained on the OTHER dataset's tables, where the
                      attacker knows which columns were protected (all schemes)

Metrics, averaged over REPS random samples (mean +- std)
  precision@k   share of flagged columns that are really protected (chance = k / all columns)
  AUC           0.5 = attacker ranks columns no better than random, 1.0 = perfect detection

Run (same folder as family1_storage_only.py and the data files):
    pip install -r requirements.txt
    python stealth_test.py                       # default: seeds 0-4, 10 samples each
    python stealth_test.py --seeds 0 1 2 3 4 5 6 7 8 9   # more seeds, more precise
    python stealth_test.py --seeds 0 --reps 5    # quick check

Each seed is one full experiment: new secret keys AND a newly trained ML attacker.
The ML attacker's result depends strongly on its training data, so results are
averaged over several seeds. Same --seeds and --reps = identical table on every machine.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from fractions import Fraction

import numpy as np
import pandas as pd
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

N_ROWS = 5000   # rows per random sample
REPS = 10       # random samples per dataset per seed (override with --reps)
RNG = np.random.default_rng(0)  # all randomness of the test; reset by --seed
HERE = os.path.dirname(os.path.abspath(__file__))

spec = importlib.util.spec_from_file_location("f1", os.path.join(HERE, "family1_storage_only.py"))
f1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f1)


def use_seed(seed: int):
    """Make the whole test reproducible: secret seeds, encoder randomness and AES
    nonces all come from one seeded generator instead of os.urandom."""
    global RNG
    RNG = np.random.default_rng(seed)
    f1.new_seed = lambda: RNG.bytes(32)
    f1.fresh_random = lambda n, modulus: RNG.integers(0, modulus, n, dtype=np.int64)


# =============================================================== data
def full_tables(name):
    """Real columns of the dataset (same cleaning and row order as family1_storage_only.py)."""
    if name == "retail":
        cache = os.path.join(HERE, "retail_full.pkl")
        if os.path.exists(cache):
            d = pd.read_pickle(cache)
        else:
            d = pd.concat(pd.read_excel(os.path.join(HERE, "online_retail_II.xlsx"), sheet_name=None).values(),
                          ignore_index=True)
            d.to_pickle(cache)
        d = d[d["Customer ID"].notna() & (d.Quantity > 0) & (d.Price > 0)].reset_index(drop=True)
        plain = pd.DataFrame({
            "LineID": np.arange(len(d), dtype=np.int64),   # row id: stored anyway, SKAS needs it to rebuild
            "Invoice": pd.to_numeric(d.Invoice.astype(str).str.replace("C", ""), errors="coerce").fillna(0).astype(np.int64),
            "StockCode": d.StockCode.astype(str).astype("category").cat.codes.astype(np.int64),
            "Description": d.Description.astype(str).astype("category").cat.codes.astype(np.int64),
            "Quantity": d.Quantity.astype(np.int64),
            "InvoiceDate": d.InvoiceDate.astype("datetime64[s]").astype(np.int64),
            "CustomerID": d["Customer ID"].astype(np.int64),
            "Country": d.Country.astype("category").cat.codes.astype(np.int64),
        })
        cover = "Quantity"
    else:
        d = pd.read_csv(os.path.join(HERE, "taxi_small.csv.gz"))
        d = d[(d.N_POINTS > 1) & d.END_LAT.notna()].reset_index(drop=True)
        plain = pd.DataFrame({
            "TRIP_ID": d.TRIP_ID.astype(np.int64),
            "CALL_TYPE": d.CALL_TYPE.astype("category").cat.codes.astype(np.int64),
            "ORIGIN_CALL": d.ORIGIN_CALL.fillna(-1).astype(np.int64),
            "ORIGIN_STAND": d.ORIGIN_STAND.fillna(-1).astype(np.int64),
            "TAXI_ID": d.TAXI_ID.astype(np.int64),
            "TIMESTAMP": d.TIMESTAMP.astype(np.int64),
            "DAY_TYPE": d.DAY_TYPE.astype("category").cat.codes.astype(np.int64),
            "N_POINTS": d.N_POINTS.astype(np.int64),
            "START_LAT": np.round(d.START_LAT * 1e6).astype(np.int64),
            "START_LON": np.round(d.START_LON * 1e6).astype(np.int64),
            "END_LON": np.round(d.END_LON * 1e6).astype(np.int64),
        })
        cover = "N_POINTS"
    return plain, cover


# =============================================================== baselines
def _rng(seed):
    return np.random.default_rng(int.from_bytes(seed[:8], "little"))


def rahmani2023a(seed, secret, cover, domain):
    """4 shares of q(x) = cover + secret*x + a2 x^2 + a3 x^3 at a fixed secret vector, + OPE index."""
    rng = _rng(seed)
    X = [int(x) for x in rng.choice(np.arange(2, 200), 4, replace=False)]
    a2 = rng.integers(1, 1 << 20, len(secret))
    a3 = rng.integers(1, 1 << 20, len(secret))
    cols = {f"r23_s{j + 1}": cover + secret * x + a2 * x * x + a3 * x ** 3 for j, x in enumerate(X)}
    vals = np.unique(domain)
    codes = np.cumsum(rng.integers(1, 1000, len(vals)))
    cols["r23_idx"] = codes[np.searchsorted(vals, secret)]
    return cols


def rahmani2022(seed, secret, pk, domain, L=50, x=(29, 81), n_part=400, gap=100):
    rng = _rng(seed)
    pi = rng.permutation(L)
    lo, hi = int(np.min(domain)), int(np.max(domain))
    edges = np.linspace(lo, hi + 1, n_part + 1)
    offsets = np.cumsum(rng.integers(1, gap + 1, n_part)) * (hi - lo + 1)
    part = np.clip(np.searchsorted(edges, secret, side="right") - 1, 0, n_part - 1)
    w = (secret + offsets[part]) * L + pi[np.asarray(pk) % L]
    a = rng.integers(1, 1 << 30, len(secret))
    return {"r22_s1": w + a * x[0], "r22_s2": w + a * x[1]}


def srsose(seed, secret, domain, H=1 << 25, t=250, P=(1 << 61) - 1):
    rng = _rng(seed)
    vals = np.unique(domain)
    idx = np.searchsorted(vals, secret)
    per = int(np.ceil(len(vals) / t))
    sec_size = H // t
    width = max(sec_size // per, 1)
    z = rng.permutation(t)[idx // per] * sec_size + (idx % per) * width + rng.integers(0, width, len(secret))
    rho = (secret.astype(object) - z.astype(object)) % P

    def share(s, xs):
        a1 = rng.integers(0, 1 << 60, len(secret)).astype(object)
        a2 = rng.integers(0, 1 << 60, len(secret)).astype(object)
        return [(s + a1 * x + a2 * x * x) % P for x in xs]
    xs = [int(v) for v in rng.integers(2, 1 << 40, 3)]
    xr = [int(v) for v in rng.integers(2, 1 << 40, 3)]
    cols = {f"srs_seed{j + 1}": c for j, c in enumerate(share(z.astype(object), xs))}
    cols.update({f"srs_res{j + 1}": c for j, c in enumerate(share(rho, xr))})
    return cols


def aes_column(seed, secret):
    out = []
    for v in np.asarray(secret, dtype=np.int64):
        nonce = RNG.bytes(16)
        enc = Cipher(algorithms.AES(seed[:16]), modes.CTR(nonce)).encryptor()
        out.append((nonce + enc.update(int(v).to_bytes(8, "little", signed=True))).hex())
    return {"aes": np.array(out)}


# =============================================================== column features
def numeric_view(col):
    col = np.asarray(col)
    if col.dtype.kind in "OUS":
        try:
            return np.array([float(int(v)) for v in col])
        except (ValueError, TypeError):
            return np.array([float(int(str(v)[:12], 16)) for v in col])
    return col.astype(np.float64)


def is_text_blob(col):
    col = np.asarray(col)
    return col.dtype.kind in "OUS" and isinstance(col[0], str) and not str(col[0]).lstrip("-").isdigit()


FEATURES = ["uniformity", "last_digit_entropy", "uniqueness", "max_corr", "max_nmi", "text_blob", "log_magnitude"]


def features(table):
    names = list(table)
    X = {n: numeric_view(table[n]) for n in names}
    ranks = {n: stats.rankdata(X[n]) for n in names}
    bins = {n: np.minimum((ranks[n] - 1) * 32 // len(ranks[n]), 31).astype(int) for n in names}
    F = {}
    for n in names:
        x = X[n]
        u = (x - x.min()) / (x.max() - x.min() + 1e-12)
        digits = np.array([int(str(int(abs(v)))[-1]) for v in x])
        corr, nmi = 0.0, 0.0
        for m in names:
            if m == n:
                continue
            c = abs(stats.spearmanr(ranks[n], ranks[m]).statistic)
            corr = max(corr, 0.0 if np.isnan(c) else c)
            joint = np.histogram2d(bins[n], bins[m], bins=32)[0] / len(x)
            px, py = joint.sum(1), joint.sum(0)
            nz = joint > 0
            mi = (joint[nz] * np.log(joint[nz] / (px[:, None] * py[None, :])[nz])).sum()
            h = -(px[px > 0] * np.log(px[px > 0])).sum()
            nmi = max(nmi, mi / h if h > 0 else 0.0)
        F[n] = {"uniformity": 1 - stats.kstest(u, "uniform").statistic,
                "last_digit_entropy": stats.entropy(np.bincount(digits, minlength=10) + 1e-9) / np.log(10),
                "uniqueness": len(np.unique(x)) / len(x),
                "max_corr": corr, "max_nmi": nmi, "text_blob": float(is_text_blob(table[n])),
                "log_magnitude": float(np.log10(np.median(np.abs(x)) + 1))}
    return F


DETECTORS = {
    "entropy scanner": lambda f: f["text_blob"] * 2 + f["uniformity"] + f["last_digit_entropy"],
    "independence test": lambda f: 1 - f["max_nmi"],
    "linear-link test": lambda f: f["max_corr"] if f["max_corr"] > 0.999 else f["max_corr"] - 1,
}


def precision_at_k(scores, protected):
    k = len(protected)
    top = sorted(scores, key=lambda n: -scores[n])[:k]
    return sum(n in protected for n in top) / k


def auc(scores, protected):
    names = list(scores)
    return float(roc_auc_score([int(n in protected) for n in names], [scores[n] for n in names]))


# =============================================================== experiment
def build(name):
    """All (dataset, rep, scheme) tables with their features."""
    table, secret_col, fam, names, _ = f1.load(name)
    plain, cover = full_tables(name)
    assert len(plain) == len(table), "row mismatch with family1_storage_only.py"
    secret_all = table[secret_col].values
    out = []
    for rep in range(REPS):
        rows = np.sort(np.random.default_rng(1000 + rep).choice(len(table), N_ROWS, replace=False))
        seed = f1.new_seed()
        s = secret_all[rows]
        p = plain.iloc[rows].reset_index(drop=True)
        cols = fam.split(seed, rows, s, table.iloc[rows].reset_index(drop=True))
        schemes = {
            "SKAS (3 columns)": dict(zip(names, cols)),
            "Rahmani 2023a": rahmani2023a(seed, s, p[cover].values, secret_all),
            "Rahmani 2022": rahmani2022(seed, s, rows, secret_all),
            "SRSOSE": srsose(seed, s, secret_all),
            "AES": aes_column(seed, s),
        }
        for scheme, prot in schemes.items():
            pl = p.drop(columns=[cover]) if scheme == "Rahmani 2023a" else p  # cover is embedded in its shares
            tab = {f"plain_{c}": pl[c].values for c in pl.columns}
            tab.update({f"prot_{k}": np.asarray(v) for k, v in prot.items()})
            out.append({"dataset": name, "rep": rep, "scheme": scheme, "features": features(tab),
                        "protected": {f"prot_{k}" for k in prot}, "n_cols": len(tab)})
        print(f"  {name}: sample {rep + 1}/{REPS} done")
    return out


def main():
    global REPS
    ap = argparse.ArgumentParser(description="SKAS stealth test")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                    help="random seeds; each is one full experiment (same seeds = same results)")
    ap.add_argument("--reps", type=int, default=REPS, help="random samples per dataset per seed")
    args = ap.parse_args()
    REPS = args.reps
    print(f"Stealth test: seeds {args.seeds}, {REPS} samples of {N_ROWS:,} rows per dataset per seed")
    rows = []
    for seed in args.seeds:
        print(f"\n--- seed {seed} ---")
        use_seed(seed)
        runs = []
        for name in ["retail", "taxi"]:
            print(f"Building tables for {name} ...")
            runs += build(name)
        rows += evaluate(runs, seed)

    report(pd.DataFrame(rows), args.seeds)


def evaluate(runs, seed):
    rows = []
    for other, this in [("taxi", "retail"), ("retail", "taxi")]:
        Xtr, ytr = [], []
        for r in runs:
            if r["dataset"] == other:
                for n, f in r["features"].items():
                    Xtr.append([f[k] for k in FEATURES])
                    ytr.append(int(n in r["protected"]))
        clf = RandomForestClassifier(n_estimators=300, random_state=0).fit(Xtr, ytr)
        for r in runs:
            if r["dataset"] != this:
                continue
            F, prot = r["features"], r["protected"]
            row = {"dataset": this, "scheme": r["scheme"], "seed": seed, "rep": r["rep"],
                   "chance": len(prot) / r["n_cols"]}
            for det, fn in DETECTORS.items():
                sc = {n: fn(F[n]) for n in F}
                row[f"{det} P@k"] = precision_at_k(sc, prot)
                row[f"{det} AUC"] = auc(sc, prot)
            p = clf.predict_proba([[F[n][k] for k in FEATURES] for n in F])[:, 1]
            sc = dict(zip(F, p))
            row["ML classifier P@k"] = precision_at_k(sc, prot)
            row["ML classifier AUC"] = auc(sc, prot)
            rows.append(row)
    return rows


def report(df, seeds):
    df.to_csv(os.path.join(HERE, "stealth_results_all_samples.csv"), index=False)
    metrics = [c for c in df.columns if c.endswith("P@k") or c.endswith("AUC")]
    for ds in ["retail", "taxi"]:
        sub = df[df.dataset == ds]
        print(f"\n=== {ds.upper()} ({len(seeds)} seeds x {REPS} samples of {N_ROWS:,} rows; mean +- std over all samples) ===")
        for kind in ["P@k", "AUC"]:
            cols = [c for c in metrics if c.endswith(kind)]
            print(f"\n  {('precision@k  (chance shown)' if kind == 'P@k' else 'AUC  (0.5 = random guessing)')}")
            header = f"  {'scheme':18s}" + ("  chance" if kind == "P@k" else "") + "".join(f" {c.replace(' ' + kind, ''):>20s}" for c in cols)
            print(header)
            for scheme, g in sub.groupby("scheme", sort=False):
                line = f"  {scheme:18s}"
                if kind == "P@k":
                    line += f"  {g.chance.mean():6.2f}"
                line += "".join(f" {g[c].mean():12.2f} +-{g[c].std():5.2f}" for c in cols)
                print(line)
        print("\n  ML classifier precision@k per seed (each seed = a newly trained attacker)")
        for scheme, g in sub.groupby("scheme", sort=False):
            per = g.groupby("seed")["ML classifier P@k"].mean()
            print(f"  {scheme:18s}  " + "  ".join(f"{v:.2f}" for v in per.values) +
                  f"   | mean {per.mean():.2f}, std across seeds {per.std():.2f}")
    print("\nPer-sample results saved to stealth_results_all_samples.csv")


if __name__ == "__main__":
    main()
