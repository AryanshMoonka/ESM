import logging
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from idk_some_shit import fetch_experimental_pdb
from linear_epitope import load, rsasa, true_epitope, auc
from epitope_benchmark import pick_antigen, CSV_IN, CSV_OUT

logging.disable(logging.WARNING)

KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
      "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
      "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}
CHARGE = {"D": -1, "E": -1, "K": 1, "R": 1, "H": 0.1}
FEATURES = ["rsasa", "patch_rsasa", "hydro", "charge", "n_neigh"]
CACHE = "epitope_features.csv"


def build():
    sab = pd.read_csv(CSV_IN, low_memory=False).dropna(subset=["Hchain", "Lchain"])
    sab["Hchain"], sab["Lchain"] = sab.Hchain.astype(str), sab.Lchain.astype(str)
    sab["pdb4"] = sab["PDB"].str[-4:].str.upper()
    res = pd.read_csv(CSV_OUT)
    res = res[res.status == "ok"].dropna(subset=["auc_rsasa", "auc_patch"])
    frames = []
    for pdb, ag in zip(res.pdb, res.antigen):
        try:
            model = load(fetch_experimental_pdb(pdb))
            rows = sab[sab.pdb4 == pdb]
            skip = set(rows.Hchain) | set(rows.Lchain)
            partners = None
            for _, r in rows.iterrows():
                if pick_antigen(model, r.Hchain, r.Lchain, skip) == ag:
                    partners = [r.Hchain, r.Lchain]
                    break
            if partners is None:
                print(f"{pdb}: antigen/partners not recovered")
                continue
            tab = rsasa(model, ag)
            ep = true_epitope(model, ag, partners)
            aas = [t[1] for t in tab]
            base = np.array([t[2] for t in tab])
            xyz = np.array([t[3] for t in tab])
            patch, neigh = [], []
            for i in range(len(tab)):
                near = np.linalg.norm(xyz - xyz[i], axis=1) < 10.0
                patch.append(base[near].mean())
                neigh.append(near.sum())
            frames.append(pd.DataFrame({
                "pdb": pdb,
                "rsasa": base,
                "patch_rsasa": patch,
                "hydro": [KD[a] for a in aas],
                "charge": [CHARGE.get(a, 0.0) for a in aas],
                "n_neigh": neigh,
                "y": [int(t[0] in ep) for t in tab],
            }))
            print(f"{pdb}: {len(tab)} residues")
        except Exception as e:
            print(f"{pdb}: skipped ({str(e)[:60]})")
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(CACHE, index=False)
    return data


def main(k=5, seed=0):
    data = pd.read_csv(CACHE) if os.path.exists(CACHE) else build()
    X, y, groups = data[FEATURES].values, data.y.values, data.pdb.values
    data["pred"] = np.nan
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=1000, class_weight="balanced"))
        m.fit(X[tr], y[tr])
        data.loc[te, "pred"] = m.predict_proba(X[te])[:, 1]

    rows = []
    for pdb, g in data.groupby("pdb"):
        if 0 < g.y.sum() < len(g):
            rows.append((pdb, auc(g.pred, g.y), auc(g.rsasa, g.y)))
    r = pd.DataFrame(rows, columns=["pdb", "auc_model", "auc_rsasa"])
    d = (r.auc_model - r.auc_rsasa).values

    rng = np.random.default_rng(seed)
    boots = [rng.choice(d, len(d)).mean() for _ in range(5000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])

    print(f"\nComplexes: {len(r)} (grouped {k}-fold CV, folds split by complex)")
    print(f"Mean AUC  model: {r.auc_model.mean():.3f}   rSASA alone: {r.auc_rsasa.mean():.3f}")
    print(f"Mean difference (model - rSASA): {d.mean():.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"Model beats rSASA in {(d > 0).sum()}/{len(d)} complexes")
    print(f"Wilcoxon paired p = {wilcoxon(d, method='approx').pvalue:.4f}")

    m = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=1000, class_weight="balanced"))
    m.fit(X, y)
    print("\nStandardized coefficients (full-data fit):")
    for f, c in zip(FEATURES, m[-1].coef_[0]):
        print(f"  {f:12s} {c:+.3f}")


if __name__ == "__main__":
    main()