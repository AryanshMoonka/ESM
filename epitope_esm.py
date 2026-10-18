import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import logging
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel

from idk_some_shit import fetch_experimental_pdb
from linear_epitope import load, rsasa, auc
from epitope_model import FEATURES, CACHE
from epitope_benchmark import CSV_OUT

logging.disable(logging.WARNING)

MODEL = "facebook/esm2_t12_35M_UR50D"
EMB_CACHE = "epitope_esm_emb.npz"


def embed(data):
    res = pd.read_csv(CSV_OUT)
    res = res[res.status == "ok"]
    ag_of = dict(zip(res.pdb, res.antigen))
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModel.from_pretrained(MODEL).eval()
    emb = np.zeros((len(data), 480), dtype=np.float32)
    keep = np.ones(len(data), dtype=bool)
    for pdb in data.pdb.unique():
        idx = np.where(data.pdb.values == pdb)[0]
        try:
            tab = rsasa(load(fetch_experimental_pdb(pdb)), ag_of[pdb])
            seq = "".join(t[1] for t in tab)
            if len(seq) != len(idx) or len(seq) > 1020:
                raise ValueError(f"length mismatch or too long ({len(seq)} vs {len(idx)})")
            with torch.no_grad():
                h = mdl(**tok(seq, return_tensors="pt")).last_hidden_state[0, 1:-1]
            emb[idx] = h.numpy()
            print(f"{pdb}: embedded {len(seq)} residues")
        except Exception as e:
            print(f"{pdb}: skipped ({str(e)[:60]})")
            keep[idx] = False
    np.savez(EMB_CACHE, emb=emb, keep=keep)
    return emb, keep


def cv_predict(X, y, groups, C, k=5):
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))
        m.fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1]
    return pred


def main(seed=0):
    data = pd.read_csv(CACHE)
    if os.path.exists(EMB_CACHE):
        z = np.load(EMB_CACHE)
        emb, keep = z["emb"], z["keep"]
    else:
        emb, keep = embed(data)
    data, emb = data[keep].reset_index(drop=True), emb[keep]

    y, groups = data.y.values, data.pdb.values
    X_geo = data[FEATURES].values
    X_esm = np.hstack([X_geo, emb])
    data["p_geo"] = cv_predict(X_geo, y, groups, C=1.0)
    data["p_esm"] = cv_predict(X_esm, y, groups, C=0.01)

    rows = []
    for pdb, g in data.groupby("pdb"):
        if 0 < g.y.sum() < len(g):
            rows.append((pdb, auc(g.rsasa, g.y), auc(g.p_geo, g.y), auc(g.p_esm, g.y)))
    r = pd.DataFrame(rows, columns=["pdb", "rsasa", "geo", "esm"])
    d = (r.esm - r.rsasa).values
    rng = np.random.default_rng(seed)
    lo, hi = np.percentile([rng.choice(d, len(d)).mean() for _ in range(5000)], [2.5, 97.5])

    print(f"\nComplexes: {len(r)} (grouped 5-fold CV, folds split by complex)")
    print(f"Mean AUC  rSASA: {r.rsasa.mean():.3f}   geometric model: {r.geo.mean():.3f}   geometric + ESM-2: {r.esm.mean():.3f}")
    print(f"ESM-2 model minus rSASA: {d.mean():+.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"ESM-2 model beats rSASA in {(d > 0).sum()}/{len(d)} complexes")
    print(f"Wilcoxon paired p = {wilcoxon(d, method='approx').pvalue:.4f}")


if __name__ == "__main__":
    main()