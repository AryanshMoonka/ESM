import os
from xml.parsers.expat import model
import numpy as np
import pandas as pd
import random
from difflib import SequenceMatcher
from scipy.stats import wilcoxon
from idk_some_shit import fetch_experimental_pdb, _AA3
from linear_epitope import load, true_epitope, evaluate
import logging, traceback
logging.disable(logging.WARNING)
import traceback

CSV_IN, CSV_OUT = "sabdab_summary_all.csv", "epitope_results.csv"

def header_info(txt):
    method, res = None, None
    for line in txt.splitlines():
        if line.startswith("EXPDTA"):
            method = line[10:].strip()
        elif line.startswith("REMARK   2 RESOLUTION."):
            try:
                res = float(line.split()[3])
            except Exception:
                pass
        elif line.startswith("ATOM"):
            break
    return method, res

def chain_seq(ch):
    return "".join(_AA3.get(r.get_resname(), "X") for r in ch if r.id[0] == " ")

def pick_antigen(model, h, l, skip):
    if h not in model or l not in model:
        return None
    best = None
    for ch in model:
        if ch.id in skip:
            continue
        n = sum(1 for r in ch if r.id[0] == " ")
        if not 50 <= n <= 700:
            continue
        k = len(true_epitope(model, ch.id, [h, l]))
        if k >= 5 and (best is None or k > best[1]):
            best = (ch.id, k)
    return best[0] if best else None

def summarize(out):
    df = pd.DataFrame(out)
    print("\nStatus counts:\n", df.status.str.slice(0, 20).value_counts())
    ok = df[df.status == "ok"]
    ok = ok.dropna(subset=["auc_rsasa", "auc_patch"])    
    if len(ok) < 3:
        print("Too few complexes to summarize yet")
        return
    rng = np.random.default_rng(0)
    for col in ["auc_rsasa", "auc_patch"]:
        x = ok[col].to_numpy(float)
        boots = [rng.choice(x, len(x)).mean() for _ in range(5000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"{col}: mean {x.mean():.3f}  95% CI [{lo:.3f}, {hi:.3f}]  (n={len(x)})")
    d = ok.auc_patch - ok.auc_rsasa
    print(f"patch beats rSASA in {(d > 0).sum()}/{len(d)} complexes")
    p = wilcoxon(ok.auc_patch.to_numpy(float), ok.auc_rsasa.to_numpy(float), method="approx").pvalue
    print(f"Wilcoxon paired p = {p:.4f}")
def main(target=60, seed=0):
    df = pd.read_csv(CSV_IN, low_memory=False).dropna(subset=["Hchain", "Lchain"])
    df["Hchain"], df["Lchain"] = df.Hchain.astype(str), df.Lchain.astype(str)
    df["pdb4"] = df["PDB"].str[-4:].str.upper()
    ids = sorted(df.pdb4.unique())
    random.Random(seed).shuffle(ids)
    out = pd.read_csv(CSV_OUT).fillna("").to_dict("records") if os.path.exists(CSV_OUT) else []
    tried = {r["pdb"] for r in out}
    for pdb in ids:
        if sum(r["status"] == "ok" for r in out) >= target:
            break
        if pdb in tried:
            continue
        rec = {"pdb": pdb, "status": "", "antigen": "", "seq": "", "n_res": 0,
               "n_epitope": 0, "auc_rsasa": np.nan, "auc_patch": np.nan}
        try:
            txt = fetch_experimental_pdb(pdb)
            method, res = header_info(txt)
            rows = df[df.pdb4 == pdb]
            if method is None or "X-RAY" not in method:
                rec["status"] = "not_xray"
            elif res is None or res > 3.0:
                rec["status"] = "low_res"
            else:
                model = load(txt)
                skip = set(rows.Hchain) | set(rows.Lchain)
                ag = h = l = None
                for _, r in rows.iterrows():
                    ag = pick_antigen(model, r.Hchain, r.Lchain, skip)
                    if ag:
                        h, l = r.Hchain, r.Lchain
                        break
                if ag is None:
                    rec["status"] = "no_protein_antigen"
                else:
                    seq = chain_seq(model[ag])
                    if any(SequenceMatcher(None, seq, o["seq"], autojunk=False).ratio() >= 0.7
                           for o in out if o["status"] == "ok"):
                        rec["status"] = "redundant"
                    else:
                        a1, a2, n, k = evaluate(model, ag, [h, l])
                    if n == 0 or k == 0:
                        rec["status"] = "empty"
                    else:
                        rec.update(status="ok", antigen=ag, seq=seq, n_res=n,
                                   n_epitope=k, auc_rsasa=a1, auc_patch=a2) 
        except Exception as e:
            rec["status"] = f"error: {str(e)[:60]}"
        out.append(rec)
        pd.DataFrame(out).to_csv(CSV_OUT, index=False)
        print(f">>> {pdb}: {rec['status']}")
    summarize(out)

if __name__ == "__main__":
    main()