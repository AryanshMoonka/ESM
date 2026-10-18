import sys
sys.path.insert(0, ".")
from idk_some_shit import fetch_sequence, predict_structure, compute_sasa, KYTE_DOOLITTLE, benchmark_one_chain
from scipy import stats
import random, requests
import numpy as np

def fetch_pdb_chains(n=150, seed=42):
    query = {
        "query": {
            "type": "group", "logical_operator": "and",
            "nodes": [
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "exptl.method", "operator": "exact_match",
                    "value": "X-RAY DIFFRACTION"}},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_entry_info.resolution_combined",
                    "operator": "less_or_equal", "value": 2.5}},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "entity_poly.rcsb_entity_polymer_type",
                    "operator": "exact_match", "value": "Protein"}},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "entity_poly.rcsb_sample_sequence_length",
                    "operator": "range", "value": {"from": 50, "to": 250,
                    "include_lower": True, "include_upper": True}}},
            ],
        },
        "return_type": "polymer_entity",
        "request_options": {
            "group_by": {"aggregation_method": "sequence_identity",
                         "similarity_cutoff": 30},
            "group_by_return_type": "representatives",
            "return_all_hits": True,
        },
    }
    r = requests.post("https://search.rcsb.org/rcsbsearch/v2/query",
                      json=query, timeout=60)
    r.raise_for_status()
    ids = [h["identifier"] for h in r.json()["result_set"]]  # e.g. "1ABC_1"
    random.Random(seed).shuffle(ids)

    out = []
    for ent in ids:
        if len(out) >= n:
            break
        pdb_id, ent_id = ent.split("_")
        d = requests.get(
            f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{ent_id}",
            timeout=30).json()
        chains = d["rcsb_polymer_entity_container_identifiers"]["auth_asym_ids"]
        out.append((pdb_id, chains[0]))
    return out

import json, os, math

CACHE = "tm_cache.json"

def clear_failed():
    cache = json.load(open(CACHE))
    json.dump({k: v for k, v in cache.items() if v is not None}, open(CACHE, "w"))

def build_results(chains, good_cut=0.5):
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    out = []
    for i, (pdb_id, chain) in enumerate(chains):
        key = f"{pdb_id}:{chain}"
        if key not in cache:
            try:
                r = benchmark_one_chain(pdb_id, chain)
                bad = r.error or math.isnan(r.tm_score)
                cache[key] = None if bad else r.tm_score
            except Exception as e:
                print(f"{key} FAILED: {e}")
                cache[key] = None
            json.dump(cache, open(CACHE, "w"))
        tm = cache[key]
        if tm is not None:
            out.append((pdb_id, chain, tm, "good" if tm >= good_cut else "poor"))
        if i % 10 == 0:
            print(f"{i}/{len(chains)}")
    return out

KNOWN_RESULTS = build_results(fetch_pdb_chains(100))

def surface_hydrophobicity_fraction(sequence, sasa, sasa_threshold=30.0):
    """Fraction of surface-exposed residues that are hydrophobic (Kyte-Doolittle > 0)."""
    exposed_hydrophobic = 0
    exposed_total = 0
    length = min(len(sequence), len(sasa))
    for i in range(length):
        if sasa[i] > sasa_threshold:
            exposed_total += 1
            if KYTE_DOOLITTLE.get(sequence[i], 0) > 0:
                exposed_hydrophobic += 1
    return exposed_hydrophobic / exposed_total if exposed_total else None

print(f"{'PDB:Chain':<12}{'Category':<10}{'TM-score':<12}{'SurfHydroFrac'}")
print("-" * 55)

good_fracs = []
poor_fracs = []

for pdb_id, chain, tm_score, category in KNOWN_RESULTS:
    try:
        sequence = fetch_sequence(pdb_id, chain=chain)
        predicted_pdb = predict_structure(sequence)
        sasa = compute_sasa(predicted_pdb, chain=None)
        frac = surface_hydrophobicity_fraction(sequence, sasa)
        print(f"{pdb_id}:{chain:<9}{category:<10}{tm_score:<12}{frac}")
        if frac is not None:
            if category == "good":
                good_fracs.append(frac)
            else:
                poor_fracs.append(frac)
    except Exception as e:
        print(f"{pdb_id}:{chain:<9}FAILED: {e}")

print("\n=== COMPARISON ===")
if good_fracs:
    print(f"Good chains  — mean surface hydrophobic fraction: {sum(good_fracs)/len(good_fracs):.3f} (n={len(good_fracs)})")
if poor_fracs:
    print(f"Poor chains  — mean surface hydrophobic fraction: {sum(poor_fracs)/len(poor_fracs):.3f} (n={len(poor_fracs)})")

def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    pooled = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1))
                     / (len(a)+len(b)-2))
    return (a.mean() - b.mean()) / pooled

if len(good_fracs) >= 2 and len(poor_fracs) >= 2:
    t, p = stats.ttest_ind(good_fracs, poor_fracs, equal_var=False)
    u, p_mw = stats.mannwhitneyu(good_fracs, poor_fracs, alternative="two-sided")
    print(f"Welch t = {t:.3f}, p = {p:.4f}")
    print(f"Mann-Whitney U = {u:.1f}, p = {p_mw:.4f}")
    print(f"Cohen's d = {cohens_d(good_fracs, poor_fracs):.2f}")
else:
    print("Not enough chains in one group for a test")
