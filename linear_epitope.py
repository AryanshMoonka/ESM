import io, sys, copy
import numpy as np
from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.SASA import ShrakeRupley
from scipy.stats import mannwhitneyu
from idk_some_shit import fetch_experimental_pdb, _AA3

# Max accessible area per residue type (Tien et al. 2013), in square angstroms
MAX_ASA = {"A":129,"R":274,"N":195,"D":193,"C":167,"Q":225,"E":223,"G":104,"H":224,
           "I":197,"L":201,"K":236,"M":224,"F":240,"P":159,"S":155,"T":172,"W":285,
           "Y":263,"V":174}

def load(pdb_text):
    return PDBParser(QUIET=True).get_structure("x", io.StringIO(pdb_text))[0]

def list_chains(model):
    for ch in model:
        res = [r for r in ch if r.id[0] == " "]
        seq = "".join(_AA3.get(r.get_resname(), "X") for r in res)
        print(f"chain {ch.id}: {len(res)} residues, starts {seq[:20]}")

def rsasa(model, chain_id):
    """Relative SASA per residue of ONE chain on its own (unbound state)."""
    ch = copy.deepcopy(model[chain_id])
    for r in [r for r in ch if r.id[0] != " "]:
        ch.detach_child(r.id)
    ShrakeRupley().compute(ch, level="R")
    out = []
    for r in ch:
        aa = _AA3.get(r.get_resname(), "X")
        if aa in MAX_ASA and "CA" in r:
            out.append((r.id, aa, min(r.sasa / MAX_ASA[aa], 1.0), r["CA"].coord))
    return out

def true_epitope(model, antigen, partners, cutoff=4.5):
    atoms = [a for p in partners for r in model[p] if r.id[0] == " " for a in r]
    ns = NeighborSearch(atoms)
    ep = set()
    for r in model[antigen]:
        if r.id[0] == " " and any(ns.search(a.coord, cutoff) for a in r):
            ep.add(r.id)
    return ep

def auc(scores, labels):
    s, l = np.array(scores), np.array(labels, bool)
    return mannwhitneyu(s[l], s[~l]).statistic / (l.sum() * (~l).sum())

def evaluate(model, antigen, partners):
    rows = rsasa(model, antigen)
    ep = true_epitope(model, antigen, partners)
    labels = [r[0] in ep for r in rows]
    base = [r[2] for r in rows]
    xyz = np.array([r[3] for r in rows])
    patch = []
    for i in range(len(rows)):
        near = np.linalg.norm(xyz - xyz[i], axis=1) < 10.0
        patch.append(np.mean([base[j] for j in np.where(near)[0]]))
    print(f"{len(rows)} residues, {sum(labels)} in true epitope")
    print(f"AUC, residue rSASA alone:     {auc(base, labels):.3f}")
    print(f"AUC, 10 A patch-mean rSASA:   {auc(patch, labels):.3f}")
    return auc(base, labels), auc(patch, labels), len(rows), sum(labels)
if __name__ == "__main__":
    model = load(fetch_experimental_pdb(sys.argv[1]))
    if len(sys.argv) == 2:
        list_chains(model)
    else:
        evaluate(model, sys.argv[2], sys.argv[3].split(","))
   