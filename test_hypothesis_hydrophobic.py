import sys
sys.path.insert(0, ".")
from idk_some_shit import fetch_sequence, predict_structure, compute_sasa, KYTE_DOOLITTLE
from scipy import stats

# (pdb_id, chain, known_tm_score, category)
KNOWN_RESULTS = [
    ("1UBQ", "A", 0.954, "good"),
    ("1LYZ", "A", 0.983, "good"),
    ("1AKE", "A", 0.978, "good"),
    ("1AKE", "B", 0.979, "good"),
    ("3LZT", "A", 0.869, "good"),
    ("4HHB", "A", 0.992, "good"),   # alpha-globin
    ("4HHB", "C", 0.993, "good"),   # alpha-globin
    ("1MLC", "E", 0.981, "good"),   # lysozyme
    ("1MLC", "F", 0.988, "good"),   # lysozyme
    ("4HHB", "B", 0.339, "poor"),   # beta-globin
    ("4HHB", "D", 0.340, "poor"),   # beta-globin
    ("2HHB", "A", 0.348, "poor"),   # beta-globin (note: check if A/C are actually beta here)
    ("2HHB", "C", 0.349, "poor"),
    ("1MLC", "A", 0.114, "poor"),   # antibody chain
    ("1MLC", "B", 0.114, "poor"),
    ("1MLC", "C", 0.116, "poor"),
    ("1MLC", "D", 0.133, "poor"),
]

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

t_stat, p_value = stats.ttest_ind(good_fracs, poor_fracs)
print(f"t-statistic: {t_stat:.3f}, p-value: {p_value:.4f}")
