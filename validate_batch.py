import sys
sys.path.insert(0, ".")
from idk_some_shit import find_epitope_candidates, fetch_experimental_pdb, get_chain_ids
import math

# (pdb_id, antigen_chain, list_of_antibody_chains)
TEST_SET = [
    ("1VFB", "Y", ["H", "L"]),
    ("1JPS", "T", ["H", "L"]),
    ("1P2C", "P", ["H", "L"]),
    ("2JEL", "P", ["H", "L"]),
    ("3MXW", "C", ["A", "B"]),
    ("2CMR", "A", ["H", "L"]),
]
# NOTE: chain letters above are best-guess from memory of these entries —
# verify each against RCSB's "Entities" tab before trusting fully; if wrong,
# get_real_contact_residues will just return an empty set (visible as 0 in output).

CONTACT_CUTOFF = 5.0  # Angstroms, any-atom distance

def parse_atoms(pdb_text, chain):
    atoms = []
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") and line[21].strip() == chain:
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                resi = int(line[22:26])
                atoms.append((resi, x, y, z))
            except ValueError:
                continue
    return atoms

def get_real_contact_residues(pdb_text, antigen_chain, antibody_chains, cutoff=CONTACT_CUTOFF):
    antigen_atoms = parse_atoms(pdb_text, antigen_chain)
    antibody_atoms = []
    for c in antibody_chains:
        antibody_atoms.extend(parse_atoms(pdb_text, c))

    contact_residues = set()
    for resi, x1, y1, z1 in antigen_atoms:
        for _, x2, y2, z2 in antibody_atoms:
            dist = math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
            if dist <= cutoff:
                contact_residues.add(resi)
                break
    return contact_residues

results = []

for pdb_id, antigen_chain, antibody_chains in TEST_SET:
    print(f"\n=== {pdb_id} (antigen chain {antigen_chain}) ===")
    try:
        pdb_text = fetch_experimental_pdb(pdb_id)
        real_contacts = get_real_contact_residues(pdb_text, antigen_chain, antibody_chains)
        print(f"Real contact residues ({len(real_contacts)}): {sorted(real_contacts)}")

        candidates = find_epitope_candidates(pdb_id, antigen_chain)
        predicted = {c["residue_index"] for c in candidates}
        print(f"Predicted candidates ({len(predicted)}): {sorted(predicted)}")

        hits = predicted & real_contacts
        print(f"Overlap (true positives): {sorted(hits)}")

        precision = len(hits) / len(predicted) if predicted else None
        recall = len(hits) / len(real_contacts) if real_contacts else None
        print(f"Precision: {precision}, Recall: {recall}")

        results.append((pdb_id, len(predicted), len(real_contacts), len(hits), precision, recall))
    except Exception as e:
        print(f"FAILED: {e}")
        results.append((pdb_id, None, None, None, None, None))

print("\n\n=== SUMMARY ===")
print(f"{'PDB':<8}{'Pred':<8}{'Real':<8}{'Hits':<8}{'Precision':<12}{'Recall'}")
for r in results:
    print(f"{r[0]:<8}{str(r[1]):<8}{str(r[2]):<8}{str(r[3]):<8}{str(r[4]):<12}{str(r[5])}")