import sys
sys.path.insert(0, ".")
from idk_some_shit import find_epitope_candidates, group_linear_epitopes

pdb_id = "1MLC"
chain = "E"  # replace with the actual lysozyme chain letter from RCSB's page for 1MLC

candidates = find_epitope_candidates(pdb_id, chain)
print(f"Candidates found: {len(candidates)}")
for c in candidates:
    print(c)

grouped = group_linear_epitopes(candidates)
print(f"\nGrouped regions: {len(grouped)}")
for g in grouped:
    print(g)
    
pdb_id = "1MLC"
chain = "E"