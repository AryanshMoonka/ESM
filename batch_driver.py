import sys
sys.path.insert(0, ".")
from idk_some_shit import benchmark_all_chains
import csv
import statistics

PDB_IDS = [
    "1UBQ",   # monomer
    "4HHB",   # hetero-tetramer
    "6VXX",   # homo-trimer
    "1LYZ",   # monomer
    "1MLC",   # hetero-complex (antibody-antigen)
    "2HHB",   # hetero-tetramer (alt state)
    "1AKE",   # monomer
    "3LZT",   # monomer
    # add/remove IDs here as we confirm which ones fetch cleanly
]

all_results = []

for pdb_id in PDB_IDS:
    print(f"\n=== {pdb_id} ===")
    try:
        chain_results = benchmark_all_chains(pdb_id)
        for r in chain_results:
            all_results.append({
                "pdb_id": pdb_id,
                "rmsd": r.rmsd,
                "tm_score": r.tm_score,
                "error": r.error or "",
            })
            print(f"  RMSD={r.rmsd}, TM-score={r.tm_score}, error={r.error}")
    except Exception as e:
        print(f"  FAILED: {e}")
        all_results.append({"pdb_id": pdb_id, "rmsd": None, "tm_score": None, "error": str(e)})

# Save to CSV
with open("benchmark_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["pdb_id", "rmsd", "tm_score", "error"])
    writer.writeheader()
    writer.writerows(all_results)

# Summary stats
valid_rmsd = [r["rmsd"] for r in all_results if r["rmsd"] is not None]
valid_tm = [r["tm_score"] for r in all_results if r["tm_score"] is not None]

print("\n\n=== SUMMARY ===")
print(f"Total chain results: {len(all_results)}")
print(f"Valid RMSD count: {len(valid_rmsd)}")
if valid_rmsd:
    print(f"Mean RMSD: {statistics.mean(valid_rmsd):.3f}, Median: {statistics.median(valid_rmsd):.3f}, Stdev: {statistics.stdev(valid_rmsd) if len(valid_rmsd) > 1 else 0:.3f}")
if valid_tm:
    print(f"Mean TM-score: {statistics.mean(valid_tm):.3f}, Median: {statistics.median(valid_tm):.3f}, Stdev: {statistics.stdev(valid_tm) if len(valid_tm) > 1 else 0:.3f}")

print(f"\nResults saved to benchmark_results.csv")