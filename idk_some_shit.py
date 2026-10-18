##Copyright Reserved (C) (2026) by Aryansh Moonka. All rights reserved. Unathorized copying, distribution, or modifications of this file, via any medium is strictly prohibited. Proprietary and confidential.

import csv
import io
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional
import hashlib
import os
import numpy as np
import requests
import streamlit as st
import py3Dmol
from Bio.PDB import PDBParser, PDBIO, Superimposer
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.Structure import Structure
from difflib import SequenceMatcher

THREE_TO_ONE = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
    "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
    "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"
}

def sequence_from_atoms(pdb_text: str, chain: str) -> str:
    """Build a sequence directly from ATOM CA records for a given chain, avoiding fragile FASTA header parsing."""
    seen = {}
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") and line[21].strip() == chain:
            atom_name = line[12:16].strip()
            if atom_name == "CA":
                resi = int(line[22:26])
                resn = line[17:20].strip()
                if resn in THREE_TO_ONE:
                    seen[resi] = THREE_TO_ONE[resn]
    return "".join(seen[k] for k in sorted(seen))

# ---------------------------------------------------------------------------
# Config / constants
# ---------------------------------------------------------------------------
MAX_ESMFOLD_LENGTH = 400  # ESM Atlas API practical limit
CHUNK_OVERLAP = 50  # residues shared between adjacent chunks, used for alignment

ESMFOLD_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"
PDB_FETCH_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
PDB_SEQ_URL = "https://www.rcsb.org/fasta/entry/{pdb_id}"
PDB_METADATA_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"

PATHOGEN_KEYWORDS = [
    "virus", "viral", "spike glycoprotein", "capsid", "envelope glycoprotein",
    "hemagglutinin", "coronavirus", "influenza", "bacteri", "toxin",
    "pathogen", "antigen", "nucleocapsid",
]

STANDARD_AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}

# Fill this in with 15-25 PDB IDs, chosen deliberately for variety:
# different lengths, fold classes (all-alpha, all-beta, mixed), and some
# recent/less-common structures. Document WHY you picked each one in your
# research data book -- judges will ask.
PROTEIN_LIST = [
    "1UBQ",   # Ubiquitin - small, all-beta, classic benchmark
    "1LYZ",   # Lysozyme - small, mixed alpha/beta
    "6LYZ",
    "1CRN",   # Crambin - very small
    "4HHB",   # Hemoglobin subunit - all-alpha
    # Add more PDB IDs here to reach 15-25 total.
]

# ---------------------------------------------------------------------------
# Disk prediction cache — persists across app restarts and redeployments
# ---------------------------------------------------------------------------
CACHE_DIR = "prediction_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_key(sequence: str) -> str:
    return hashlib.md5(sequence.encode()).hexdigest()

def get_cached_prediction(sequence: str) -> Optional[str]:
    path = os.path.join(CACHE_DIR, _cache_key(sequence) + ".pdb")
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read()
    return None

def save_cached_prediction(sequence: str, pdb_text: str):
    path = os.path.join(CACHE_DIR, _cache_key(sequence) + ".pdb")
    with open(path, "w") as f:
        f.write(pdb_text)


@dataclass
class BenchmarkResult:
    pdb_id: str
    sequence_length: int
    rmsd: float
    tm_score: float
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Chunking + stitching (for sequences over MAX_ESMFOLD_LENGTH)
# ---------------------------------------------------------------------------
def chunk_sequence(sequence: str, max_length: int = 350, overlap: int = CHUNK_OVERLAP):
    """Split a long sequence into overlapping windows, each <= max_length."""
    chunks = []
    step = max_length - overlap
    start = 0
    while start < len(sequence):
        end = min(start + max_length, len(sequence))
        chunks.append((start, end, sequence[start:end]))
        if end == len(sequence):
            break
        start += step
    return chunks


def _parse_pdb_text(pdb_text: str):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("chunk", io.StringIO(pdb_text))
    return structure[0]["A"]  # ESMFold single-chain output, chain A


def stitch_predictions(chunk_results, chunk_ranges):
    """
    chunk_results: list of PDB text strings (one per chunk, in order)
    chunk_ranges: list of (start, end) residue index tuples matching chunk_results
    Returns: single merged PDB text string
    """
    ref_chain = _parse_pdb_text(chunk_results[0])
    merged_residues = list(ref_chain.get_residues())

    for i in range(1, len(chunk_results)):
        prev_start, prev_end = chunk_ranges[i - 1]
        cur_start, cur_end = chunk_ranges[i]
        overlap_len = prev_end - cur_start

        cur_chain = _parse_pdb_text(chunk_results[i])
        cur_residues = list(cur_chain.get_residues())

        fixed_atoms = [res["CA"] for res in merged_residues[-overlap_len:] if "CA" in res]
        moving_atoms = [res["CA"] for res in cur_residues[:overlap_len] if "CA" in res]

        n = min(len(fixed_atoms), len(moving_atoms))
        sup = Superimposer()
        sup.set_atoms(fixed_atoms[:n], moving_atoms[:n])
        sup.apply(cur_chain.get_atoms())

        merged_residues.extend(cur_residues[overlap_len:])

    structure = Structure("stitched")
    model = Model(0)
    chain = Chain("A")
    for idx, res in enumerate(merged_residues, start=1):
        res.id = (" ", idx, " ")
        chain.add(res)
    model.add(chain)
    structure.add(model)

    output = io.StringIO()
    io_writer = PDBIO()
    io_writer.set_structure(structure)
    io_writer.save(output)
    return output.getvalue()


def predict_structure_chunked(sequence: str, retries: int = 5) -> str:
    if len(sequence) <= MAX_ESMFOLD_LENGTH:
        return predict_structure(sequence, retries=retries)

    chunks = chunk_sequence(sequence)
    chunk_results = []
    progress_bar = st.progress(0, text="Starting chunked prediction...")

    for i, (start, end, subseq) in enumerate(chunks):
        progress_bar.progress(i / len(chunks), text=f"Chunk {i + 1}/{len(chunks)}: residues {start}-{end}")
        pdb_text = predict_structure(subseq, retries=retries)
        chunk_results.append(pdb_text)

    progress_bar.progress(1.0, text="Stitching chunks together...")
    chunk_ranges = [(s, e) for s, e, _ in chunks]
    result = stitch_predictions(chunk_results, chunk_ranges)
    progress_bar.empty()
    return result


# ---------------------------------------------------------------------------
# Fetching data
# ---------------------------------------------------------------------------
@st.cache_data
def fetch_experimental_pdb(pdb_id: str, retries: int = 3) -> str:
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.get(PDB_FETCH_URL.format(pdb_id=pdb_id), timeout=30)
            response.raise_for_status()
            return response.text
        except requests.RequestException as error:
            last_error = error
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch experimental PDB for {pdb_id} after {retries} attempts: {last_error}")


@st.cache_data
def fetch_pdb_metadata(pdb_id: str) -> dict:
    """Fetch entry metadata (title, organism) from RCSB's data API."""
    response = requests.get(PDB_METADATA_URL.format(pdb_id=pdb_id), timeout=60)
    response.raise_for_status()
    data = response.json()
    title = data.get("struct", {}).get("title", "")
    return {"title": title, "raw": data}


def is_likely_pathogen(pdb_id: str) -> bool:
    """Heuristic keyword match against entry title. Not true classification —
    flags entries whose title suggests a pathogen-associated protein."""
    try:
        metadata = fetch_pdb_metadata(pdb_id)
        title = metadata["title"].lower()
        return any(keyword in title for keyword in PATHOGEN_KEYWORDS)
    except Exception as e:
        st.warning(f"Pathogen metadata check failed for {pdb_id}: {e}")
        return False


@st.cache_data
def fetch_sequence(pdb_id: str, chain: str = None) -> str:
    response = requests.get(PDB_SEQ_URL.format(pdb_id=pdb_id), timeout=30)
    response.raise_for_status()
    lines = response.text.strip().splitlines()

    records = {}
    current_chains: List[str] = []
    current_seq: List[str] = []

    def flush():
        if current_chains and current_seq:
            seq = "".join(current_seq)
            for c in current_chains:
                records[c] = seq

    for line in lines:
        if line.startswith(">"):
            flush()
            current_seq = []
            header = line[1:]
            chain_part = header.split("|")[1] if "|" in header else ""
            chain_part = chain_part.replace("Chains", "").replace("Chain", "")
            current_chains = [c.strip() for c in chain_part.split(",") if c.strip()]
        else:
            current_seq.append(line.strip())
    flush()

    if chain is not None:
        if chain not in records:
            raise ValueError(f"Chain {chain} not found in FASTA for {pdb_id}")
        return records[chain]

    return next(iter(records.values()))

@st.cache_data
def predict_structure(sequence, retries=5):
    # Check disk cache first — returns instantly if already predicted
    cached = get_cached_prediction(sequence)
    if cached:
        return cached

    if len(sequence) > MAX_ESMFOLD_LENGTH:
        result = predict_structure_chunked(sequence, retries=retries)
        save_cached_prediction(sequence, result)
        return result

    last_error = None
    for attempt in range(retries):
        try:
            response = requests.post(ESMFOLD_URL, data=sequence, timeout=300)
            response.raise_for_status()
            result = response.text
            save_cached_prediction(sequence, result)
            return result
        except requests.RequestException as error:
            last_error = error
            time.sleep(10 * (attempt + 1))

    raise RuntimeError(f"ESMFold prediction failed after {retries} attempts: {last_error}")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def extract_ca_coordinates(pdb_text: str, chain: Optional[str] = None) -> np.ndarray:
    """
    Extract C-alpha atom coordinates in residue order.
    If `chain` is given, only that chain is used; otherwise the first
    chain encountered is used (important for multi-chain PDB entries).
    """
    coords = []
    target_chain = chain
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue
        this_chain = line[21].strip()
        if target_chain is None:
            target_chain = this_chain
        if this_chain != target_chain:
            continue
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        coords.append((x, y, z))
    return np.array(coords, dtype=float)


def is_protein_chain(pdb_text: str, chain: str) -> bool:
    """Check whether a chain is protein (has CA atoms with standard amino acid names)."""
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[21].strip() != chain:
            continue
        atom_name = line[12:16].strip()
        res_name = line[17:20].strip()
        if atom_name == "CA" and res_name in STANDARD_AMINO_ACIDS:
            return True
    return False


def get_chain_ids(pdb_id: str) -> list:
    pdb_text = fetch_experimental_pdb(pdb_id)
    chains = []
    for line in pdb_text.splitlines():
        if line.startswith("ATOM"):
            c = line[21].strip()
            if c and c not in chains:
                chains.append(c)
    return chains


def get_protein_chains(pdb_id: str) -> list:
    """Return only protein chains for a PDB ID, skipping DNA/RNA chains."""
    experimental_pdb = fetch_experimental_pdb(pdb_id)
    all_chains = get_chain_ids(pdb_id)
    protein_chains = [c for c in all_chains if is_protein_chain(experimental_pdb, c)]
    for c in set(all_chains) - set(protein_chains):
        print(f"[{pdb_id}] Skipping chain {c}: nucleic acid or non-protein, not scoreable by ESMFold.")
    print(f"DEBUG [{pdb_id}]: all_chains={all_chains}, protein_chains={protein_chains}")
    return protein_chains


# ---------------------------------------------------------------------------
# Kabsch alignment + RMSD
# ---------------------------------------------------------------------------
def kabsch_rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """
    Optimally superpose coords_b onto coords_a (equal length, paired by
    residue index) and return the RMSD after alignment.
    """
    assert coords_a.shape == coords_b.shape, "Coordinate sets must be equal length"

    a_centered = coords_a - coords_a.mean(axis=0)
    b_centered = coords_b - coords_b.mean(axis=0)

    h = b_centered.T @ a_centered
    u, s, vt = np.linalg.svd(h)

    d = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1, 1, d])
    rotation = vt.T @ correction @ u.T

    b_aligned = (rotation @ b_centered.T).T

    diff = a_centered - b_aligned
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def kabsch_transform(coords_a: np.ndarray, coords_b: np.ndarray):
    mean_a = coords_a.mean(axis=0)
    mean_b = coords_b.mean(axis=0)
    a_centered = coords_a - mean_a
    b_centered = coords_b - mean_b

    h = b_centered.T @ a_centered
    u, s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1, 1, d])
    rotation = vt.T @ correction @ u.T

    return rotation, mean_a, mean_b


def apply_transform(coords: np.ndarray, rotation, mean_a, mean_b):
    return (rotation @ (coords - mean_b).T).T + mean_a


# ---------------------------------------------------------------------------
# TM-score
# ---------------------------------------------------------------------------
def tm_score(coords_a: np.ndarray, coords_b: np.ndarray, target_length: int) -> float:
    """
    Standard TM-score (Zhang & Skolnick, 2004) using the fixed d0 formula.
    coords_a, coords_b must already be paired by residue index.
    """
    assert coords_a.shape == coords_b.shape

    a_centered = coords_a - coords_a.mean(axis=0)
    b_centered = coords_b - coords_b.mean(axis=0)

    h = b_centered.T @ a_centered
    u, s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1, 1, d])
    rotation = vt.T @ correction @ u.T
    b_aligned = (rotation @ b_centered.T).T

    distances = np.sqrt(np.sum((a_centered - b_aligned) ** 2, axis=1))

    if target_length > 15:
        d0 = 1.24 * (target_length - 15) ** (1 / 3) - 1.8
    else:
        d0 = 0.5

    d0 = max(d0, 0.5)

    score_terms = 1.0 / (1.0 + (distances / d0) ** 2)
    return float(np.sum(score_terms) / target_length)


# ---------------------------------------------------------------------------
# Main benchmarking loop
# ---------------------------------------------------------------------------
def benchmark_one(pdb_id: str) -> BenchmarkResult:
    print(f"[{pdb_id}] fetching experimental structure...")
    experimental_pdb = fetch_experimental_pdb(pdb_id)
    experimental_coords = extract_ca_coordinates(experimental_pdb)

    print(f"[{pdb_id}] fetching sequence...")
    sequence = fetch_sequence(pdb_id)
    if not sequence:
        return BenchmarkResult(pdb_id, 0, float("nan"), float("nan"),
                                error="Could not retrieve sequence")

    print(f"[{pdb_id}] requesting ESMFold prediction ({len(sequence)} residues)...")
    predicted_pdb = predict_structure(sequence)
    predicted_coords = extract_ca_coordinates(predicted_pdb)

    # Experimental PDB files often have missing residues (crystal gaps).
    # Truncating to the shorter length is a simple safeguard, but a more
    # rigorous approach aligns by residue NUMBER, not just position.
    # Flag this as a known limitation in your methodology write-up.
    length = min(len(experimental_coords), len(predicted_coords))
    if length == 0:
        return BenchmarkResult(pdb_id, len(sequence), float("nan"), float("nan"),
                                error="No matching CA atoms found")

    exp_trimmed = experimental_coords[:length]
    pred_trimmed = predicted_coords[:length]

    rmsd = kabsch_rmsd(exp_trimmed, pred_trimmed)
    tm = tm_score(exp_trimmed, pred_trimmed, target_length=length)

    print(f"[{pdb_id}] RMSD = {rmsd:.2f} Å, TM-score = {tm:.3f}")
    return BenchmarkResult(pdb_id, len(sequence), rmsd, tm)


def benchmark_one_chain(pdb_id: str, chain: str) -> BenchmarkResult:
    print(f"[{pdb_id}:{chain}] fetching experimental structure...")
    experimental_pdb = fetch_experimental_pdb(pdb_id)
    obs_seq, experimental_coords = extract_ca_residues(experimental_pdb, chain)

    print(f"[{pdb_id}:{chain}] fetching sequence...")
    sequence = fetch_sequence(pdb_id, chain=chain)
    if not sequence:
        return BenchmarkResult(pdb_id, 0, float("nan"), float("nan"),
                               error="Could not retrieve sequence")

    print(f"[{pdb_id}:{chain}] requesting ESMFold prediction ({len(sequence)} residues)...")
    predicted_pdb = predict_structure(sequence)
    predicted_coords = extract_ca_coordinates(predicted_pdb)

    sm = SequenceMatcher(None, sequence, obs_seq, autojunk=False)
    pi, ei = [], []
    for a, b, size in sm.get_matching_blocks():
        pi.extend(range(a, a + size))
        ei.extend(range(b, b + size))
    pi, ei = np.array(pi, dtype=int), np.array(ei, dtype=int)
    keep = pi < len(predicted_coords)
    pi, ei = pi[keep], ei[keep]

    length = len(pi)
    if length < 20 or length < 0.5 * len(obs_seq):
        return BenchmarkResult(pdb_id, len(sequence), float("nan"), float("nan"),
                               error="Sequence/structure alignment too short")

    exp_trimmed = experimental_coords[ei]
    pred_trimmed = predicted_coords[pi]

    rmsd = kabsch_rmsd(exp_trimmed, pred_trimmed)
    tm = tm_score(exp_trimmed, pred_trimmed, target_length=length)

    print(f"[{pdb_id}:{chain}] RMSD = {rmsd:.2f} Å, TM-score = {tm:.3f}")
    return BenchmarkResult(pdb_id, len(sequence), rmsd, tm)

def benchmark_all_chains(pdb_id: str) -> list:
    """Benchmarks only protein chains — DNA/RNA chains are skipped since
    ESMFold cannot predict or meaningfully score them (see get_protein_chains)."""
    chain_ids = get_protein_chains(pdb_id)
    results = []
    for c in chain_ids:
        try:
            results.append(benchmark_one_chain(pdb_id, c))
        except Exception as exc:
            print(f"[{pdb_id}:{c}] FAILED: {exc}")
    return results

_AA3 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
        "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
        "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V","MSE":"M"}

def extract_ca_residues(pdb_text, chain):
    """One-letter sequence and CA coords of the resolved residues of one chain.
    First model only, first altloc only."""
    seq, coords, seen = [], [], set()
    for line in pdb_text.splitlines():
        if line.startswith("ENDMDL"):
            break
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        if line[12:16].strip() != "CA" or line[21].strip() != chain:
            continue
        if line[16] not in (" ", "A"):
            continue
        resname = line[17:20].strip()
        if line.startswith("HETATM") and resname != "MSE":
            continue
        rid = (line[22:26], line[26])
        if rid in seen:
            continue
        seen.add(rid)
        seq.append(_AA3.get(resname, "X"))
        coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return "".join(seq), np.array(coords, dtype=float)

def get_chain_prediction(pdb_id: str, chain: str):
    experimental_pdb = fetch_experimental_pdb(pdb_id)
    obs_seq, exp_coords = extract_ca_residues(experimental_pdb, chain)
    sequence = fetch_sequence(pdb_id, chain=chain)
    predicted_pdb = predict_structure(sequence)
    pred_coords = extract_ca_coordinates(predicted_pdb)

    sm = SequenceMatcher(None, sequence, obs_seq, autojunk=False)
    pi, ei = [], []
    for a, b, size in sm.get_matching_blocks():
        pi.extend(range(a, a + size))
        ei.extend(range(b, b + size))
    pi, ei = np.array(pi, dtype=int), np.array(ei, dtype=int)
    keep = pi < len(pred_coords)
    pi, ei = pi[keep], ei[keep]
    return exp_coords[ei], pred_coords[pi]

def benchmark_quaternary(pdb_id: str):
    chain_ids = get_chain_ids(pdb_id)
    if len(chain_ids) < 2:
        return {"error": f"{pdb_id} has only one chain — no quaternary structure to assess."}

    chain_data = {}
    for c in chain_ids:
        exp, pred = get_chain_prediction(pdb_id, c)
        chain_data[c] = {"exp": exp, "pred": pred}
    print(f"DEBUG [{pdb_id}] chain {c}: exp_coords len={len(exp)}, pred_coords len={len(pred)}")

    ref_chain = chain_ids[0]
    rotation, mean_a, mean_b = kabsch_transform(
        chain_data[ref_chain]["exp"], chain_data[ref_chain]["pred"]
    )

    exp_all, pred_all = [], []
    for c in chain_ids:
        exp_coords = chain_data[c]["exp"]
        pred_coords = chain_data[c]["pred"]
        if len(exp_coords) == 0 or len(pred_coords) == 0:
            continue
        exp_all.append(exp_coords)
        pred_all.append(apply_transform(pred_coords, rotation, mean_a, mean_b))

    if not exp_all:
        return {"error": f"{pdb_id} has no chains with valid matching coordinates."}

    exp_complex = np.vstack(exp_all)
    pred_complex = np.vstack(pred_all)
    complex_rmsd = float(np.sqrt(np.mean(np.sum((exp_complex - pred_complex) ** 2, axis=1))))

    return {
        "pdb_id": pdb_id,
        "chains": chain_ids,
        "reference_chain": ref_chain,
        "complex_rmsd": complex_rmsd,
        "exp_coords": exp_all,
        "pred_coords": pred_all,
    }


# ---------------------------------------------------------------------------
# SASA / pLDDT / epitope analysis
# ---------------------------------------------------------------------------
def compute_sasa(pdb_text: str, chain: str = None):
    """Simplified SASA proxy using neighbour count — works on ESMFold output."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", io.StringIO(pdb_text))
    ca_coords = []
    for model in structure:
        for c in model:
            if chain and c.id != chain:
                continue
            for residue in c:
                if residue.has_id("CA"):
                    ca_coords.append(residue["CA"].get_vector().get_array())
        break
    if not ca_coords:
        return []
    ca_coords = np.array(ca_coords)
    sasa_proxy = []
    for i, coord in enumerate(ca_coords):
        distances = np.linalg.norm(ca_coords - coord, axis=1)
        neighbours = np.sum(distances < 10.0) - 1
        sasa_proxy.append(max(0.0, 100.0 - neighbours * 8.0))
    return sasa_proxy

def extract_plddt(pdb_text: str, chain: str = None):
    """ESMFold stores per-residue pLDDT in the B-factor column."""
    plddt = []
    seen_residues = set()
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue
        this_chain = line[21].strip()
        if chain and this_chain != chain:
            continue
        res_num = line[22:26].strip()
        if res_num in seen_residues:
            continue
        seen_residues.add(res_num)
        try:
            bfactor = float(line[60:66].strip())
        except ValueError:
            bfactor = 0.0
        plddt.append(bfactor)
    return plddt


def find_epitope_candidates(pdb_id: str, chain: str, sasa_threshold=30.0, plddt_threshold=0.7):
    """Flag residues that are both surface-exposed AND confidently predicted."""
    experimental_pdb = fetch_experimental_pdb(pdb_id)
    sequence = sequence_from_atoms(experimental_pdb, chain)
    predicted_pdb = predict_structure(sequence)
    sasa = compute_sasa(predicted_pdb, chain=None)  # predicted structure is single-chain
    print(f"SASA raw values: {sasa[:10]}")
    plddt = extract_plddt(predicted_pdb, chain=None)
    print(f"pLDDT raw values: {plddt[:10]}")
    length = min(len(sasa), len(plddt))
    candidates = []
    for i in range(length):
     if sasa[i] > sasa_threshold and plddt[i] >= plddt_threshold:
            candidates.append({"residue_index": i + 1, "sasa": round(sasa[i], 1), "plddt": round(plddt[i] if i < len(plddt) else 0.0, 1)})
    return candidates


def group_linear_epitopes(candidates: list, min_length: int = 5, max_gap: int = 1) -> list:
    """Group individual epitope-candidate residues into contiguous linear stretches."""
    if not candidates:
        return []
    indices = sorted(c["residue_index"] for c in candidates)
    groups = []
    current = [indices[0]]
    for idx in indices[1:]:
        if idx - current[-1] <= max_gap + 1:
            current.append(idx)
        else:
            groups.append(current)
            current = [idx]
    groups.append(current)
    return [
        {"start": g[0], "end": g[-1], "length": g[-1] - g[0] + 1}
        for g in groups if (g[-1] - g[0] + 1) >= min_length
    ]


def find_glycosylation_sites(sequence: str) -> list:
    """Find N-X-S/T sequon motifs (N-glycosylation sites), X != Proline."""
    sites = []
    for match in re.finditer(r"N[^P][ST]", sequence):
        sites.append(match.start() + 1)  # 1-indexed residue position
    return sites

# ---------------------------------------------------------------------------
# Epitope propensity scales (Kyte-Doolittle hydrophilicity, Parker, Karplus-Schulz flexibility)
# ---------------------------------------------------------------------------

# Kyte-Doolittle hydrophobicity (LOWER = more hydrophilic = more epitope-likely)
KYTE_DOOLITTLE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

# Parker hydrophilicity (HIGHER = more epitope-likely)
PARKER_HYDROPHILICITY = {
    'A': 2.1, 'R': 4.2, 'N': 7.0, 'D': 10.0, 'C': 1.4,
    'Q': 6.0, 'E': 7.8, 'G': 5.7, 'H': 2.1, 'I': -8.0,
    'L': -9.2, 'K': 5.7, 'M': -4.2, 'F': -9.2, 'P': 2.1,
    'S': 6.5, 'T': 5.2, 'W': -10.0, 'Y': -1.9, 'V': -3.7
}

# Karplus-Schulz flexibility (HIGHER = more flexible = more epitope-likely)
KARPLUS_SCHULZ_FLEXIBILITY = {
    'A': 0.984, 'R': 1.008, 'N': 1.048, 'D': 1.068, 'C': 0.906,
    'Q': 1.037, 'E': 1.094, 'G': 1.031, 'H': 0.950, 'I': 0.927,
    'L': 0.935, 'K': 1.102, 'M': 0.952, 'F': 0.915, 'P': 1.049,
    'S': 1.046, 'T': 0.997, 'W': 0.904, 'Y': 0.929, 'V': 0.931
}

def _normalize(values: list) -> list:
    """Min-max normalize a list of floats to 0-1 range."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]

def compute_epitope_scores(sequence: str, sasa: list, plddt: list) -> list:
    """
    Combine SASA, pLDDT, hydrophilicity, and flexibility into a single
    per-residue epitope propensity score (0-1, higher = more epitope-likely).
    """
    length = min(len(sequence), len(sasa), len(plddt))

    hydrophobicity = [KYTE_DOOLITTLE.get(sequence[i], 0.0) for i in range(length)]
    hydrophilicity_raw = [-h for h in hydrophobicity]  # flip sign: lower hydrophobicity = more hydrophilic
    flexibility = [KARPLUS_SCHULZ_FLEXIBILITY.get(sequence[i], 1.0) for i in range(length)]

    sasa_n = _normalize(sasa[:length])
    plddt_n = plddt[:length]  # already 0-1 scale
    hydrophilicity_n = _normalize(hydrophilicity_raw)
    flexibility_n = _normalize(flexibility)

    # Weighted combination — weights can be tuned against IEDB validation later
    weights = {"sasa": 0.35, "plddt": 0.20, "hydrophilicity": 0.30, "flexibility": 0.15}

    scores = []
    for i in range(length):
        score = (
            weights["sasa"] * sasa_n[i] +
            weights["plddt"] * plddt_n[i] +
            weights["hydrophilicity"] * hydrophilicity_n[i] +
            weights["flexibility"] * flexibility_n[i]
        )
        scores.append(round(score, 4))
    return scores

def analyze_novel_sequence(sequence: str, sasa_threshold=30.0, plddt_threshold=0.7, min_length=5, max_gap=1) -> dict:
    """Full pipeline for a sequence with no known experimental structure."""
    predicted_pdb = predict_structure(sequence)
    sasa = compute_sasa(predicted_pdb, chain=None)
    plddt = extract_plddt(predicted_pdb, chain=None)
    length = min(len(sasa), len(plddt))
    candidates = []
    print(f"NOVEL SEQ: length={len(sequence)}")
    print(f"SASA raw values (first 20): {sasa[:20]}")
    print(f"SASA max={max(sasa) if sasa else 'empty'}, min={min(sasa) if sasa else 'empty'}")
    print(f"pLDDT raw values (first 20): {plddt[:20]}")
    for i in range(length):
            if sasa[i] > sasa_threshold and plddt[i] >= plddt_threshold:
                candidates.append({"residue_index": i + 1, "sasa": round(sasa[i], 1), "plddt": round(plddt[i], 1)})
    print(f"Raw candidates found: {len(candidates)}")
    print(f"Candidate residue indices: {[c['residue_index'] for c in candidates]}")
    linear_epitopes = group_linear_epitopes(candidates)
    glyco_sites = find_glycosylation_sites(sequence)
    epitope_scores = compute_epitope_scores(sequence, sasa, plddt)
    ranked_candidates = sorted(
        [{"residue_index": i + 1, "score": epitope_scores[i], "sasa": round(sasa[i], 1), "plddt": round(plddt[i], 2)}
         for i in range(len(epitope_scores))],
        key=lambda x: x["score"], reverse=True
    )[:15]
    for epi in linear_epitopes:
            epi["glycan_overlap"] = any(epi["start"] <= g <= epi["end"] for g in glyco_sites)
    return {
            "predicted_pdb": predicted_pdb,
            "residue_candidates": candidates,
            "linear_epitopes": linear_epitopes,
            "glycosylation_sites": glyco_sites,
            "ranked_candidates": ranked_candidates
        }

    

# ---------------------------------------------------------------------------
# Batch benchmarking (CSV export)
# ---------------------------------------------------------------------------
def run_benchmark(pdb_ids: List[str], output_csv: str = "results.csv") -> List[BenchmarkResult]:
    def _benchmark_one_safe(pdb_id: str) -> BenchmarkResult:
        try:
            result = benchmark_one(pdb_id)
        except Exception as exc:
            print(f"[{pdb_id}] FAILED: {exc}")
            result = BenchmarkResult(pdb_id, 0, float("nan"), float("nan"), error=str(exc))
        return result

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(_benchmark_one_safe, pdb_ids))

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pdb_id", "sequence_length", "rmsd_angstrom", "tm_score", "error"])
        for r in results:
            writer.writerow([r.pdb_id, r.sequence_length, r.rmsd, r.tm_score, r.error or ""])

    print(f"\nSaved {len(results)} results to {output_csv}")
    return results


# ---------------------------------------------------------------------------
# 3D visualization helpers
# ---------------------------------------------------------------------------
def show_structure(pdb_id, chain_id=None):
    view = py3Dmol.view(query=f"pdb:{pdb_id}", width=700, height=450)
    if chain_id:
        view.setStyle({"chain": chain_id}, {"cartoon": {"color": "spectrum"}})
    else:
        view.setStyle({"cartoon": {"colorscheme": "chain"}})
    view.addStyle({"hetflag": True}, {"stick": {"colorscheme": "orangeCarbon"}})
    view.zoomTo()
    st.iframe(view._make_html(), height=470)


def show_quaternary_structure(pdb_id):
    view = py3Dmol.view(query=f"pdb:{pdb_id}", width=700, height=450)
    view.setStyle({}, {"cartoon": {"colorscheme": "chain"}})
    view.addStyle({"hetflag": True}, {"stick": {"colorscheme": "orangeCarbon"}})
    view.zoomTo()
    st.iframe(view._make_html(), height=470)


# ---------------------------------------------------------------------------
# Streamlit
# ---------------------------------------------------------------------------
st.title("ESMFold Benchmarking Pipeline")


def _on_pdb_ids_change():
    pdb_ids_raw = st.session_state.get("pdb_ids_input", "")
    pdb_id_list = [p.strip() for p in pdb_ids_raw.split(",") if p.strip()]
    st.session_state["pathogen_detected_flag"] = None
    for pdb_id in pdb_id_list:
        if is_likely_pathogen(pdb_id):
            sequence = fetch_sequence(pdb_id)
            st.session_state["novel_sequence_input"] = sequence
            st.session_state["auto_triggered_pdb"] = pdb_id
            st.session_state["pathogen_detected_flag"] = pdb_id
            break


pdb_ids = st.text_input(
    "PDB IDs (comma-separated)",
    "1UBQ, 1LYZ, 6LYZ",
    key="pdb_ids_input",
    on_change=_on_pdb_ids_change,
)

if st.button("Run Benchmark"):
    results = []
    for pdb_id in [p.strip() for p in pdb_ids.split(",") if p.strip()]:
        with st.spinner(f"Processing {pdb_id}..."):
                try:
                    chain_results = benchmark_all_chains(pdb_id)
                except RuntimeError as e:
                    if "504" in str(e) or "timed out" in str(e).lower() or "ReadTimeout" in str(e):
                        st.error(f"RCSB PDB servers appear to be down or unresponsive right now for {pdb_id}.")
                    else:
                        st.error(f"Failed to benchmark {pdb_id}: {e}")
                    chain_results = []
                for r in chain_results:
                    results.append({
                        "PDB ID": pdb_id,
                        "RMSD": r.rmsd,
                        "TM-score": r.tm_score,
                        "Error": r.error or "",
                    })
    st.session_state["benchmark_results"] = results

if "benchmark_results" in st.session_state:
    st.dataframe(st.session_state["benchmark_results"])
    for pdb_id in [p.strip() for p in pdb_ids.split(",") if p.strip()]:
        st.subheader(f"{pdb_id} — Predicted vs Experimental Structure")
        show_structure(pdb_id)

st.header("Novel Pathogen Epitope Triage")

if st.session_state.get("pathogen_detected_flag"):
    st.info(
        f"Pathogen-related entry detected: '{st.session_state['pathogen_detected_flag']}' "
        "— sequence auto-loaded and analyzed below."
    )

raw_sequence = st.text_area(
    "Paste amino acid sequence (FASTA sequence only, no header)",
    key="novel_sequence_input",
)

st.subheader("Epitope Detection Settings")
col1, col2 = st.columns(2)
with col1:
    sasa_threshold = st.slider("SASA threshold (Å²)", 0.0, 100.0, 20.0, step=5.0)
with col2:
    plddt_threshold = st.slider("pLDDT threshold", 0.0, 100.0, 50.0, step=5.0)

# Recommend thresholds based on sequence length
if raw_sequence and raw_sequence.strip():
    seq_len = len(raw_sequence.strip().upper().replace(" ", "").replace("\n", ""))
    
    if seq_len < 100:
        size_label = "very small protein"
        sasa_rec = 15.0
        plddt_rec = 50.0
        note = "Very small proteins have limited surface area — use lower SASA threshold to avoid missing candidates."
    elif seq_len < 200:
        size_label = "small protein"
        sasa_rec = 20.0
        plddt_rec = 55.0
        note = "Small proteins fold compactly — moderate thresholds recommended."
    elif seq_len < 400:
        size_label = "medium protein"
        sasa_rec = 25.0
        plddt_rec = 60.0
        note = "Medium proteins have a good balance of buried and exposed regions — standard thresholds work well."
    elif seq_len < 700:
        size_label = "large protein"
        sasa_rec = 30.0
        plddt_rec = 65.0
        note = "Large proteins have extensive surface — stricter thresholds reduce noise."
    else:
        size_label = "very large protein (chunked prediction)"
        sasa_rec = 30.0
        plddt_rec = 60.0
        note = "Chunked prediction may introduce seam artifacts — pLDDT threshold relaxed slightly to account for boundary uncertainty."

    st.info(
        f"**Sequence length: {seq_len} residues ({size_label})**\n\n"
        f"Recommended thresholds → SASA: {sasa_rec} Å², pLDDT: {plddt_rec}\n\n"
        f"{note}"
    )

    col1, col2 = st.columns(2)
    with col1:
        min_length_ui = st.slider("Minimum epitope region length", min_value=2, max_value=10, value=5)
    with col2:
        max_gap_ui = st.slider("Maximum gap between residues", min_value=0, max_value=5, value=1)

manual_run = st.button("Analyze Sequence")
auto_run = st.session_state.pop("auto_triggered_pdb", None) is not None

if manual_run or auto_run:
    cleaned_sequence = raw_sequence.strip().upper()
    valid_chars = set("ACDEFGHIKLMNPQRSTVWY")
    if not cleaned_sequence or not set(cleaned_sequence).issubset(valid_chars):
        st.error("Please enter a valid amino acid sequence (standard 20 amino acid letters only).")
    else:
        with st.spinner("Predicting structure and scanning for epitope candidates..."):
            analysis = analyze_novel_sequence(cleaned_sequence, min_length=min_length_ui, max_gap=max_gap_ui)
        st.session_state["novel_analysis_result"] = analysis

if "novel_analysis_result" in st.session_state:
    analysis = st.session_state["novel_analysis_result"]
    n_grouped = len(analysis["linear_epitopes"])
    n_individual = len(analysis["residue_candidates"])

    if n_grouped > 0:
        st.subheader(f"Found {n_grouped} candidate linear epitope regions "
                     f"({n_individual} individual surface-exposed, high-confidence residues)")
    elif analysis.get("ranked_candidates"):
        st.subheader(f"No contiguous epitope regions found — showing top {len(analysis['ranked_candidates'])} ranked candidate residues instead")
    else:
        st.subheader("No candidate epitope residues found")
    if len(analysis["linear_epitopes"]) == 0 and analysis.get("ranked_candidates"):
        st.info("No contiguous 5+ residue stretches passed the strict threshold. Showing top-ranked individual residues by combined epitope propensity score instead:")
        for rc in analysis["ranked_candidates"]:
            st.write(f"Residue {rc['residue_index']}: score={rc['score']} (SASA={rc['sasa']}, pLDDT={rc['plddt']})")
    for epi in analysis["linear_epitopes"]:
        flag = " ⚠️ overlaps predicted glycosylation site" if epi["glycan_overlap"] else ""
        st.write(f"Residues {epi['start']}–{epi['end']} (length {epi['length']}){flag}")

    st.subheader("Predicted Structure")
    view = py3Dmol.view(width=700, height=450)
    view.addModel(analysis["predicted_pdb"], "pdb")
    view.setStyle({"cartoon": {"color": "lightgray"}})
    for epi in analysis["linear_epitopes"]:
        color = "orange" if epi["glycan_overlap"] else "red"
        view.addStyle({"resi": f"{epi['start']}-{epi['end']}"}, {"cartoon": {"color": color}})
    if len(analysis["linear_epitopes"]) == 0 and analysis.get("ranked_candidates"):
        for rc in analysis["ranked_candidates"]:
            view.addStyle({"resi": str(rc["residue_index"])}, {"cartoon": {"color": "blue"}})
    view.zoomTo()
    st.iframe(view._make_html(), height=470)
    
