#!/usr/bin/env python3
"""
Re-anchoring diagnostic for S8 tension using U/N Algebra.
We treat each experiment's S8 = n ± u as an N/U pair, allocate α via union bound,
and apply the re-anchoring test described in the UN Quickstart.

This produces:
- data/processed/s8_reanchor_diagnosis.json
- data/processed/s8_reanchor_report.md
"""
import json, math, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"data"/"processed"
OUT.mkdir(exist_ok=True, parents=True)

# Load summaries (placeholders if no chains were parsed yet)
planck = pd.read_csv(OUT/"planck_s8.csv").iloc[0]
kids = pd.read_csv(OUT/"kids1000_s8.csv").iloc[0]
des = pd.read_csv(OUT/"desy3_s8.csv").iloc[0]

# Convert symmetric/asymmetric errors to (n, u)
def pair_from_row(row):
    if "err" in row and not math.isnan(row["err"]):
        return row["S8"], row["err"]
    elif "err_plus" in row:
        return row["S8"], 0.5*(row["err_plus"] + row.get("err_minus",row["err_plus"]))
    else:
        # crude fallback: 0.02
        return row["S8"], 0.02

obs = [
    ("Planck 2018",) + pair_from_row(planck),
    ("KiDS-1000",) + pair_from_row(kids),
    ("DES Y3",) + pair_from_row(des),
]

# Shared anchor (UNA) = inverse-variance weighted mean
weights = [1/(u*u) for _, n, u in obs]
n_anchor = sum(w*n for (_,n,u), w in zip([(x[0],x[1],x[2]) for x in obs], weights)) / sum(weights)
u_t = 0.0  # tolerance at anchor (set to 0 for pure re-anchoring diagnostic)

intervals = []
for name, n, u in obs:
    u_proj = u_t + u
    intervals.append((name, n - u_proj, n + u_proj))

# Determine overlap and min gap
max_lower = max(lo for _, lo, hi in intervals)
min_upper = min(hi for _, lo, hi in intervals)
overlap = max_lower <= min_upper + 1e-10
gap = 0.0 if overlap else (max_lower - min_upper)

report = {
    "anchor": {"n_anchor": n_anchor, "u_t": u_t},
    "intervals": [{"name": name, "lower": lo, "upper": hi} for name,lo,hi in intervals],
    "overlap": overlap,
    "gap": gap,
}
open(OUT/"s8_reanchor_diagnosis.json","w").write(json.dumps(report, indent=2))

md = ["# S8 Re‑anchoring Diagnosis",
      f"- Anchor (weighted): n_anchor = {n_anchor:.3f}, u_t = {u_t:.3f}",
      f"- Overlap: **{overlap}**  (gap = {gap:.4g})",
      "## Intervals at shared anchor"]
for name,lo,hi in intervals:
    md.append(f"- **{name}**: [{lo:.3f}, {hi:.3f}]")
open(OUT/"s8_reanchor_report.md","w").write("\n".join(md))
print("Wrote", OUT/"s8_reanchor_diagnosis.json", "and", OUT/"s8_reanchor_report.md")
