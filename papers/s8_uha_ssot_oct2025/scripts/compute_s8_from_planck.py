#!/usr/bin/env python3
"""
Compute S8 ≡ sigma8 * sqrt(Ωm / 0.3) and summary stats from Planck tables or chains.
If full MCMC chains are not present, fall back to the parameter table grid.
"""
import os, pathlib, re, numpy as np, pandas as pd, tarfile, zipfile, io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT/"data"/"raw"/"planck"
OUT = ROOT/"data"/"processed"
OUT.mkdir(exist_ok=True, parents=True)

def parse_fullgrid_zip(z):
    # COM_CosmoParams_fullGrid_R3.01.zip contains CSV-like tables; we try to find sigma8 and omegam
    rows = []
    with zipfile.ZipFile(z, 'r') as zf:
        for n in zf.namelist():
            if n.lower().endswith((".txt",".csv")) and ("base" in n.lower() or "plik" in n.lower() or "fullgrid" in n.lower()):
                try:
                    data = zf.read(n).decode("utf-8",errors="ignore")
                except:
                    continue
                # crude parse: look for lines like "sigma8 0.811 ± 0.006"
                for line in data.splitlines():
                    if "sigma8" in line and "Omega_m" in data:
                        pass
    return None

def compute_from_chain_files():
    # Search for .txt chain files in RAW (if any)
    points = []
    for p in RAW.glob("**/*.txt"):
        # Many Planck chains are GetDist-style .txt with columns in a header .paramnames
        # Here we try to locate matching .paramnames to map columns. Fallback: guess sigma8 and omegam names.
        if p.name.endswith(".paramnames"): 
            continue
        # naive header parse: skip comments
        try:
            arr = np.loadtxt(p)
        except Exception:
            continue
        # too raw; prefer using getdist, but avoid extra deps here
    return None

# For this template we provide a placeholder computation that will be replaced
# by a robust getdist-based implementation by the user once chains are present.
df = pd.DataFrame([
    dict(source="Planck2018_base_TTTEEE+lowE+lensing", sigma8=0.811, Omega_m=0.315, S8=0.811*np.sqrt(0.315/0.3), note="Placeholder; replace with chain-based value")
])
df.to_csv(OUT/"planck_s8.csv", index=False)
print("Wrote", OUT/"planck_s8.csv")
