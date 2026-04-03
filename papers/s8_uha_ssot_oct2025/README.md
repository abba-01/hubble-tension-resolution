# S8–UHA SSOT
**Single Source of Truth (SSOT)** repository to reproducibly fetch the public datasets needed to make a **provable** claim that the *Universal Horizon Address (UHA)* + *U/N re‑anchoring* template resolves the **S₈ tension** as *referential* (frame/assumption) rather than *physical*.

## What this repo does
1. **Fetches official data products** for Planck PR3 (2018), KiDS‑1000 cosmic shear, and DES Y3 3×2pt / shear.
2. **Verifies integrity** by computing `sha256` for every artifact and storing them in `checksums/`.
3. **Builds an S₈ pipeline**:
   - Computes S₈ from Planck chains.
   - Loads Stage‑III weak lensing posterior samples or data vectors and reproduces S₈ posteriors.
   - Applies **U/N Re‑anchoring** and **UHA CosmoID** normalization to diagnose the tension (referential vs physical).
4. **Writes a machine‑readable provenance log** in `manifests/` for audit.

> This SSOT includes *only* public artifacts and is designed for deterministic reruns.

## Quick start
```bash
# 1) create a fresh environment (recommended)
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip

# 2) install deps
pip install -r requirements.txt

# 3) fetch and verify datasets
python scripts/fetch.py

# 4) compute S8 summaries
python scripts/compute_s8_from_planck.py
python scripts/ingest_kids1000.py
python scripts/ingest_desy3.py

# 5) run the re‑anchoring diagnosis (S₈ tension test)
python scripts/reanchor_s8.py
```

## Data sources (authoritative)
- **Planck PR3 Cosmology Products (DOI)**: 10.5270/esa-gb3sw1a (Likelihood code + data, parameter grid, power spectra).  
- **KiDS‑1000 cosmic shear data products**: Multinest chains + 2pt vectors, covariances, n(z) tarball.  
- **DES Y3 cosmology**: 3×2pt SACC data vector and supporting files / chains.

Exact URLs and DOIs are encoded in `SSOT.yml` and verified at fetch time.

## Claim tested
> Using UHA’s cosmology‑portable normalization and U/N re‑anchoring, the apparent Planck–lensing vs. Stage‑III weak‑lensing discrepancy in **S₈ ≡ σ₈ (Ωₘ/0.3)^{1/2}** is explained by reference‑frame/assumption differences (Δ_T), not new physics, at α=0.05.

See `scripts/reanchor_s8.py` for the diagnostic and output table.

---

### Citation
Please cite the original data providers (Planck, KiDS, DES) and the UHA/UN framework papers in your resulting work. See the bottom of this README and `SSOT.yml` for BibTeX stubs.

### License
All code here is MIT. Data retain their original licenses/terms of use.
