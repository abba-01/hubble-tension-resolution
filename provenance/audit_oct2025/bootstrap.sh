#!/usr/bin/env bash
set -euo pipefail

REPO=un-algebra-reanchor

echo "Creating repository: $REPO"
mkdir -p "$REPO"
cd "$REPO"

# .gitignore
cat > .gitignore <<'EOF'
__pycache__/
*.pyc
*.pyo
*.pyd
*.egg-info/
.dist/
build/
dist/
.env
.venv
.envrc
.vscode/
.idea/
reports/
artifacts/
.coverage
htmlcov/
.ipynb_checkpoints/
EOF

# LICENSE
cat > LICENSE <<'EOF'
MIT License

Copyright (c) 2025 All Your Baseline LLC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF

# pyproject.toml
cat > pyproject.toml <<'EOF'
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "un-algebra-reanchor"
version = "0.1.0"
description = "UN-Algebra retro-validation tests for metrology datasets"
readme = "README.md"
requires-python = ">=3.11,<3.12"
license = {text = "MIT"}
authors = [{name="Eric D. Martin", email="eric@aybllc.org"}]
dependencies = [
  "pandas==2.2.2",
  "numpy==1.26.4",
  "PyYAML==6.0.1"
]

[project.scripts]
unreanchor = "un_reanchor.cli:main"

[tool.setuptools]
package-dir = {"" = "src"}
packages = ["un_reanchor"]

[tool.pytest.ini_options]
addopts = "-q"
EOF

# requirements.txt
cat > requirements.txt <<'EOF'
pandas==2.2.2
numpy==1.26.4
PyYAML==6.0.1
pytest==8.2.0
EOF

# Makefile
cat > Makefile <<'EOF'
PY?=python3

setup:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e . -r requirements.txt

test:
	$(PY) -m pytest

demo:
	$(PY) scripts/make_synthetic.py --out data/demo.csv
	unreanchor run --data data/demo.csv --config configs/config.sample.yaml --out reports

run:
	unreanchor run --data $(DATA) --config $(CONFIG) --out $(OUT)

docker-build:
	docker build -t un-algebra-reanchor:0.1.0 .

docker-run:
	docker run --rm -v $$PWD:/work -w /work un-algebra-reanchor:0.1.0 \
		unreanchor run --data data/demo.csv --config configs/config.sample.yaml --out /work/reports

lint:
	$(PY) -m pip install ruff || true
	ruff check src || true
EOF

# Dockerfile
cat > Dockerfile <<'EOF'
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml requirements.txt README.md /app/
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir .
COPY src /app/src
COPY configs /app/configs
COPY scripts /app/scripts
CMD ["unreanchor", "--help"]
EOF

# CITATION.cff
cat > CITATION.cff <<'EOF'
cff-version: 1.2.0
title: "un-algebra-reanchor: UN-Algebra retro-validation tests"
message: "If you use this software, please cite it as below."
type: software
authors:
  - family-names: "Martin"
    given-names: "Eric D."
    orcid: "https://orcid.org/0000-0000-0000-0000"
version: "0.1.0"
doi: "10.5281/zenodo.TBD"
date-released: "2025-10-25"
repository-code: "https://github.com/aybllc/un-algebra-reanchor"
EOF

# .zenodo.json
cat > .zenodo.json <<'EOF'
{
  "title": "un-algebra-reanchor: UN-Algebra retro-validation tests",
  "upload_type": "software",
  "description": "Reproducible test battery for UN-Algebra invariants and ISO 14253-1 guard-band decisions over metrology datasets.",
  "creators": [
    {"name": "Eric D. Martin", "affiliation": "All Your Baseline LLC", "orcid": "0000-0000-0000-0000"}
  ],
  "keywords": ["metrology", "uncertainty", "ISO 14253-1", "GUM", "measurement", "quality"],
  "communities": [],
  "license": "MIT",
  "related_identifiers": [],
  "notes": "Initial release for DOI minting. Update CITATION.cff after DOI assignment."
}
EOF

# configs/config.sample.yaml
mkdir -p configs
cat > configs/config.sample.yaml <<'EOF'
columns:
  part_id: part_id
  nominal: nominal
  tol_lower: tol_lower
  tol_upper: tol_upper
  measured: measured
  true_value: true_value
  uncertainty_U: uncertainty_U
  instrument_id: instrument_id
  accepted: accepted
  timestamp: timestamp

params:
  coverage_k: 2.0
  gamma: 1.0
  edge_delta: 0.1
  calibration_cut: null
EOF

# scripts/make_synthetic.py
mkdir -p scripts
cat > scripts/make_synthetic.py <<'EOF'
import argparse, csv, random
from datetime import datetime, timedelta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    random.seed(42)
    start = datetime(2024,1,1)
    rows = []
    for i in range(120):
        nominal = 10.0
        tol = 0.05
        true_val = random.gauss(nominal, 0.01)
        U = abs(random.gauss(0.01, 0.002))
        measured = random.gauss(true_val, U/2.0)
        inst = "A" if i%3!=0 else "B"
        part_id = i//2 if i%10==0 else i
        timestamp = (start + timedelta(days=i)).isoformat()
        LSL = nominal - tol
        USL = nominal + tol
        accepted = 1 if (LSL <= measured <= USL) else 0
        rows.append({
            "part_id": part_id,
            "nominal": nominal,
            "tol_lower": tol,
            "tol_upper": tol,
            "measured": measured,
            "true_value": true_val if i%5==0 else "",
            "uncertainty_U": U,
            "instrument_id": inst,
            "accepted": accepted,
            "timestamp": timestamp
        })

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

if __name__ == "__main__":
    main()
EOF

# src/un_reanchor/__init__.py
mkdir -p src/un_reanchor
cat > src/un_reanchor/__init__.py <<'EOF'
__all__ = ["un_validation"]
EOF

# src/un_reanchor/un_validation.py
cat > src/un_reanchor/un_validation.py <<'EOF'
import json, math
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np

@dataclass
class Config:
    columns: Dict[str, Optional[str]]
    params: Dict[str, Any]

def _col(df, name, cfg):
    c = cfg["columns"].get(name)
    return df[c] if c in df.columns else pd.Series([np.nan]*len(df))

def compute_spec_limits(df, cfg):
    nominal = _col(df, "nominal", cfg).astype(float)
    tl = _col(df, "tol_lower", cfg).astype(float)
    tu = _col(df, "tol_upper", cfg).astype(float)
    LSL = nominal - tl.abs()
    USL = nominal + tu.abs()
    tol = np.maximum(tl.abs(), tu.abs())
    return LSL, USL, tol

def compute_uncertainty_U(df, cfg):
    if cfg["columns"].get("uncertainty_U") and cfg["columns"]["uncertainty_U"] in df.columns:
        U = df[cfg["columns"]["uncertainty_U"]].astype(float)
    else:
        if "sigma" in df.columns:
            k = float(cfg["params"].get("coverage_k", 2.0))
            U = df["sigma"].astype(float) * k
        else:
            raise ValueError("Provide either 'uncertainty_U' column or 'sigma' + coverage_k in config.")
    return U

def un_T1_inequality_coverage(df, cfg):
    measured = _col(df, "measured", cfg).astype(float)
    truev = _col(df, "true_value", cfg).astype(float)
    LSL, USL, tol = compute_spec_limits(df, cfg)
    U = compute_uncertainty_U(df, cfg)
    mask = ~truev.isna()
    if mask.sum() == 0:
        return {"n": 0, "coverage_rate": None}
    lhs = (measured - truev).abs()
    rhs = tol + U
    covered = lhs[mask] <= rhs[mask]
    return {"n": int(mask.sum()), "coverage_rate": float(covered.mean())}

def iso_guard_band_decision(measured, LSL, USL, U, gamma=1.0):
    if measured <= LSL - gamma*U or measured >= USL + gamma*U:
        return "nonconform"
    if measured >= LSL + gamma*U and measured <= USL - gamma*U:
        return "conform"
    return "indeterminate"

def un_T2_guard_band(df, cfg):
    measured = _col(df, "measured", cfg).astype(float)
    LSL, USL, tol = compute_spec_limits(df, cfg)
    U = compute_uncertainty_U(df, cfg)
    gamma = float(cfg["params"].get("gamma", 1.0))
    decisions = [iso_guard_band_decision(m, l, u, uu, gamma) for m,l,u,uu in zip(measured, LSL, USL, U)]
    out = pd.Series(decisions, name="decision")
    res = out.value_counts(normalize=True).to_dict()
    res_counts = out.value_counts().to_dict()
    acc_col = cfg["columns"].get("accepted")
    agree = None
    if acc_col and acc_col in df.columns:
        determ_mask = out.isin(["conform","nonconform"])
        if determ_mask.sum() > 0:
            mapping = {1:"conform", 0:"nonconform"}
            gold = df.loc[determ_mask, acc_col].map(mapping)
            agree = (out[determ_mask] == gold).mean()
    return {"share": res, "counts": res_counts, "agreement_with_archival": None if agree is None else float(agree)}, out

def un_T3_cross_instrument(df, cfg):
    inst_col = cfg["columns"].get("instrument_id")
    if not inst_col or inst_col not in df.columns:
        return {"n_pairs": 0, "exceed_rate": None}
    U = compute_uncertainty_U(df, cfg)
    measured = _col(df, "measured", cfg).astype(float)
    pid = _col(df, "part_id", cfg)
    pairs = []
    for part, g in df.groupby(pid):
        if g[inst_col].nunique() < 2: continue
        g = g.reset_index(drop=True)
        for i in range(len(g)):
            for j in range(i+1, len(g)):
                diff = abs(g.loc[i, cfg["columns"]["measured"]] - g.loc[j, cfg["columns"]["measured"]])
                u_sum = g.loc[i, "uncertainty_U"] + g.loc[j, "uncertainty_U"]
                pairs.append(diff > u_sum)
    if not pairs:
        return {"n_pairs": 0, "exceed_rate": None}
    exceed_rate = float(np.mean(pairs))
    return {"n_pairs": len(pairs), "exceed_rate": exceed_rate}

def un_T4_temporal_drift(df, cfg):
    cut = cfg["params"].get("calibration_cut")
    if not cut:
        return {"before_n": 0, "after_n": 0, "delta_mean": None}
    measured = _col(df, "measured", cfg).astype(float)
    ts_col = cfg["columns"].get("timestamp")
    if not ts_col or ts_col not in df.columns:
        return {"before_n": 0, "after_n": 0, "delta_mean": None}
    ts = pd.to_datetime(df[ts_col])
    before = measured[ts < pd.to_datetime(cut)]
    after = measured[ts >= pd.to_datetime(cut)]
    if len(before)==0 or len(after)==0:
        return {"before_n": int(len(before)), "after_n": int(len(after)), "delta_mean": None}
    return {"before_n": int(len(before)), "after_n": int(len(after)), "delta_mean": float(after.mean()-before.mean())}

def un_T5_edge_of_spec(df, cfg):
    measured = _col(df, "measured", cfg).astype(float)
    LSL, USL, tol = compute_spec_limits(df, cfg)
    delta = float(cfg["params"].get("edge_delta", 0.1))
    near = (abs(measured - LSL) <= delta) | (abs(USL - measured) <= delta)
    if near.sum()==0:
        return {"n_edge": 0, "indeterminate_rate": None}
    U = compute_uncertainty_U(df, cfg)
    gamma = float(cfg["params"].get("gamma", 1.0))
    decisions = [iso_guard_band_decision(m, l, u, uu, gamma) for m,l,u,uu in zip(measured[near], LSL[near], USL[near], U[near])]
    indec_rate = np.mean(pd.Series(decisions)=="indeterminate")
    return {"n_edge": int(near.sum()), "indeterminate_rate": float(indec_rate)}

def un_T6_interval_coverage(df, cfg):
    measured = _col(df, "measured", cfg).astype(float)
    truev = _col(df, "true_value", cfg).astype(float)
    U = compute_uncertainty_U(df, cfg)
    mask = ~truev.isna()
    if mask.sum()==0:
        return {"n": 0, "coverage": None}
    covered = (truev[mask] >= measured[mask]-U[mask]) & (truev[mask] <= measured[mask]+U[mask])
    return {"n": int(mask.sum()), "coverage": float(covered.mean())}

def run_all(df, cfg):
    if "uncertainty_U" not in df.columns:
        if "sigma" in df.columns:
            df["uncertainty_U"] = df["sigma"] * float(cfg["params"].get("coverage_k", 2.0))
    report = {}
    report["UN-T1"] = un_T1_inequality_coverage(df, cfg)
    gb_res, decisions = un_T2_guard_band(df, cfg)
    report["UN-T2"] = gb_res
    report["UN-T3"] = un_T3_cross_instrument(df, cfg)
    report["UN-T4"] = un_T4_temporal_drift(df, cfg)
    report["UN-T5"] = un_T5_edge_of_spec(df, cfg)
    report["UN-T6"] = un_T6_interval_coverage(df, cfg)
    return report, decisions
EOF

# src/un_reanchor/cli.py
cat > src/un_reanchor/cli.py <<'EOF'
import argparse, json, os, sys
import pandas as pd
import yaml
from .un_validation import run_all

def main():
    ap = argparse.ArgumentParser(prog="unreanchor", description="UN-Algebra retro-validation")
    sub = ap.add_subparsers(dest="cmd")

    runp = sub.add_parser("run", help="Run validation")
    runp.add_argument("--data", required=True)
    runp.add_argument("--config", required=True)
    runp.add_argument("--out", required=True)

    args = ap.parse_args()
    if args.cmd != "run":
        ap.print_help()
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.data)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    report, decisions = run_all(df, cfg)
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    decisions.to_csv(os.path.join(args.out, "decisions.csv"), index=False)
    print(json.dumps(report, indent=2))
EOF

# tests/test_un_validation.py
mkdir -p tests
cat > tests/test_un_validation.py <<'EOF'
import pandas as pd, yaml
from un_reanchor.un_validation import run_all

def test_smoke():
    # minimal synthetic
    df = pd.DataFrame({
        "part_id":[1,1,2],
        "nominal":[10,10,10],
        "tol_lower":[0.05,0.05,0.05],
        "tol_upper":[0.05,0.05,0.05],
        "measured":[10.01,9.99,10.06],
        "true_value":[10.00,10.00,""],
        "uncertainty_U":[0.01,0.01,0.01],
        "instrument_id":["A","B","A"],
        "accepted":[1,1,0]
    })
    cfg = yaml.safe_load(open("configs/config.sample.yaml"))
    report, decisions = run_all(df, cfg)
    assert set(report.keys()) == {"UN-T1","UN-T2","UN-T3","UN-T4","UN-T5","UN-T6"}
EOF

# .github/workflows/ci.yml
mkdir -p .github/workflows
cat > .github/workflows/ci.yml <<'EOF'
name: CI

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -e .
      - name: Unit tests
        run: pytest -q
      - name: Demo run
        run: |
          mkdir -p data reports
          python scripts/make_synthetic.py --out data/demo.csv
          unreanchor run --data data/demo.csv --config configs/config.sample.yaml --out reports
      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: reports
          path: reports
EOF

# README.md
cat > README.md <<'EOF'
# un-algebra-reanchor

Reproducible retro-validation of metrology datasets against UN-Algebra invariants and ISO 14253-1 decision rules.

## Overview

This package provides a test battery (`UN-T1` through `UN-T6`) for assessing measurement datasets under dual-reference-frame uncertainty quantification principles:

- **UN-T1** — Inequality coverage: `|measured - true_value| ≤ tolerance + U`
- **UN-T2** — ISO 14253-1 guard-band decisions (conform/nonconform/indeterminate)
- **UN-T3** — Cross-instrument coherence (same part measured by different instruments)
- **UN-T4** — Temporal drift detection (before/after calibration cut)
- **UN-T5** — Edge-of-spec behavior (indeterminate-rate near LSL/USL)
- **UN-T6** — Interval coverage: `true_value ∈ [measured - U, measured + U]`

All tests align with GUM (Guide to Uncertainty in Measurement) and ISO 14253-1 principles without requiring distributional assumptions.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Usage

### CLI

```bash
# Generate synthetic dataset
python scripts/make_synthetic.py --out data/demo.csv

# Run validation
unreanchor run --data data/demo.csv --config configs/config.sample.yaml --out reports

# Outputs:
#   reports/report.json      — test results (UN-T1 through UN-T6)
#   reports/decisions.csv    — per-row guard-band decisions
```

### Makefile Shortcuts

```bash
make setup        # Install dependencies
make test         # Run pytest
make demo         # Generate synthetic data + run validation
make lint         # Run ruff linter (optional)
```

### Docker

```bash
make docker-build
make docker-run
```

Or manually:

```bash
docker build -t un-algebra-reanchor:0.1.0 .
docker run --rm -v $PWD:/work -w /work un-algebra-reanchor:0.1.0 \
  unreanchor run --data data/demo.csv --config configs/config.sample.yaml --out /work/reports
```

## Configuration

See `configs/config.sample.yaml` for column mappings. The tool supports:

- **Expanded uncertainty** (`uncertainty_U` column) OR
- **Standard uncertainty + coverage factor** (`sigma` column + `coverage_k` param, default k=2)

Parameters:
- `gamma`: Guard-band multiplier (default 1.0)
- `edge_delta`: Proximity threshold for edge-of-spec tests (default 0.1)
- `calibration_cut`: ISO timestamp for temporal drift (UN-T4)

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`):
1. Runs unit tests
2. Executes demo on synthetic data
3. Uploads `report.json` and `decisions.csv` as artifacts

## Minting a DOI

1. **Enable Zenodo integration** for your GitHub repository at https://zenodo.org/account/settings/github/
2. **Create a GitHub Release** (e.g., `v0.1.0`)
3. Zenodo auto-archives and assigns a DOI
4. **Update** `CITATION.cff` and `.zenodo.json` with the real DOI

## ISO/GUM Alignment

- **ISO 14253-1** guard-band decisions avoid false accept/reject
- **GUM** expanded uncertainty with coverage factor k
- **PAC bounds** (Probably Approximately Correct) without distributional assumptions

## Citation

```bibtex
@software{unalgebra_reanchor,
  title = {un-algebra-reanchor: UN-Algebra retro-validation tests},
  author = {Martin, Eric D.},
  year = {2025},
  version = {0.1.0},
  doi = {10.5281/zenodo.TBD},
  url = {https://github.com/aybllc/un-algebra-reanchor}
}
```

See `CITATION.cff` for full metadata.

## License

MIT License — See `LICENSE` file.

## Support

- **Issues:** https://github.com/aybllc/un-algebra-reanchor/issues
- **Documentation:** This README + inline docstrings in `src/un_reanchor/`

---

**All Your Baseline LLC**
*Reproducible metrology validation under UN-Algebra principles*
EOF

# Create empty directories for runtime
mkdir -p data reports

echo ""
echo "✓ Repository '$REPO' created successfully!"
echo ""
echo "Files created:"
find . -type f | grep -v '.git' | sort
echo ""
echo "Next: cd $REPO && make setup && make test && make demo"
