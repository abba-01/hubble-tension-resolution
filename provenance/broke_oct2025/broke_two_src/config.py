"""
Central configuration for all Hubble analysis scripts.
All path references should use this configuration.
"""
from pathlib import Path

# Repository root (detect from this file's location)
REPO_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = REPO_ROOT / "data"
CEPHEID_CATALOGS_DIR = DATA_DIR / "cepheid_catalogs"
SYSTEMATIC_GRID_DIR = DATA_DIR / "shoes_systematic_grid"
SYSTEMATIC_GRID_FILE = SYSTEMATIC_GRID_DIR / "J_ApJ_826_56_table3.csv"
MAST_DIR = DATA_DIR / "mast_anchors"
MCMC_METADATA_FILE = MAST_DIR / "mcmc_metadata.json"

# Results directories
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_PHASE_A = RESULTS_DIR / "phase_a"
RESULTS_PHASE_B = RESULTS_DIR / "phase_b"
RESULTS_PHASE_C = RESULTS_DIR / "phase_c"
RESULTS_PHASE_D = RESULTS_DIR / "phase_d"
RESULTS_VALIDATION = RESULTS_DIR / "validation"

# Reference files
REFERENCE_DIR = REPO_ROOT / "reference"

# SAID directories
SAID_DIR = REPO_ROOT / "SAID"
AUDIT_DIR = SAID_DIR / "audit_logs"
VERIFICATION_DIR = SAID_DIR / "verification"
PROVENANCE_DIR = SAID_DIR / "provenance"

# Create directories if they don't exist
for directory in [DATA_DIR, CEPHEID_CATALOGS_DIR, SYSTEMATIC_GRID_DIR,
                  RESULTS_DIR, RESULTS_PHASE_A, RESULTS_PHASE_B,
                  RESULTS_PHASE_C, RESULTS_PHASE_D, RESULTS_VALIDATION,
                  REFERENCE_DIR, AUDIT_DIR, VERIFICATION_DIR, PROVENANCE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Phase-specific output paths (in phase subdirectories)
PHASE_A_TENSORS = RESULTS_PHASE_A / "anchor_tensors.json"
PHASE_A_DISTANCES = RESULTS_PHASE_A / "cross_anchor_distances.json"
PHASE_B_H0 = RESULTS_PHASE_B / "phase_b_h0_from_raw_data.json"
PHASE_B_CONTRIBUTIONS = RESULTS_PHASE_B / "phase_b_per_object_contributions.csv"
PHASE_C_EIGENSPECTRUM = RESULTS_PHASE_C / "covariance_eigenspectrum.json"
PHASE_C_DECOMPOSITION = RESULTS_PHASE_C / "systematic_decomposition.json"
PHASE_C_VALIDATION = RESULTS_PHASE_C / "covariance_validation.json"
PHASE_C_COVARIANCE_NPY = RESULTS_PHASE_C / "empirical_covariance_210x210.npy"
PHASE_C_CORRELATION_NPY = RESULTS_PHASE_C / "empirical_correlation_210x210.npy"
PHASE_D_RESOLUTION = RESULTS_PHASE_D / "resolution_100pct_mcmc.json"

# Validation outputs
CONCORDANCE_EMPIRICAL = RESULTS_VALIDATION / "concordance_empirical.json"
MONTE_CARLO_COVERAGE = RESULTS_VALIDATION / "monte_carlo_coverage.json"
UN_TEST_RESULTS = RESULTS_VALIDATION / "un_test_results.json"
