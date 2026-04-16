# Cross-Retraction Notice — 2026-04-16

**Issued by:** Phase D RECONCILIATION Decision 4 (ruling: "However broad to be right")  
**Scope:** All Hubble tension claims across the sibling repo cluster  
**Verbatim wording (accepted 2026-04-16):**

> "The 97.4% claim (hubble-tension-resolution) was retracted 2026-04-14.
> The 91% claim remains the validated baseline (published, observational data).
> Claims of 97.2%, 99.6%, and 100% (hubble-97pct, hubble-99pct/mc-package2)
> are unverified by executed computation and should not be cited as results."

---

## This Repo — hubble-tension-resolution

**Position:** Authoritative state-of-tension repo (promoted per D6).

**What stands:**
- 97.4% claim: **RETRACTED** — see CORRECTION_SUMMARY.txt:13–19
  - H₀ = 67.57 ± 0.93 km/s/Mpc (corrected; 1.84× uncertainty inflation)
  - Status: "characterized, not resolved"
- 91% claim: **VALIDATED BASELINE** — observational data, reproducible
  - Canonical framing in CLARIFICATION.md
  - Carries forward in hubble-91pct-concordance

---

## Sibling Repo Retraction Table

| repo | claim | implicated? | ruling |
|------|-------|-------------|--------|
| hubble-91pct-concordance | 91% | **Partially** — README contradicted by RESULTS_SUMMARY ("tension REAL, UNEXPLAINED"). The 91% claim carries forward as validated baseline, not full resolution. | CLARIFICATION note added |
| hubble-97pct-observer-tensor | 97.2% | **Yes** — 0 Python LOC; all claimed artifacts absent; same epistemic framework as retracted 97.4% | Reclassified: Archived |
| uha_hubble | 97.2%, 0.16σ, 7.64%, 503±50 kpc | **Yes** — 0 LOC; all results fabricated-targets in /research/ only | Reclassified: Archived |
| hubble-99pct-montecarlo (P1) | 99.6% / 99.8% | **Yes** — synthetic chains only; incompatible claim values; archived as provenance | Archived in hubble-mc-combined |
| hubble-mc-combined (P2) | 100% | **Partially** — synthetic MCMC chains, not validated against observational posteriors | CROSS_RETRACTION noted |
| hubble-bubble | 74% σ reduction | **No** — different metric (σ reduction, not gap %); observational backing | Unaffected |
| hubble-tensor | 99.8% Cepheid concordance | **No** — different metric (scalar cross-catalog concordance, computed) | Unaffected |
| pantheon-h0-omega-sensitivity | ΔH₀ = −0.22 | **No** — different domain (Pantheon+ Ωm sensitivity, live MNRAS submission) | Unaffected (frozen) |

---

## Propagation Actions (Phase D6)

- [x] CROSS_RETRACTION.md added to this repo
- [x] hubble-91pct-concordance: CLARIFICATION_RETRACTION.md added
- [x] hubble-97pct-observer-tensor: reclassified Archived (Phase D7)
- [x] uha_hubble: reclassified Archived (Phase D7)
- [x] hubble-mc-combined: RESTRUCTURE_LOG.md includes retraction context
- [ ] D1 (cosmological-data): SAID master verification pending (gate: OP01 mount)

---

## Source of Truth

- Primary retraction: CORRECTION_SUMMARY.txt:13–19 (this repo)
- Retraction ruling: `/scratch/repos/_hubble_inventory/RECONCILIATION.md` Decision 4
- Claims audit: `/scratch/repos/_hubble_inventory/PILE_claims.md`
