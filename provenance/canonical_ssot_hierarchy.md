# Canonical SSOT Hierarchy

## Purpose
This document defines which uploaded package is authoritative for which claim in the Hubble-tension project tree. It is a control layer above the package set, not a replacement for any package.

## Governing rule
When two packages make different-strength claims, the higher claim does **not** override the lower one unless it also preserves or exceeds that package's evidence tier.

---

## 1. Canonical package roles

### A. Core SSOT / canonical narrative
**Authoritative package:** `hubble-97pct-observer-tensor-v1.0.1`

**Why**
- Contains the SSOT itself (`ssot_full_solution.md`)
- Most direct match to the project's declared canonical story
- Cleanest package to cite when referring to the internal master framework state

**Use for**
- Canonical four-phase storyline
- 97.2% reduction framing
- Master architecture: N/U + observer tensors + UHA

**Do not use for**
- Strongest reproducibility defense
- Strongest simulated-MC calibration claim
- Best real-data multiresolution status

---

### B. Conservative empirical branch
**Authoritative packages:**
1. `hubble-91pct-concordance-v1.0.1`
2. `Martin_EDM_2025_Conservative_Uncertainty_Hubble_Tension_Complete_Package_v1.0.0`

**Why**
- Most cautious interpretation branch
- Preserves stronger scientific restraint
- Important for reviewer-safe language and earlier evidence-tier grounding

**Use for**
- Conservative claim baseline
- Published-data-only style framing
- Evidence that the framework evolved through a more cautious stage

**Do not use for**
- Final canonical SSOT identity
- Best defense of the 68.5/0.966 sigma style result

---

### C. Internal guardrail / honesty check
**Authoritative package:** `hubble-tension-resolution-v1.0.1`
**Mandatory file:** `CLARIFICATION.md`

**Why**
- Best internal separation between:
  - validated result
  - proof-of-concept extraction
  - projected capability
- Most important cross-check against overclaiming

**Use for**
- Claim moderation
- Distinguishing current evidence from projected capability
- Internal policy on what is actually demonstrated

**Rule**
Any future SSOT should inherit the distinctions in `CLARIFICATION.md` unless later packages explicitly and credibly retire them with stronger evidence.

---

### D. Simulated-chain calibration branch
**Authoritative packages:**
1. `hubble-99pct-montecarlo-v1.0.1`
2. `hubble_montecarlo_package2_20251011`

**Why**
- Strongest pathway for 99%+ concordance rhetoric
- Clear methodology branch for synthetic/mock posterior calibration

**Use for**
- Simulation-calibrated tensor extraction
- Synthetic Monte Carlo route
- Method development and stress testing

**Do not use for**
- Raw-data external anchoring
- Claims that imply collaboration-grade chain ingestion unless separately proven

**Authority status**
- **Secondary** for empirical truth claims
- **Primary** for simulation methodology claims

---

### E. Reproducibility and adversarial validation branch
**Authoritative package:** `HubbleBubble-v1.1.1`

**Why**
- Strongest package on procedural defense
- Best validation-gate and reproducibility layer
- Most explicit on LOAO sensitivity, leakage concerns, and test battery hardening

**Use for**
- Reproducibility claims
- Verification, execution integrity, and reviewer defense
- 68.5 / 0.966 sigma procedural defense

**Do not use for**
- Standalone proof that the underlying scientific mechanism is uniquely correct

**Authority status**
- **Primary** for reproducibility/validation process
- **Secondary** for core mechanism claims

---

### F. Multi-resolution H0 + S8 expansion branch
**Authoritative package:** `multiresolution-cosmology-d.73`

**Why**
- Best precursor to the 2026 manuscript arc
- Strongest match to the later multi-resolution UHA storyline
- Extends beyond H0 into S8 and survey-crossing language

**Use for**
- Multi-resolution resolution-dial concept
- H0 + S8 combined narrative
- KiDS/TRGB/real-data extension work
- Later-stage manuscript evolution

**Caveat**
- Contains internal version drift between simulated-complete and real-data-complete language
- Must be cited selectively, with file-specific care

**Authority status**
- **Primary** for the multi-resolution branch
- **Not automatically primary** for the whole project

---

### G. Evidence-chain repair / sterile audit branch
**Authoritative package:** `cosmo-sterile-audit-V.0.9`

**Why**
- Best route toward raw-data provenance and externally anchorable workflows
- Strategically most important for fixing the current strongest reviewer objection

**Use for**
- Provenance
- raw-data audit path
- sterile execution / traceability / fetch pipelines

**Authority status**
- **Primary** for evidence-chain repair infrastructure
- **Not** a final-results package

---

### H. Background provenance framework
**Authoritative package:** `ommp-v1.0.0`

**Why**
- Provides broader observer/provenance architecture context
- Relevant to conceptual lineage

**Use for**
- Background theory provenance
- observer/integrity context

**Do not use for**
- direct Hubble empirical proof

---

## 2. Claim authority matrix

| Claim type | Primary authority | Secondary authority | Caveat |
|---|---|---|---|
| Canonical internal project story | `hubble-97pct-observer-tensor` | `ssot_full_solution.md` root copy | This is the internal canonical branch, not automatically the strongest external-proof branch |
| Conservative reviewer-safe framing | `hubble-91pct-concordance` | `Martin_EDM_2025_Conservative...` | Better for caution than for strongest-claim marketing |
| Distinction between validated vs projected | `hubble-tension-resolution/CLARIFICATION.md` | `RESULTS_SUMMARY.md` in same package | This file should be treated as policy-level guardrail |
| 99%+ concordance via synthetic chains | `hubble-99pct-montecarlo` | `hubble_montecarlo_package2` | Synthetic-chain tier only |
| Reproducibility and gate-defense | `HubbleBubble-v1.1.1` | its RENT docs and validation docs | Strong process defense, not sole mechanism proof |
| Multi-resolution H0 mechanism | `multiresolution-cosmology-d.73` | 2026 manuscript | Must separate simulated/real validation subfiles |
| S8 extension | `multiresolution-cosmology-d.73` | later manuscript | Stronger as branch expansion than as settled universal result |
| Raw-data auditability path | `cosmo-sterile-audit-V.0.9` | HubbleBubble process docs | Infrastructure, not final concordance proof |
| UHA technical specification | patent/UHA docs | multiresolution appendix | Evidence role still varies across branches |

---

## 3. Conflict-resolution rules

### Rule 1
If the SSOT package conflicts with `CLARIFICATION.md`, cite both and treat `CLARIFICATION.md` as the stronger statement about current evidence limits.

### Rule 2
If a simulated-MC package makes a stronger empirical claim than a conservative package, do **not** let the stronger number supersede the conservative one unless the evidence tier also rises.

### Rule 3
If `HubbleBubble` strengthens procedural confidence but not evidentiary source quality, classify the gain as **validation hardening**, not automatic external confirmation.

### Rule 4
If `multiresolution-cosmology-d.73` conflicts internally across files, prefer the most specific file over the broad summary file, especially when the specific file is about real-data status.

### Rule 5
Use `cosmo-sterile-audit-V.0.9` as the designated route for upgrading externally anchored credibility. Do not treat it as already-completed proof unless it contains completed result artifacts that explicitly show that.

---

## 4. Canonical stack order

### Tier 0 — Control layer
1. `canonical_ssot_hierarchy.md` (this file)

### Tier 1 — Canonical internal truth
2. `hubble-97pct-observer-tensor-v1.0.1`
3. `ssot_full_solution.md`

### Tier 2 — Required restraint / claim control
4. `hubble-tension-resolution-v1.0.1/CLARIFICATION.md`

### Tier 3 — Empirical branches
5. `hubble-91pct-concordance-v1.0.1`
6. `Martin_EDM_2025_Conservative...`
7. `HubbleBubble-v1.1.1`
8. `multiresolution-cosmology-d.73`
9. `hubble-99pct-montecarlo-v1.0.1`
10. `hubble_montecarlo_package2_20251011`

### Tier 4 — Infrastructure and provenance
11. `cosmo-sterile-audit-V.0.9`
12. `ommp-v1.0.0`
13. UHA/patent specification materials

---

## 5. Best-practice citation guidance

### When writing an internal summary
Use:
1. `hubble-97pct-observer-tensor`
2. `CLARIFICATION.md`
3. `HubbleBubble`

### When writing a reviewer-safe paper draft
Use:
1. `hubble-91pct-concordance`
2. `Martin_EDM_2025_Conservative...`
3. `CLARIFICATION.md`
4. `cosmo-sterile-audit`

### When writing the 2026 manuscript branch
Use:
1. `multiresolution-cosmology-d.73`
2. 2026 manuscript
3. `CLARIFICATION.md`
4. `HubbleBubble`

### When discussing reproducibility
Use:
1. `HubbleBubble`
2. RENT docs
3. `cosmo-sterile-audit`

---

## 6. Single-sentence project truth
The project is best understood as a branching research program whose **canonical internal SSOT** is the 97% observer-tensor branch, whose **strongest restraint text** is `CLARIFICATION.md`, whose **best procedural defense** is `HubbleBubble`, whose **next-generation expansion** is `multiresolution-cosmology`, and whose **best route to external-anchor credibility** is `cosmo-sterile-audit`.

---

## 7. Actionable recommendation
If you want one clean master SSOT going forward:
1. Keep the 97% package as the narrative base.
2. Import `CLARIFICATION.md` as a permanent “claim-boundary” section.
3. Attach `HubbleBubble` as reproducibility appendix.
4. Attach multiresolution as future-branch appendix, not default master truth.
5. Use sterile-audit as the required path before upgrading any strong empirical claim.
