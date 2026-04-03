# UHA Archive Scrub Status

**Date:** 2025-10-24
**Archive:** 10.5281/zenodo.17322471
**Location:** `/claude/doi/extracted/17322471/hubble/`

---

## ✅ COMPLETED

### 1. API Service (Production)
- **File:** `/got/uha-api-service/app/merge.py`
- **Status:** ✅ FULLY OBFUSCATED
- **Changes:**
  - `nu_merge()` → `aggregate_pair()`
  - `nu_cumulative_merge()` → `aggregate_sequential()`
  - Removed all patent references
  - Removed "N/U algebra" terminology
  - Removed explicit formulas
  - **Live at:** https://api.aybllc.org/v1/merge

### 2. Zenodo Archive Code
- **File:** `/claude/doi/extracted/17322471/hubble/07_code/hubble_analysis.py`
- **Status:** ✅ OBFUSCATED
- **Changes:**
  - `nu_merge()` → `aggregate_pair()`
  - `nu_cumulative_merge()` → `aggregate_sequential()`
  - Removed explicit merge formula from comments
  - Changed "N/U algebra merge" → "conservative aggregation"

---

## ⏸️ PENDING - AWAITING USER CLARIFICATION

### UHA (Universal Horizon Address) References in Archive

**User requirement:** "We need to keep it in system and use the token api keys instead"

**Unclear:** What should replace UHA identifiers in the public archive?

**Current UHA usage in archive:**
- 22 references to "UHA"
- UHA identifiers like: `UHA::NGC4258::maser::J1210+4711::ICRS2016`
- Directory: `/03_uha_framework/`
- Metadata: Universal Horizon Address framework

**Options discussed but not confirmed:**
1. Replace with standard astronomical IDs (NGC4258, coordinates)
2. Replace with generic Object IDs (OID)
3. Reference API endpoint for lookups
4. Something else?

**User will return to clarify.**

---

## Archive Files Summary

**Total files:** 20
**Code files obfuscated:** 1 (hubble_analysis.py)
**Documentation files:** 3 (README.md, RESULTS_SUMMARY.md, FRAMEWORK_CLAIM.md)
**UHA references remaining:** 22 (awaiting replacement strategy)

---

## When User Returns

**Questions to ask:**
1. What should UHA identifiers be replaced with in the public archive?
2. Should the `/03_uha_framework/` directory be renamed?
3. Should UHA be removed from author_credentials.json metadata?
4. What terminology should describe the coordinate/object system?

**Ready to execute once clarified.**
