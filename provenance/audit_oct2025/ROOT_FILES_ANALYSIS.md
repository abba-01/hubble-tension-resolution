# /got Root Files Analysis & Organization Plan
**Date:** 2025-10-24
**Analyst:** Claude Code
**Purpose:** Categorize, assess security, and recommend organization for all root-level files

---

## FILE INVENTORY & CATEGORIZATION

### Category 1: ZENODO DOI Documentation (4 files)
**Purpose:** Document and analyze published Zenodo DOI packages

| File | Size | Content | Security Risk |
|------|------|---------|---------------|
| `ZENODO_DISCOVERY_PATHWAY.md` | 20K | Maps 9 Zenodo DOIs to research pathways | ✅ **SAFE** - Public DOIs |
| `ZENODO_DOI_INVENTORY.md` | 12K | Inventory of all Zenodo packages | ✅ **SAFE** - Public records |
| `ZENODO_EXECUTIVE_SUMMARY.md` | 13K | Executive summary of DOI landscape | ✅ **SAFE** - Summary only |
| `ZENODO_UHA_ASSESSMENT.md` | 14K | Assessment of UHA-related DOIs | ✅ **SAFE** - Public materials |

**Recommendation:** → Move to `/got/zenodo_downloads/documentation/`  
**Rationale:** Keeps Zenodo materials together

---

### Category 2: Research Analysis & Planning (5 files)
**Purpose:** Analysis documents and paper planning

| File | Size | Content | Security Risk |
|------|------|---------|---------------|
| `COMPLETE_MATERIALS_MAP.md` | 13K | Map of all research materials | ⚠️ **CHECK** - May reference private work |
| `CONVERGENT_VALIDATION_ANALYSIS.md` | 18K | Validation analysis across methods | ⚠️ **CHECK** - May contain formulas |
| `PACKAGE_RELATIONSHIP_ANALYSIS.md` | 14K | How packages relate to each other | ✅ **SAFE** - Structural analysis |
| `PAPER3_ABSTRACT.md` | 8.6K | Draft abstract for major paper | ✅ **SAFE** - High-level only |
| `PAPER3_STRUCTURE_PLAN.md` | 16K | Publication strategy for major journal | ✅ **SAFE** - Strategy document |

**Recommendation:** → Move to `/got/papers/planning/`  
**Rationale:** Groups paper planning materials

---

### Category 3: Hubble Tension Analysis (2 files)
**Purpose:** Specific Hubble tension research documents

| File | Size | Content | Security Risk |
|------|------|---------|---------------|
| `HUBBLE_TENSION_CONVERGENCE.md` | 16K | Convergence analysis | ⚠️ **CHECK** - May contain methods |
| `HUBBLE_TENSION_DATA_AUDIT.md` | 15K | Data audit for Hubble work | ✅ **SAFE** - Data validation |

**Recommendation:** → Move to `/got/hubble/documentation/`  
**Rationale:** Related to hubble repository work

---

### Category 4: Repository Management (2 files)
**Purpose:** Repository audits and progress tracking

| File | Size | Content | Security Risk |
|------|------|---------|---------------|
| `PATENT_PROTECTION_AUDIT.md` | 14K | Patent protection audit (Oct 22) | ✅ **SAFE** - Audit report |
| `REPOSITORY_PROGRESS_REPORT.md` | 14K | Overall progress report | ✅ **SAFE** - Status tracking |

**Recommendation:** → Move to `/got/meta/reports/`  
**Rationale:** Meta-level project management

---

### Category 5: Security Reports (1 file - CURRENT)
**Purpose:** Security audit results

| File | Size | Content | Security Risk |
|------|------|---------|---------------|
| `SECURITY_AUDIT_FINAL_REPORT.md` | 9.2K | Final security audit (TODAY) | ✅ **SAFE** - Security documentation |

**Recommendation:** → Move to `/got/meta/security/`  
**Rationale:** Important security documentation

---

## SECURITY RISK ASSESSMENT

### Files Requiring Detailed Review:

#### 1. COMPLETE_MATERIALS_MAP.md
**Risk Level:** ⚠️ **MEDIUM** - Need to check
**Reason:** May reference private implementation work
**Action:** Quick scan for formulas/algorithms

#### 2. CONVERGENT_VALIDATION_ANALYSIS.md
**Risk Level:** ⚠️ **MEDIUM** - Need to check
**Reason:** "Validation analysis" may include implementation details
**Action:** Scan for ObserverTensor, epistemic formulas

#### 3. HUBBLE_TENSION_CONVERGENCE.md
**Risk Level:** ⚠️ **MEDIUM** - Need to check  
**Reason:** "Convergence" may describe methods
**Action:** Scan for merge algorithms, formulas

### Initial Quick Scan Results:

```bash
# Checking for critical implementation terms...
grep -l "ObserverTensor\|zero_t.*=.*redshift\|epistemic.*formula" /got/*.md
```

---

## RECOMMENDED DIRECTORY STRUCTURE

```
/got/
├── meta/                          # Project management
│   ├── reports/
│   │   ├── PATENT_PROTECTION_AUDIT.md
│   │   └── REPOSITORY_PROGRESS_REPORT.md
│   └── security/
│       └── SECURITY_AUDIT_FINAL_REPORT.md
│
├── papers/                        # Paper planning
│   └── planning/
│       ├── PAPER3_ABSTRACT.md
│       ├── PAPER3_STRUCTURE_PLAN.md
│       └── PACKAGE_RELATIONSHIP_ANALYSIS.md
│
├── analysis/                      # Research analysis
│   ├── COMPLETE_MATERIALS_MAP.md
│   ├── CONVERGENT_VALIDATION_ANALYSIS.md
│   ├── HUBBLE_TENSION_CONVERGENCE.md
│   └── HUBBLE_TENSION_DATA_AUDIT.md
│
├── zenodo_downloads/              # Zenodo DOI materials
│   └── documentation/
│       ├── ZENODO_DISCOVERY_PATHWAY.md
│       ├── ZENODO_DOI_INVENTORY.md
│       ├── ZENODO_EXECUTIVE_SUMMARY.md
│       └── ZENODO_UHA_ASSESSMENT.md
│
└── [existing repo directories...]
```

---

## NEXT STEPS

### 1. Security Scan (Immediate)
```bash
# Scan the 3 potentially risky files
for file in COMPLETE_MATERIALS_MAP.md CONVERGENT_VALIDATION_ANALYSIS.md HUBBLE_TENSION_CONVERGENCE.md; do
    echo "=== $file ==="
    grep -i "ObserverTensor\|zero_t.*redshift\|epistemic.*abs\|merge.*algorithm" "$file" || echo "CLEAN"
done
```

### 2. Create Directory Structure
```bash
mkdir -p meta/{reports,security}
mkdir -p papers/planning
mkdir -p analysis
mkdir -p zenodo_downloads/documentation
```

### 3. Move Files
```bash
# Meta files
mv PATENT_PROTECTION_AUDIT.md meta/reports/
mv REPOSITORY_PROGRESS_REPORT.md meta/reports/
mv SECURITY_AUDIT_FINAL_REPORT.md meta/security/

# Paper planning
mv PAPER3_ABSTRACT.md papers/planning/
mv PAPER3_STRUCTURE_PLAN.md papers/planning/
mv PACKAGE_RELATIONSHIP_ANALYSIS.md papers/planning/

# Analysis documents
mv COMPLETE_MATERIALS_MAP.md analysis/
mv CONVERGENT_VALIDATION_ANALYSIS.md analysis/
mv HUBBLE_TENSION_CONVERGENCE.md analysis/
mv HUBBLE_TENSION_DATA_AUDIT.md analysis/

# Zenodo documentation
mv ZENODO_*.md zenodo_downloads/documentation/
```

---

## RATIONALE FOR ORGANIZATION

### Why These Categories?

1. **`meta/`** - Project management overhead, not research content
2. **`papers/`** - Publication-related materials
3. **`analysis/`** - Research analysis documents
4. **`zenodo_downloads/`** - Keeps all Zenodo materials together

### Benefits:
- ✅ Cleaner root directory
- ✅ Logical grouping by purpose
- ✅ Easier to find documents
- ✅ Clear separation of concerns
- ✅ Better for git history

---

## PRELIMINARY SECURITY ASSESSMENT

Based on file names and initial review:

**LOW RISK** (11 files):
- All Zenodo documentation (public DOIs)
- Paper planning documents (strategy only)
- Patent audit (audit report)
- Repository progress (tracking)
- Security report (our work today)

**MEDIUM RISK** (3 files - Need detailed scan):
- COMPLETE_MATERIALS_MAP.md
- CONVERGENT_VALIDATION_ANALYSIS.md
- HUBBLE_TENSION_CONVERGENCE.md

**Recommended:** Run detailed security scan on medium-risk files before finalizing organization.

---

## EXECUTION PLAN

**Option A: Conservative (Recommended)**
1. Run security scan on 3 medium-risk files
2. Review any findings
3. Clean if necessary
4. Create directory structure
5. Move files

**Option B: Quick**
1. Create directory structure
2. Move files immediately
3. Scan in-place

**Estimated Time:** 10-15 minutes for Option A

