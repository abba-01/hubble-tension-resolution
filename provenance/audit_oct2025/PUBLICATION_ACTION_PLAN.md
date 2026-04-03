# Publication Action Plan - Ready to Execute
**Date:** 2025-10-24
**Status:** ✅ ALL REPOSITORIES SECURED AND READY
**Next Step:** Execute commands below

---

## ✅ VERIFICATION COMPLETE

**Security Scan Results:**
- 19 repositories scanned
- 0 security issues found
- 0 exposed formulas
- 0 exposed credentials

**Repository Status:**
- Currently PUBLIC: 5 repositories
- Currently PRIVATE: 14 repositories
- Ready to publish: **ALL 14 private repositories**

**Critical Finding Addressed:**
- ✅ CHRONOLOGICAL_PROGRESSION removed from uha_blackbox
- ✅ CHRONOLOGICAL_PROGRESSION removed from /claude/doi
- ✅ Both backed up to ~/private_backup/chronological_progression/

---

## 🚀 READY TO EXECUTE - NO BLOCKERS

All repositories are clean and ready. You can proceed with publication immediately.

---

## PHASE 1: PUBLISH CORE REPOSITORIES (DO NOW)

### Step 1: Make uha-blackbox Public + Get DOI

**Repository:** `/got/uha_blackbox`
**Current Status:** 🔒 Private, ✅ Secured, ✅ Verified
**GitHub Remote:** `abba-01/uha-blackbox` (already renamed to kebab-case)

**Commands:**
```bash
cd /got/uha_blackbox

# 1. Make repository public
gh repo edit abba-01/uha-blackbox --visibility public

# 2. Verify it's public
gh repo view abba-01/uha-blackbox --json visibility

# 3. Create release tag
git tag -a v1.0.0 -m "UHA Blackbox v1.0.0 - Binary Distribution

First public release of Universal Hubble Aggregation blackbox tool.

Features:
- Binary-only distribution (proprietary algorithms protected)
- Template-based configuration system
- Conservative uncertainty propagation framework
- TLV binary format for data serialization
- Cosmological parameter support

This release provides a secure, citable tool for uncertainty
quantification in cosmological measurements without exposing
implementation details.

Patent: US 63/902,536 (Patent Pending)
Owner: All Your Baseline LLC
Author: Eric D. Martin (ORCID: 0009-0006-5944-1742)
"

# 4. Push tag to GitHub
git push origin v1.0.0

# 5. Create GitHub release
gh release create v1.0.0 \
    --title "UHA Blackbox v1.0.0" \
    --notes "**First Public Release**

Binary-only distribution of the Universal Hubble Aggregation (UHA) tool for conservative uncertainty propagation in cosmological measurements.

## Features
- 🔒 Binary-only distribution (implementation protected)
- ⚙️ Template-based configuration
- 📊 Conservative uncertainty propagation
- 🔢 TLV binary format
- 🌌 Cosmological parameter support

## Installation
\`\`\`bash
pip install uha-official>=1.0.0
\`\`\`

## Quick Start
See [QUICKSTART.md](https://github.com/abba-01/uha-blackbox/blob/master/QUICKSTART.md)

## Citation
DOI will be automatically assigned by Zenodo.

## Patent
US Provisional Patent 63/902,536 (Patent Pending)

## License
Proprietary - All Your Baseline LLC

## Author
Eric D. Martin (eric.martin1@wsu.edu)
ORCID: 0009-0006-5944-1742
" \
    --latest

# 6. Verify release created
gh release view v1.0.0
```

**Expected Result:**
- Repository is now public at https://github.com/abba-01/uha-blackbox
- Release v1.0.0 created
- Tag v1.0.0 visible

---

### Step 2: Enable Zenodo Integration for uha-blackbox

**Manual Steps (Browser):**

1. Go to https://zenodo.org/account/settings/github/
2. Sign in with GitHub if not already
3. Find `abba-01/uha-blackbox` in the list
4. Toggle switch to **ON**
5. Zenodo will automatically:
   - Archive the v1.0.0 release
   - Assign a DOI (e.g., 10.5281/zenodo.XXXXXXX)
   - Create citation page

**Verification:**
- Check Zenodo dashboard: https://zenodo.org/deposit
- Look for uha-blackbox v1.0.0
- Note the assigned DOI

---

### Step 3: Add DOI Badge to uha-blackbox README

**After Zenodo assigns DOI:**

```bash
cd /got/uha_blackbox

# Edit README.md to add DOI badge at the top
# Add this line after the title:
# [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

# Example edit (replace XXXXXXX with actual DOI):
cat > /tmp/add_doi.txt << 'EOF'
# UHA Official - Single Source of Truth (SSOT)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

**Universal Horizon Address - Official Implementation**
EOF

# Then manually edit README.md or use Edit tool
```

**Commit and push:**
```bash
git add README.md
git commit -m "Add Zenodo DOI badge"
git push origin master
```

---

### Step 4: Make hubble-tension-resolution Public

**Repository:** `/got/hubble`
**Current Status:** 🔒 Private, ✅ Secured (Oct 24), ✅ Verified
**GitHub Remote:** `abba-01/hubble-tension-resolution` (already renamed)

**Commands:**
```bash
cd /got/hubble

# 1. Make repository public
gh repo edit abba-01/hubble-tension-resolution --visibility public

# 2. Verify it's public
gh repo view abba-01/hubble-tension-resolution --json visibility

# 3. Optionally create release
gh release create v1.0.0 \
    --title "Hubble Tension Resolution v1.0.0" \
    --notes "Public release of cleaned Hubble tension resolution repository.

**Note:** Implementation details have been removed for patent protection.
This repository provides data, validation results, and links to published
DOI packages.

See SECURITY_NOTICE.md for details on removed content.

Patent: US 63/902,536 (Patent Pending)
" \
    --latest
```

**Expected Result:**
- Repository is now public at https://github.com/abba-01/hubble-tension-resolution

---

## PHASE 2: STANDARDIZE NAMING (NEXT STEP)

### Rename Repositories to kebab-case

**Priority: High (These are already public)**

#### HubbleBubble → hubble-bubble

```bash
cd /got/HubbleBubble

# Rename on GitHub
gh repo rename hubble-bubble

# Update local remote
git remote set-url origin git@github.com:abba-01/hubble-bubble.git

# Verify
git remote -v
gh repo view abba-01/hubble-bubble
```

#### nualgebra → nu-algebra

```bash
cd /got/nualgebra

# Rename on GitHub
gh repo rename nu-algebra

# Update local remote
git remote set-url origin git@github.com:abba-01/nu-algebra.git

# Verify
git remote -v
```

---

**Priority: Medium (Currently private, will be public)**

#### nu_anthropology → nu-anthropology

```bash
cd /got/nualgebra_anthropology

# Rename on GitHub
gh repo rename nu-anthropology

# Update local remote
git remote set-url origin git@github.com:abba-01/nu-anthropology.git

# Make public
gh repo edit abba-01/nu-anthropology --visibility public
```

#### nua_psychology → nu-psychology

```bash
cd /got/nualgebra_psychology

# Rename on GitHub
gh repo rename nu-psychology

# Update local remote
git remote set-url origin git@github.com:abba-01/nu-psychology.git

# Make public
gh repo edit abba-01/nu-psychology --visibility public
```

---

**Priority: Low (Optional)**

#### uso → universal-system-origin

```bash
cd /got/uso

# Rename on GitHub (optional - "uso" as acronym is acceptable)
gh repo rename universal-system-origin

# Update local remote
git remote set-url origin git@github.com:abba-01/universal-system-origin.git

# Make public (optional)
gh repo edit abba-01/universal-system-origin --visibility public
```

#### uha_hubble → uha-hubble

```bash
cd /got/uha_hubble

# Rename on GitHub
gh repo rename uha-hubble

# Update local remote
git remote set-url origin git@github.com:abba-01/uha-hubble.git
```

---

## PHASE 3: ENABLE ZENODO FOR ALL PUBLIC REPOS

**After Phase 1 and 2 complete:**

1. Go to https://zenodo.org/account/settings/github/
2. Enable Zenodo integration for these repositories:
   - ✅ uha-blackbox (already done in Phase 1)
   - ✅ hubble-tension-resolution
   - ✅ hubble-bubble
   - ✅ nu-algebra
   - ✅ nu-anthropology
   - ✅ nu-psychology
3. Create releases for each
4. Zenodo will assign DOIs
5. Add DOI badges to READMEs

---

## OPTIONAL: PUBLISH SUPPORTING REPOSITORIES

**These can be published at your discretion:**

- abba
- aiwared.org
- count2infinity
- Eric-D-Martin
- theories
- uha
- uha-hubble
- ebios

**Command template:**
```bash
cd /got/[REPOSITORY]
gh repo edit abba-01/[REPO-NAME] --visibility public
```

---

## NOT RECOMMENDED: hubble-tensor

**Repository:** `hubble-tensor`
**Reason:** Contains patent application materials
**Recommendation:** Consult IP attorney before publishing

Patents are public by definition once filed, but you may want to:
- Wait until patent prosecution is complete
- Keep attorney work product private
- Selectively publish patent claims only

---

## VERIFICATION CHECKLIST

After executing Phase 1:

### uha-blackbox
- [ ] Repository is public (check https://github.com/abba-01/uha-blackbox)
- [ ] Release v1.0.0 exists
- [ ] Tag v1.0.0 exists
- [ ] Zenodo integration enabled
- [ ] DOI assigned by Zenodo
- [ ] DOI badge added to README
- [ ] Repository has LICENSE file
- [ ] Repository has CITATION.cff file (optional)

### hubble-tension-resolution
- [ ] Repository is public (check https://github.com/abba-01/hubble-tension-resolution)
- [ ] SECURITY_NOTICE.md is visible
- [ ] Implementation files are removed
- [ ] Release v1.0.0 exists (optional)

---

## ROLLBACK PLAN

If you need to revert:

### Make Repository Private Again
```bash
gh repo edit abba-01/[REPO-NAME] --visibility private
```

### Delete Release
```bash
gh release delete v1.0.0 --yes
git push origin :refs/tags/v1.0.0
git tag -d v1.0.0
```

**Note:** Zenodo DOIs are PERMANENT and cannot be deleted. You can only:
- Create new versions
- Add errata
- Deprecate old versions

---

## IMPORTANT REMINDERS

### Before Publishing Any Repository:

1. ✅ **Security scan passed** (already done - all clean)
2. ✅ **Git history cleaned** (already done for uha_blackbox, hubble)
3. ✅ **No credentials exposed** (verified)
4. ✅ **No proprietary formulas** (verified)
5. ⚠️ **IP attorney consulted** (recommended before hubble-tensor)

### After Publishing:

1. Add DOI badge to README
2. Add LICENSE file if missing
3. Add CITATION.cff file
4. Update documentation
5. Announce on appropriate channels (Twitter, mailing lists, etc.)

---

## ESTIMATED TIMELINE

**Phase 1 (Core Repos):** 1-2 hours
- uha-blackbox: 30 min (including Zenodo setup)
- hubble-tension-resolution: 15 min

**Phase 2 (Renaming):** 30 min
- 4-6 repos to rename

**Phase 3 (Zenodo Integration):** 1 hour
- Enable for multiple repos
- Create releases
- Get DOIs
- Update READMEs

**Total:** 2-3 hours for complete publication

---

## NEXT IMMEDIATE ACTION

**Execute this command:**
```bash
cd /got/uha_blackbox
gh repo edit abba-01/uha-blackbox --visibility public
```

**Then proceed with remaining commands in Phase 1.**

---

## SUPPORT DOCUMENTS

Created today for your reference:

1. `/got/SECURITY_VERIFICATION_REPORT_2025-10-24.md` - Complete scan results
2. `/got/PUBLIC_REPOSITORY_STRATEGY.md` - Comprehensive strategy guide
3. `/got/PUBLICATION_ACTION_PLAN.md` - This document
4. `/claude/doi/DOI_DIRECTORY_ANALYSIS.md` - DOI package analysis
5. `/claude/doi/DOI_INVENTORY.md` - Package inventory

---

**STATUS:** ✅ READY TO EXECUTE
**BLOCKERS:** None
**RISK:** Low (all repos verified clean)
**RECOMMENDATION:** Proceed with Phase 1 immediately

---

**Last Updated:** 2025-10-24 06:45 PDT
