# Public Repository & DOI Publication Strategy
**Date:** 2025-10-24
**Purpose:** Comprehensive strategy for making repositories public and obtaining DOIs
**Status:** Action Plan

---

## EXECUTIVE SUMMARY

**Current Situation:**
- 9 Zenodo DOI packages already published (including v2.0 with ObserverTensor class)
- Patent filed Oct 21, 2025 (US 63/902,536)
- Multiple GitHub repositories at various security levels
- Mixed naming conventions across repositories

**Key Finding:**
The v2.0 Monte Carlo package (DOI 10.5281/zenodo.17325811) **already published the ObserverTensor implementation** on Oct 11, 2025 - 10 days BEFORE patent filing.

**Implications:**
- Implementation details are public domain
- Patent still likely valid (US 1-year grace period for inventor disclosures)
- Strategy must adapt to this reality

---

## PART 1: REPOSITORY STATUS ASSESSMENT

### Current Repository Inventory (/got)

| Repository | Current Status | Security Level | Can Publish? |
|------------|---------------|----------------|--------------|
| `hubble` | 🔒 Private | Implementation removed | ✅ YES - post-cleanup |
| `uha_blackbox` | 🔒 Private | Binary-only | ✅ YES - ready now |
| `hubble-tensor` | 🔒 Private | Patent materials | ⚠️ REVIEW - patents are public |
| `nualgebra` | ? | Open source library | ✅ YES - already open |
| `uso` | ? | Transport layer | ✅ YES - no proprietary algorithms |
| `HubbleBubble` | ? | Validation tool | ✅ YES - already on Zenodo |
| `uha_nbs` | ? | Notebooks | ⚠️ REVIEW - may contain formulas |

---

## PART 2: MAKING REPOSITORIES PUBLIC

### Step-by-Step Process

#### Before Making Any Repository Public

**Security Checklist:**
```bash
cd /got/[REPOSITORY]

# 1. Scan for ObserverTensor class
git grep -i "class ObserverTensor"

# 2. Scan for epistemic formulas
git grep -i "u_epistemic.*abs\|delta_T.*sqrt"

# 3. Scan for tensor formulas
git grep -i "zero_t.*=.*redshift\|P_m.*=.*sigma"

# 4. Check git history
git log --all -p -S "ObserverTensor" | grep "class ObserverTensor"

# 5. Check for credential/secrets
git grep -i "api_key\|secret\|password\|token" | grep -v ".gitignore"
```

**If ANY of above scans return results:**
- 🛑 STOP - Do not publish
- Clean files or move to private backup
- Clean git history with `git filter-branch`
- Verify cleaning, then proceed

---

### Repository-by-Repository Publishing Plan

#### 1. uha_blackbox - READY TO PUBLISH ✅

**Status:** Ready (security cleanup completed Oct 24, 2025)

**What was secured:**
- ✅ Configuration templates obfuscated
- ✅ Binary-only distribution (no formulas)
- ✅ Git history cleaned
- ✅ Documentation genericized

**Publishing Steps:**
```bash
cd /got/uha_blackbox

# 1. Final verification
git status
git log --oneline -10

# 2. Make repository public on GitHub
# Via GitHub web interface:
# Settings → Danger Zone → Change visibility → Make public

# 3. Or create new public repo
gh repo create abba-01/uha-blackbox --public --source=. --remote=origin
git push -u origin master

# 4. Add description and topics
gh repo edit --description "UHA Binary Distribution - Conservative Uncertainty Propagation"
gh repo edit --add-topic hubble-tension
gh repo edit --add-topic cosmology
gh repo edit --add-topic uncertainty-quantification
```

**Post-Publication:**
- Add DOI badge to README (after Zenodo integration)
- Add license (MIT recommended for tools)
- Add citation file (CITATION.cff)

---

#### 2. hubble - READY AFTER VERIFICATION ⚠️

**Status:** Implementation cleaned, needs verification

**What was secured:**
- ✅ montecarlo/ directory removed (55+ files)
- ✅ Implementation code removed
- ✅ Git history cleaned
- ✅ SECURITY_NOTICE.md added

**Pre-Publication Verification:**
```bash
cd /got/hubble

# Run comprehensive security scan
bash /tmp/verify_hubble.sh

# Expected results:
# ✅ No ObserverTensor class
# ✅ No observer tensor formulas
# ✅ No epistemic distance calculations
# ✅ montecarlo/ removed
# ✅ Critical files removed
```

**If verification passes:**
```bash
# Make public
gh repo create abba-01/hubble-tension-resolution --public --source=. --remote=origin
git push -u origin master
```

---

#### 3. nualgebra - LIKELY ALREADY PUBLIC ✅

**Status:** Open source library (already on Zenodo as 17283314)

**Action Required:**
```bash
cd /got/nualgebra

# 1. Check if remote exists
git remote -v

# 2. If private, make public
# GitHub Settings → Make public

# 3. Add open source license if missing
# MIT or Apache 2.0 recommended

# 4. Update documentation
# Add installation instructions
# Add usage examples
# Add citation info
```

---

#### 4. hubble-tensor - REVIEW BEFORE PUBLISHING ⚠️

**Status:** Patent application materials

**Consideration:**
- Patents are PUBLIC by definition once filed
- Application materials may be redundant
- May want to keep private until patent prosecution complete

**Recommendation:**
- ⏸️ Keep private until patent examiner reviews
- Consult with IP attorney
- If publishing, redact attorney work product

---

#### 5. uso, HubbleBubble, uha_nbs - QUICK REVIEW NEEDED

**Action:**
```bash
# For each repository:
cd /got/[REPO]

# Security scan
git grep -i "ObserverTensor\|epistemic.*abs\|zero_t.*redshift"

# If clean → publish
# If dirty → clean first, then publish
```

---

## PART 3: ZENODO DOI STRATEGY

### Zenodo-GitHub Integration (Recommended)

#### Setup (One-Time)

1. **Enable Integration:**
   - Go to https://zenodo.org/account/settings/github/
   - Sign in with GitHub
   - Authorize Zenodo to access repositories
   - Enable for specific repos (e.g., `abba-01/uha-blackbox`)

2. **How It Works:**
   - Create GitHub release → Zenodo auto-archives → DOI assigned
   - Permanent snapshot preserved
   - Citation metadata generated
   - DOI badge provided

#### Creating DOI for uha_blackbox

```bash
cd /got/uha_blackbox

# 1. Ensure repository is public (see above)

# 2. Verify content is ready
ls -la
cat README.md

# 3. Create git tag for release
git tag -a v1.0.0 -m "UHA Blackbox v1.0.0 - Binary Distribution

First public release of UHA (Universal Hubble Aggregation) blackbox tool.

Features:
- Binary-only distribution (proprietary algorithms protected)
- Template-based configuration
- Cosmological parameter support
- TLV binary format for data serialization

This release supports conservative uncertainty propagation for
cosmological measurements without exposing implementation details.

Patent: US 63/902,536 (Patent Pending)
"

# 4. Push tag to GitHub
git push origin v1.0.0

# 5. Create release on GitHub
gh release create v1.0.0 \
    --title "UHA Blackbox v1.0.0" \
    --notes "First public release. Binary-only distribution of UHA tool for conservative uncertainty propagation in cosmological measurements.

**Installation:**
\`\`\`bash
pip install uha-official>=1.0.0
\`\`\`

**Usage:**
See README.md for configuration and examples.

**Citation:**
DOI will be automatically assigned by Zenodo upon release.

**Patent:** US 63/902,536 (Patent Pending)" \
    --latest

# 6. Zenodo automatically:
# - Archives the release
# - Assigns DOI (e.g., 10.5281/zenodo.XXXXXXX)
# - Creates citation page
# - Generates badge

# 7. Add DOI badge to README
# After DOI assigned, edit README.md:
# [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
```

---

### Alternative: Manual Zenodo Upload

If GitHub integration is not available:

```bash
# 1. Create archive
cd /got/uha_blackbox
git archive --format=zip --prefix=uha-blackbox-v1.0.0/ --output=../uha-blackbox-v1.0.0.zip HEAD

# 2. Go to https://zenodo.org/deposit/new
# 3. Upload ZIP file
# 4. Fill metadata form:
#    - Title: "UHA Blackbox v1.0.0 - Binary Distribution"
#    - Authors: Eric D. Martin (ORCID: 0009-0006-5944-1742)
#    - Description: [Copy from release notes]
#    - Keywords: hubble-tension, cosmology, uncertainty-quantification
#    - License: MIT (or proprietary if preferred)
#    - Related identifiers: Link to GitHub repo
# 5. Publish
# 6. Get assigned DOI
# 7. Add DOI to README and repository
```

---

## PART 4: NAMING CONVENTION STANDARDIZATION

### Current Naming Issues

| Repository | Current Name | Issue |
|------------|--------------|-------|
| uha_blackbox | snake_case | Inconsistent |
| hubble-tensor | kebab-case | ✅ Preferred |
| HubbleBubble | PascalCase | Inconsistent |
| nualgebra | lowercase | Unclear separation |
| uso | acronym | Too cryptic |

### Recommended Standard: kebab-case

**Rationale:**
- ✅ Most common in open source (linux, react, tensorflow)
- ✅ URL-friendly (no encoding needed)
- ✅ Git-friendly (no case sensitivity issues on Windows/Mac)
- ✅ Easy to read with clear word separation
- ✅ GitHub convention

**Proposed Renamings:**

| Current | Recommended | Alias? |
|---------|-------------|--------|
| uha_blackbox | `uha-blackbox` | Yes - keep redirect |
| HubbleBubble | `hubble-bubble` | Yes - tool is known |
| nualgebra | `nu-algebra` | Yes - clearer |
| uso | `universal-system-origin` | Yes - less cryptic |
| hubble | `hubble-tension-resolution` | Optional - more descriptive |
| hubble-tensor | `hubble-tensor` | ✅ Already correct |

---

### Repository Renaming Process

**Important:** GitHub automatically creates redirects from old names to new names, preserving:
- ✅ Stars and watchers
- ✅ Forks
- ✅ Issues and pull requests
- ✅ External links (redirect automatically)

#### Method 1: GitHub Web Interface (Recommended)

```
For each repository:
1. Go to https://github.com/abba-01/[REPO-NAME]
2. Click Settings
3. Scroll to "Repository name"
4. Enter new kebab-case name
5. Click "Rename"
6. GitHub shows warning about redirects
7. Confirm

Local update:
git remote set-url origin https://github.com/abba-01/[NEW-NAME].git
```

#### Method 2: GitHub CLI

```bash
# For each repository:
cd /got/[OLD_NAME]

# Rename on GitHub
gh repo rename [NEW-NAME]

# Update local remote
git remote set-url origin https://github.com/abba-01/[NEW-NAME].git

# Verify
git remote -v
```

#### Batch Renaming Script

```bash
#!/bin/bash

# Rename all repositories to kebab-case standard
declare -A RENAMES=(
    ["uha_blackbox"]="uha-blackbox"
    ["HubbleBubble"]="hubble-bubble"
    ["nualgebra"]="nu-algebra"
    ["uso"]="universal-system-origin"
)

for OLD in "${!RENAMES[@]}"; do
    NEW="${RENAMES[$OLD]}"
    echo "Renaming $OLD → $NEW"

    cd "/got/$OLD"

    # Rename on GitHub (requires gh CLI with auth)
    gh repo rename "$NEW" 2>/dev/null || echo "  Warning: rename failed, may need manual action"

    # Update local remote
    git remote set-url origin "https://github.com/abba-01/$NEW.git"

    echo "  ✓ Complete"
done
```

---

## PART 5: CONSOLIDATED ACTION PLAN

### Phase 1: Preparation (Today)

**Immediate Actions:**
- [ ] Run security verification on all repos
- [ ] Document current state of each repo
- [ ] Consult IP attorney about v2.0 disclosure implications

**Decision Points:**
- [ ] Confirm which repos to make public
- [ ] Decide on naming convention adoption
- [ ] Determine DOI publication timeline

---

### Phase 2: Secure & Rename (This Week)

**For each repository to be published:**

1. **Security Scan:**
   ```bash
   cd /got/[REPO]
   git grep -i "ObserverTensor\|epistemic.*abs\|zero_t.*redshift"
   ```

2. **If issues found:**
   - Backup to ~/private_backup/
   - Remove sensitive files
   - Clean git history
   - Verify cleaning

3. **Rename to kebab-case:**
   ```bash
   gh repo rename [new-kebab-case-name]
   git remote set-url origin https://github.com/abba-01/[new-name].git
   ```

---

### Phase 3: Publish Repositories (Next Week)

**Priority Order:**

1. **uha-blackbox** (ready now)
   - Make public
   - Add comprehensive README
   - Add LICENSE
   - Add CITATION.cff

2. **nu-algebra** (likely already public)
   - Verify security
   - Update documentation
   - Add examples

3. **hubble-tension-resolution** (after verification)
   - Final security check
   - Make public
   - Link to Zenodo packages

4. **hubble-bubble** (validation tool)
   - Security check
   - Make public
   - Update version

5. **Other repos** (as needed)
   - Assess case-by-case

---

### Phase 4: Zenodo DOI Integration (Next Week)

1. **Enable Zenodo-GitHub integration:**
   - Link GitHub account to Zenodo
   - Enable for public repositories

2. **Create releases for key repos:**
   - uha-blackbox v1.0.0
   - nu-algebra v[current]
   - hubble-tension-resolution v1.0.0

3. **Get DOIs assigned:**
   - Zenodo auto-creates on release
   - Add DOI badges to READMEs
   - Update citations

4. **Cross-link materials:**
   - Link repos to DOIs
   - Link DOIs to repos
   - Create unified citation guide

---

## PART 6: RISK ASSESSMENT & MITIGATION

### Risk 1: v2.0 Monte Carlo Already Exposed Implementation

**Risk Level:** 🔴 HIGH (but past event)

**Impact:**
- ObserverTensor class is public domain
- Implementation details accessible to competitors
- Patent may have reduced scope

**Mitigation:**
- ✅ Patent filed within 1-year grace period (likely valid)
- ✅ Binary-only distribution for future tools
- ⚠️ Consult IP attorney on enforcement strategy
- 📝 Document that v3.0 is separate invention

**Going Forward:**
- No point hiding what's already public
- Focus on protecting v3.0 (Observer Tensor without N/U or MC)
- Consider publishing v2.0 openly since it's already exposed

---

### Risk 2: Additional Exposures During Publication

**Risk Level:** 🟡 MEDIUM (controllable)

**Impact:**
- Accidental exposure of proprietary v3.0 details
- Trade secrets in git history

**Mitigation:**
- ✅ Comprehensive security scans before publishing
- ✅ Git history cleaning protocol
- ✅ Verification scripts
- 📋 Checklist for each repository

---

### Risk 3: Patent Invalidity

**Risk Level:** 🟡 MEDIUM (requires legal review)

**Impact:**
- v2.0 publication may invalidate patent
- Prior art concerns

**Mitigation:**
- ⚠️ **IMMEDIATE:** Consult IP attorney
- Document v3.0 as separate invention
- File continuation if needed
- Consider trade secret strategy for v3.0

---

### Risk 4: Naming Changes Break External Links

**Risk Level:** 🟢 LOW (handled by GitHub)

**Impact:**
- External links to old names
- Cloned repositories

**Mitigation:**
- ✅ GitHub auto-redirects old names to new
- 📝 Update documentation with new names
- 📧 Notify collaborators of changes

---

## PART 7: INTELLECTUAL PROPERTY STRATEGY

### Current IP Assets

1. **Patent US 63/902,536** (Filed Oct 21, 2025)
   - Provisional patent
   - 12 months to file non-provisional
   - Covers observer tensor framework

2. **Published Implementations**
   - v1.0 N/U Algebra (DOI 10.5281/zenodo.17322470) - PUBLIC
   - v2.0 Monte Carlo (DOI 10.5281/zenodo.17325811) - PUBLIC with ObserverTensor
   - v3.0 Observer Tensor (97.2%) - PRIVATE

3. **Trade Secrets**
   - v3.0 implementation details
   - Calibration algorithms
   - Anchor tensor formulas

---

### Recommended IP Strategy

**Short Term (Next 3 Months):**
- [ ] Consult IP attorney about v2.0 disclosure impact
- [ ] File non-provisional patent within 12 months (by Oct 21, 2026)
- [ ] Keep v3.0 as trade secret (binary-only distribution)
- [ ] Document v3.0 as separate invention from v2.0

**Medium Term (6-12 Months):**
- [ ] Publish v1.0 and v2.0 openly (already public)
- [ ] Market uha-blackbox as commercial product
- [ ] License v3.0 implementation separately
- [ ] Consider patent continuation for additional claims

**Long Term (1-2 Years):**
- [ ] Build commercial products around UHA technology
- [ ] License to industry (defense, aerospace, finance)
- [ ] Publish papers on v3.0 methodology (after patent grants)

---

## PART 8: PUBLICATION CHECKLIST

### Pre-Publication Checklist (For Each Repository)

```markdown
Repository: _______________

Security:
- [ ] Scan for ObserverTensor class
- [ ] Scan for epistemic formulas
- [ ] Scan for tensor calculation formulas
- [ ] Check git history for leaks
- [ ] Verify no credentials/secrets
- [ ] Clean git history if needed
- [ ] Add .gitignore for sensitive files

Documentation:
- [ ] README.md comprehensive
- [ ] Installation instructions
- [ ] Usage examples
- [ ] Citation information
- [ ] License file (LICENSE or LICENSE.txt)
- [ ] CITATION.cff file
- [ ] Security notice if relevant

Repository Settings:
- [ ] Rename to kebab-case if needed
- [ ] Add description
- [ ] Add topics/tags
- [ ] Enable Issues if appropriate
- [ ] Enable Discussions if appropriate
- [ ] Add collaborators if needed

Quality:
- [ ] Code passes lint/formatting
- [ ] Tests run successfully
- [ ] Documentation builds
- [ ] Examples work

Legal:
- [ ] License appropriate
- [ ] Patent notice included
- [ ] Copyright notices correct
- [ ] Third-party licenses acknowledged

Publishing:
- [ ] Make repository public
- [ ] Create initial release
- [ ] Enable Zenodo integration
- [ ] Verify DOI assignment
- [ ] Add DOI badge to README
- [ ] Announce on appropriate channels
```

---

## PART 9: IMMEDIATE NEXT STEPS

### What to Do Right Now

1. **Consult IP Attorney** (URGENT)
   - Show them v2.0 Monte Carlo package
   - Discuss patent implications
   - Get guidance on v3.0 protection strategy
   - Ask about filing non-provisional patent early

2. **Run Security Verification**
   ```bash
   cd /got
   for repo in */; do
       echo "=== $repo ==="
       cd "$repo"
       git grep -i "class ObserverTensor" || echo "  ✓ Clean"
       cd ..
   done
   ```

3. **Make First Repository Public**
   - Start with uha-blackbox (already secured)
   - Gain experience with publishing process
   - Verify nothing breaks

4. **Set Up Zenodo Integration**
   - Link GitHub account
   - Enable for uha-blackbox
   - Create first release with DOI

5. **Rename Repositories**
   - Adopt kebab-case standard
   - Update local remotes
   - Update documentation

---

## SUMMARY

**Current State:**
- Multiple repos with mixed security levels
- v2.0 Monte Carlo already exposed ObserverTensor
- Patent filed but implementation partially public

**Recommended Strategy:**
1. Accept v2.0 is public, protect v3.0 aggressively
2. Make cleaned repositories public
3. Standardize names to kebab-case
4. Get DOIs via Zenodo-GitHub integration
5. Consult IP attorney immediately

**Timeline:**
- Week 1: IP consultation, security verification
- Week 2: Repository renaming, preparation
- Week 3: Publish first repos with DOIs
- Week 4: Complete remaining repos

**Critical Path:**
IP attorney consultation → Security verification → Publish uha-blackbox → Zenodo DOI → Remaining repos

---

**Next Document:** See `/got/NAMING_CONVENTION_MIGRATION.md` for detailed renaming instructions.

---

**Last Updated:** 2025-10-24
