# Zenodo DOI Integration - Step-by-Step Guide
**Date:** 2025-10-24
**Purpose:** Get DOIs for published GitHub repositories
**Target:** uha-blackbox, hubble-tension-resolution, and others

---

## CURRENT STATUS

### Repositories Ready for Zenodo ✅

**With Releases (Priority):**
1. ✅ **uha-blackbox** - v1.0.0 release created
2. ✅ **hubble-tension-resolution** - v1.0.0 release created

**Public, Ready for Releases:**
3. ✅ **hubble-bubble** (renamed from HubbleBubble)
4. ✅ **nu-algebra** (renamed from nualgebra)
5. ✅ **nu-anthropology** (renamed & public)

**Currently Private (Not for Zenodo):**
- theories (made private)
- Eric-D-Martin (made private)
- count2infinity (made private)
- nu-psychology (made private)

---

## STEP-BY-STEP: ZENODO INTEGRATION

### Step 1: Access Zenodo GitHub Settings

**Open in Browser:**
```
https://zenodo.org/account/settings/github/
```

**You will need:**
- Zenodo account (create if needed at https://zenodo.org/signup/)
- GitHub account (already have)

---

### Step 2: Authorize GitHub Access (First Time Only)

**If not already connected:**

1. Click **"Connect"** or **"Sign in with GitHub"**
2. Authorize Zenodo to access your repositories
3. Grant permissions when prompted

**Permissions needed:**
- ✅ Read repository metadata
- ✅ Read releases
- ✅ Create webhooks

---

### Step 3: Enable Repositories for Zenodo

**On the Zenodo GitHub settings page, you'll see a list of your repositories.**

**Enable these repositories by toggling them ON:**

#### Priority 1 (Has Releases):
- [ ] **abba-01/uha-blackbox** ← Toggle ON (has v1.0.0)
- [ ] **abba-01/hubble-tension-resolution** ← Toggle ON (has v1.0.0)

#### Priority 2 (Will Create Releases):
- [ ] **abba-01/hubble-bubble** ← Toggle ON
- [ ] **abba-01/nu-algebra** ← Toggle ON
- [ ] **abba-01/nu-anthropology** ← Toggle ON

**What happens when you toggle ON:**
- Zenodo creates a webhook in your GitHub repository
- Future releases automatically trigger DOI creation
- **If a release already exists (v1.0.0), Zenodo archives it immediately**

---

### Step 4: Wait for Zenodo to Process

**After enabling repositories:**

1. Zenodo will process existing releases (v1.0.0 for uha-blackbox and hubble-tension-resolution)
2. This usually takes **1-5 minutes**
3. You'll receive an email notification when DOI is assigned

**Check processing status:**
```
https://zenodo.org/deposit
```

You should see:
- uha-blackbox v1.0.0 - "Published" or "Processing"
- hubble-tension-resolution v1.0.0 - "Published" or "Processing"

---

### Step 5: Get Your Assigned DOIs

**Once processed, each repository will have a DOI like:**
```
10.5281/zenodo.XXXXXXX
```

**Find your DOIs:**

1. Go to https://zenodo.org/deposit
2. Click on each deposit to see details
3. Note the DOI for each repository

**Expected DOIs:**
- uha-blackbox v1.0.0: `10.5281/zenodo.???????`
- hubble-tension-resolution v1.0.0: `10.5281/zenodo.???????`

---

## STEP-BY-STEP: ADDING DOI BADGES

### Step 1: Add Badge to uha-blackbox README

**Once you have the DOI (e.g., 10.5281/zenodo.1234567):**

```bash
cd /got/uha_blackbox

# Create backup
cp README.md README.md.backup

# Edit README.md to add badge after title
# Add this line:
# [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
```

**Example edit:**
```markdown
# UHA Official - Single Source of Truth (SSOT)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1234567.svg)](https://doi.org/10.5281/zenodo.1234567)

**Universal Horizon Address - Official Implementation**
```

**Commit and push:**
```bash
git add README.md
git commit -m "Add Zenodo DOI badge

DOI: 10.5281/zenodo.XXXXXXX
Release: v1.0.0
"
git push origin master
```

---

### Step 2: Add Badge to hubble-tension-resolution README

**Same process:**

```bash
cd /got/hubble

# Check if README exists
ls -la README.md

# If exists, edit to add badge at top
# If not, create minimal README with badge

git add README.md
git commit -m "Add Zenodo DOI badge"
git push origin master
```

---

### Step 3: Update Release Notes with DOI

**Edit the release notes to include actual DOI:**

```bash
cd /got/uha_blackbox

# Edit release v1.0.0
gh release edit v1.0.0 \
    --notes "**First Public Release**

Binary-only distribution of the Universal Hubble Aggregation (UHA) tool.

## DOI
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

**Cite this release:**
\`\`\`
Martin, E.D. (2025). UHA Blackbox v1.0.0 (Version v1.0.0) [Software].
Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX
\`\`\`

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

## Patent
US Provisional Patent 63/902,536 (Patent Pending)

## License
Proprietary - All Your Baseline LLC

## Author
Eric D. Martin (eric.martin1@wsu.edu)
ORCID: 0009-0006-5944-1742"
```

---

## OPTIONAL: CREATE RELEASES FOR OTHER REPOS

### hubble-bubble

```bash
cd /got/HubbleBubble

git tag -a v1.1.1 -m "HubbleBubble v1.1.1

Validation and visualization tool for Hubble tension analysis.

Author: Eric D. Martin
"

git push origin v1.1.1

gh release create v1.1.1 \
    --title "HubbleBubble v1.1.1" \
    --notes "Validation and visualization tool for Hubble tension resolution research.

## Features
- Interactive visualizations
- Data validation
- Results comparison

## Citation
DOI will be automatically assigned by Zenodo.

## Author
Eric D. Martin (eric.martin1@wsu.edu)" \
    --latest
```

### nu-algebra

```bash
cd /got/nualgebra

git tag -a v1.0.0 -m "N/U Algebra v1.0.0

Nominal/Uncertainty algebra library for conservative uncertainty propagation.

Author: Eric D. Martin
"

git push origin v1.0.0

gh release create v1.0.0 \
    --title "N/U Algebra v1.0.0" \
    --notes "**First Public Release**

Nominal/Uncertainty (N/U) Algebra library for conservative uncertainty propagation.

## Features
- Conservative merge operations
- Uncertainty propagation
- Statistical validation

## Installation
\`\`\`bash
pip install nu-algebra
\`\`\`

## Citation
DOI will be automatically assigned by Zenodo.

## License
MIT

## Author
Eric D. Martin (eric.martin1@wsu.edu)" \
    --latest
```

### nu-anthropology

```bash
cd /got/nualgebra_anthropology

git tag -a v1.0.0 -m "N/U Anthropology v1.0.0

Application of N/U Algebra to anthropological research.

Author: Eric D. Martin
"

git push origin v1.0.0

gh release create v1.0.0 \
    --title "N/U Anthropology v1.0.0" \
    --notes "Application of Nominal/Uncertainty (N/U) Algebra to anthropological research.

## Citation
DOI will be automatically assigned by Zenodo.

## Author
Eric D. Martin (eric.martin1@wsu.edu)" \
    --latest
```

---

## TROUBLESHOOTING

### Issue: "Repository not showing in Zenodo list"

**Solution:**
1. Ensure repository is PUBLIC on GitHub
2. Refresh Zenodo GitHub settings page
3. Wait a few minutes and try again
4. Check GitHub App permissions at https://github.com/settings/installations

---

### Issue: "Zenodo not creating DOI for existing release"

**Solution:**
1. Toggle repository OFF then ON again in Zenodo settings
2. Or create a new release (v1.0.1) which will trigger DOI creation
3. Contact Zenodo support if issue persists

---

### Issue: "DOI badge not showing in README"

**Solution:**
1. Verify DOI is correct (check https://zenodo.org)
2. Ensure badge URL is correct format:
   ```
   [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
   ```
3. Wait a few minutes for GitHub to render markdown
4. Hard refresh browser (Ctrl+F5)

---

## VERIFICATION CHECKLIST

After Zenodo integration:

### For uha-blackbox:
- [ ] Repository enabled in Zenodo GitHub settings
- [ ] DOI assigned by Zenodo (10.5281/zenodo.???????)
- [ ] DOI badge added to README.md
- [ ] Release notes updated with DOI
- [ ] DOI resolves at https://doi.org/10.5281/zenodo.???????
- [ ] Zenodo page shows correct metadata (title, author, description)

### For hubble-tension-resolution:
- [ ] Repository enabled in Zenodo GitHub settings
- [ ] DOI assigned by Zenodo
- [ ] DOI badge added to README.md (if README exists)
- [ ] DOI resolves correctly

### For other repositories:
- [ ] Enabled in Zenodo (if desired)
- [ ] Releases created (if desired)
- [ ] DOIs assigned

---

## CITATION FORMAT

### BibTeX Format (Auto-generated by Zenodo)

Zenodo provides citation in multiple formats. Example:

```bibtex
@software{martin_2025_uha_blackbox,
  author       = {Martin, Eric D.},
  title        = {UHA Blackbox v1.0.0},
  month        = oct,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.XXXXXXX},
  url          = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
```

### APA Format

```
Martin, E. D. (2025). UHA Blackbox (Version v1.0.0) [Computer software].
Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX
```

---

## ZENODO METADATA

**Zenodo automatically extracts from GitHub:**
- Title (from repository name)
- Description (from repository description)
- Authors (from GitHub account)
- License (from LICENSE file)
- Keywords (from repository topics)

**To improve metadata:**

1. Add topics to GitHub repository:
   ```bash
   gh repo edit abba-01/uha-blackbox \
       --add-topic hubble-tension \
       --add-topic cosmology \
       --add-topic uncertainty-quantification \
       --add-topic conservative-propagation
   ```

2. Ensure LICENSE file exists in repository

3. Add comprehensive description via GitHub settings

---

## NEXT STEPS AFTER DOI ASSIGNMENT

1. **Update all documentation** with DOI
2. **Add CITATION.cff file** to repository
3. **Announce release** on appropriate channels:
   - Twitter
   - arXiv (if publishing paper)
   - Mailing lists (cosmology, statistics)
4. **Update paper manuscripts** with DOI citations
5. **Link DOI to ORCID** profile (https://orcid.org)

---

## SUMMARY

**What You Need to Do (Manual Steps):**

1. Go to https://zenodo.org/account/settings/github/
2. Enable these repositories:
   - abba-01/uha-blackbox
   - abba-01/hubble-tension-resolution
   - (Optional: hubble-bubble, nu-algebra, nu-anthropology)
3. Wait 1-5 minutes for DOI assignment
4. Get DOI from https://zenodo.org/deposit
5. Add DOI badge to README files
6. Commit and push changes

**What Happens Automatically:**
- Zenodo creates webhook in GitHub
- Existing releases (v1.0.0) are archived
- DOI is assigned
- Citation metadata is extracted
- DOI becomes permanent and citable

**Estimated Time:** 10-15 minutes

---

**Last Updated:** 2025-10-24
