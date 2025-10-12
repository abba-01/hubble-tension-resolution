# SH0ES Data Access via MAST

**Last Updated**: 2025-10-12
**Status**: Direct MAST query capability installed and tested ✅

## Summary

Successfully installed and tested direct MAST (Mikulski Archive for Space Telescopes) query capabilities for accessing SH0ES program data. Found **2,870 observations** across 5 HST programs.

## Installation

```bash
pip install astroquery --user
```

**Dependencies installed**:
- astroquery 0.4.11
- astropy 7.1.1
- pyvo 1.7
- beautifulsoup4 4.14.2
- html5lib 1.1

## Query Results

### Programs Queried

| Program ID | Description | Cycle | Observations |
|------------|-------------|-------|--------------|
| GO-12880 | SH0ES SNe Ia observations | 19 | 1,085 |
| GO-13691 | Anchoring Distance Scale with Cepheids | 22 | 781 |
| GO-14216 | Reducing H0 Uncertainty | 23 | 531 |
| GO-15698 | Completing the Legacy | 26 | 456 |
| GO-16664 | Precision Distance Ladder | 28 | 17 |
| **TOTAL** | | | **2,870** |

### Instruments

- **ACS/WFC**: Advanced Camera for Surveys, Wide Field Channel
- **WFC3/IR**: Wide Field Camera 3, Infrared channel
- **WFC3/UVIS**: Wide Field Camera 3, UV/Visible channel

### Filters Used

**Primary Cepheid filters**:
- F555W (V-band, 555nm) - Optical reference
- F814W (I-band, 814nm) - Period-luminosity calibration
- F160W (H-band, 1.6μm) - Infrared, less extinction

**Additional filters**:
- F606W (broad V), F110W, F125W (J, H bands)
- F350LP (long-pass), F438W (B-band)
- G141 (grism for spectroscopy)
- F126N (narrow-band)

### Target Galaxies

**72 unique targets** including:

**Key Cepheid hosts** (with WFC3 observations):
- NGC-4536 (5 obs)
- NGC-4639 (14 obs)
- NGC-3370 (7 obs)
- NGC-3021 (5 obs)
- NGC-3982 (7 obs)
- NGC-1309 (5 obs)

**Notable targets**:
- NGC-4258 (Megamaser anchor - no direct WFC3 query results)
- M101 (Pinwheel Galaxy fields)
- NGC-1365, NGC-1448, NGC-5917
- IC-1613 (Local Group galaxy)
- Sculptor dwarf spheroidal fields

### Observation Timeline

- **Start**: MJD 56275.9 (≈ 2012-11-17, Cycle 19)
- **End**: MJD 59997.4 (≈ 2023-03-06, Cycle 28)
- **Span**: ~10 years of HST observations

## What Data Is Available

### Raw Observation Data

Each observation typically includes:
- **FLC**: Calibrated, CTE-corrected (ACS)
- **FLT**: Calibrated (WFC3)
- **DRZ**: Drizzled, combined images
- Associated calibration files

### Data Products Per Observation

- Science images (FITS format)
- Error/uncertainty maps
- Data quality flags
- World Coordinate System (WCS) headers
- Photometric zero-points
- Exposure metadata

### File Sizes

- Single calibrated image: ~50-200 MB
- Complete observation set: ~500 MB - 1 GB
- Full program: 10s to 100s of GB

## Usage Scripts

### Query Script: `mast_shoes_query.py`

Located in: `/run/media/root/OP01/got/hubble/data/mast_shoes_query.py`

**Functions**:
1. `query_shoes_programs()` - Query by HST program ID
2. `query_by_pi()` - Query by PI name (Note: pi_name filter not supported by current MAST API)
3. `query_cepheid_hosts()` - Query specific host galaxies
4. `get_available_products()` - List downloadable data products
5. `download_sample_data()` - Download subset for testing
6. `save_observation_summary()` - Export metadata to JSON/CSV

### Running Queries

```bash
cd /run/media/root/OP01/got/hubble/data
python mast_shoes_query.py
```

**Output files**:
- `shoes_data/shoes_programs_summary.json` - Metadata summary
- `shoes_data/shoes_observations_full.csv` - Complete observation table
- `shoes_data/shoes_cepheid_hosts_summary.json` - Cepheid host query results

## What This Means for Phase C Integration

### Current Status (Package 1)

- **91% reduction** using published aggregate H₀ values
- Observer tensors assigned by methodology
- No access to individual Cepheid measurements

### What SH0ES MAST Data Provides

**Potential improvements**:

1. **Individual Cepheid photometry**
   - Period-luminosity scatter analysis
   - Extinction corrections per star
   - Metallicity gradients
   - Individual measurement uncertainties

2. **Covariance structure extraction**
   - Filter-to-filter correlations (F555W, F814W, F160W)
   - Intra-galaxy systematics
   - Distance modulus uncertainty breakdowns

3. **Empirical observer tensor calibration**
   - Temporal domain: Observation epoch variations
   - Material probability: Metallicity dependencies
   - Spatial: Galactocentric distance effects
   - Angular: Orientation/inclination impacts

### Limitations

**What MAST data does NOT provide directly**:

1. ❌ **Pre-computed MCMC chains** - Would need to run analysis pipeline
2. ❌ **Full covariance matrices** - Present in published papers, not raw data
3. ❌ **Distance ladder propagation** - Requires multi-step calibration
4. ❌ **Ready-to-use H₀ posteriors** - Final results published separately

**What we need for 100% resolution**:

The critical missing piece is **published MCMC chains from Riess et al. papers**, specifically:
- Riess et al. 2022 (ApJ, R22): Full posterior samples
- Systematic uncertainty correlation matrices
- Covariance between anchor distances, Cepheid PL parameters, and H₀

These are typically available as **supplementary data** with the published papers, not in MAST archive.

## Next Steps for Empirical Tensor Extraction

### Option A: Use Raw Cepheid Data (Complex)

**Workflow**:
1. Download WFC3 images for key host galaxies
2. Extract Cepheid photometry (requires SExtractor, DOLPHOT, etc.)
3. Fit period-luminosity relations
4. Compute distance moduli with uncertainties
5. Propagate to H₀ with full covariance

**Effort**: ~2-3 months for complete pipeline
**Benefit**: Fully independent analysis with custom tensor extraction

### Option B: Access Published MCMC Chains (Recommended)

**Workflow**:
1. Download supplementary data from Riess et al. 2022 (ApJ)
2. Extract posterior samples for H₀ and nuisance parameters
3. Compute covariance matrix from MCMC chains
4. Run empirical tensor extraction (existing Phase C pipeline)
5. Validate against published results

**Effort**: ~1-2 weeks
**Benefit**: Validates Phase C methodology with real published data

### Option C: Request Data from SH0ES Team

**Workflow**:
1. Contact Adam Riess (STScI/JHU)
2. Request MCMC posterior samples from R22 paper
3. Cite as "private communication" or "data provided by authors"

**Effort**: ~1 week (waiting for response)
**Benefit**: Most direct path to real covariance structures

## Recommended Immediate Action

**For PhD applications (short-term)**:
- ✅ Use Package 1 validated results (91% reduction)
- ✅ Cite MAST data availability as "future work"
- ✅ Demonstrate query capability (shows technical competence)

**For publication track (medium-term)**:
- [ ] Download Riess et al. 2022 supplementary data
- [ ] Extract MCMC chains (if available)
- [ ] Run Phase C with real covariance matrices
- [ ] Publish updated results with empirical tensors

**For complete independence (long-term)**:
- [ ] Download raw Cepheid images from MAST
- [ ] Build independent photometry pipeline
- [ ] Generate alternative distance ladder
- [ ] Cross-validate with SH0ES published results

## Data Access Examples

### Example 1: Query Specific Target

```python
from astroquery.mast import Observations

# Query NGC-4258 (megamaser anchor)
obs = Observations.query_criteria(
    target_name="NGC-4258",
    obs_collection="HST",
    instrument_name="WFC3/UVIS"
)

print(f"Found {len(obs)} observations")
```

### Example 2: Get Data Products

```python
from astroquery.mast import Observations

# Get products for an observation
products = Observations.get_product_list(obs[0])

# Filter for calibrated images
cal_images = products[
    (products['productType'] == 'SCIENCE') &
    (products['productSubGroupDescription'] == 'FLC')
]

print(f"Found {len(cal_images)} calibrated images")
```

### Example 3: Download Data

```python
from astroquery.mast import Observations

# Download first 5 calibrated images
manifest = Observations.download_products(
    cal_images[:5],
    download_dir="./downloads"
)

print(f"Downloaded {len(manifest)} files")
```

## References

### MAST Documentation
- Main portal: https://mast.stsci.edu/
- API docs: https://astroquery.readthedocs.io/en/latest/mast/mast.html
- HST data guide: https://archive.stsci.edu/hst/

### SH0ES Publications
- **R22**: Riess et al. 2022, ApJ, 934, L7 (H₀ = 73.04 ± 1.04 km/s/Mpc)
- **R21**: Riess et al. 2021, ApJ, 908, L6
- **R19**: Riess et al. 2019, ApJ, 876, 85
- **R16**: Riess et al. 2016, ApJ, 826, 56

### SH0ES Team
- **PI**: Adam Riess (STScI/JHU, 2011 Nobel Prize)
- **Key collaborators**: Stefano Casertano, Wenlong Yuan, Lucas Macri

## Summary

✅ **Installed**: astroquery and dependencies
✅ **Tested**: Successfully queried 2,870 SH0ES observations
✅ **Documented**: 72 targets, 3 instruments, 5 HST programs
✅ **Scripts**: Complete query and download pipeline ready

**Next action**: Decide between Option A (complex, independent), Option B (recommended, published data), or Option C (fastest, collaboration).

For PhD applications, **the current 91% validated result is sufficient**. MAST data access demonstrates technical capability and future research direction.

---

**For questions or data access issues, consult**:
- MAST Help Desk: https://stsci.service-now.com/mast
- SH0ES website: https://sites.google.com/view/shoes-h0
- Adam Riess contact: ariess@stsci.edu
