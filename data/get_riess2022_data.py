#!/usr/bin/env python3
"""
Download Riess et al. 2022 (R22) Supplementary Data

Attempts to download published MCMC chains and supplementary data from:
- ApJ journal supplementary materials
- Zenodo (if deposited)
- SH0ES team website

Target paper: Riess et al. 2022, ApJ, 934, L7
"A Comprehensive Measurement of the Local Value of the Hubble Constant
with 1 km/s/Mpc Uncertainty from the Hubble Space Telescope and the
SH0ES Team"

Author: Eric D. Martin
Date: 2025-10-12
"""

import requests
from pathlib import Path
import json
from datetime import datetime

class Riess2022DataFetcher:
    """Fetch supplementary data from Riess et al. 2022."""

    def __init__(self, output_dir: str = "./riess2022_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Paper details
        self.paper_info = {
            'title': 'A Comprehensive Measurement of the Local Value of the Hubble Constant',
            'authors': 'Riess et al.',
            'journal': 'ApJ',
            'volume': 934,
            'page': 'L7',
            'year': 2022,
            'doi': '10.3847/2041-8213/ac5c5b',
            'ads': '2022ApJ...934L...7R'
        }

    def get_apj_data_url(self):
        """Construct ApJ data supplement URL."""
        # ApJ data supplements typically at:
        # https://iopscience.iop.org/article/{doi}/meta with "Data behind the figures"
        doi = self.paper_info['doi']
        return f"https://doi.org/{doi}"

    def check_zenodo_deposit(self):
        """Check if supplementary data on Zenodo."""
        print("\n[ZENODO] Searching for Riess 2022 data deposits...")

        # Search Zenodo API
        search_url = "https://zenodo.org/api/records"
        params = {
            'q': 'Riess Hubble constant 2022',
            'sort': 'mostrecent'
        }

        try:
            response = requests.get(search_url, params=params, timeout=10)
            if response.status_code == 200:
                results = response.json()
                hits = results.get('hits', {}).get('hits', [])

                if hits:
                    print(f"✓ Found {len(hits)} potential matches on Zenodo")
                    for i, hit in enumerate(hits[:5], 1):
                        print(f"\n  [{i}] {hit['metadata']['title']}")
                        print(f"      DOI: {hit['doi']}")
                        print(f"      URL: {hit['links']['html']}")
                else:
                    print("  - No Zenodo deposits found")
            else:
                print(f"  ✗ Zenodo API error: {response.status_code}")

        except Exception as e:
            print(f"  ✗ Error querying Zenodo: {e}")

    def check_shoes_website(self):
        """Check SH0ES team website for data."""
        print("\n[SH0ES WEBSITE] Checking team data portal...")

        urls_to_check = [
            "https://sites.google.com/view/shoes-h0",
            "https://sites.google.com/view/shoes-h0/data",
            "https://archive.stsci.edu/prepds/shoes/"
        ]

        for url in urls_to_check:
            try:
                response = requests.head(url, timeout=5, allow_redirects=True)
                if response.status_code == 200:
                    print(f"  ✓ Accessible: {url}")
                else:
                    print(f"  - Not found: {url} ({response.status_code})")
            except Exception as e:
                print(f"  ✗ Error checking {url}: {e}")

    def check_vizier_catalog(self):
        """Check VizieR for published tables."""
        print("\n[VIZIER] Checking for published catalog data...")

        # VizieR query via API
        vizier_url = "https://vizier.cds.unistra.fr/viz-bin/votable"
        params = {
            '-source': 'J/ApJ/934/L7',  # Journal reference
            '-out.max': '1'  # Just check existence
        }

        try:
            response = requests.get(vizier_url, params=params, timeout=10)
            if response.status_code == 200 and len(response.content) > 1000:
                print(f"  ✓ VizieR catalog exists: J/ApJ/934/L7")
                print(f"    URL: https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/ApJ/934/L7")
                print(f"    Contains: Published tables from R22 paper")
            else:
                print("  - No VizieR catalog found for this paper")
        except Exception as e:
            print(f"  ✗ Error querying VizieR: {e}")

    def download_arxiv_source(self):
        """Download arXiv source files (may contain data)."""
        print("\n[ARXIV] Checking for arXiv version and source files...")

        # R22 arXiv ID: 2112.04510
        arxiv_id = "2112.04510"
        arxiv_pdf = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        arxiv_source = f"https://arxiv.org/e-print/{arxiv_id}"

        try:
            # Check if source tarball exists
            response = requests.head(arxiv_source, timeout=5)
            if response.status_code == 200:
                print(f"  ✓ arXiv source available: {arxiv_id}")
                print(f"    Source: {arxiv_source}")
                print(f"    PDF: {arxiv_pdf}")

                # Download source
                print(f"\n  Downloading arXiv source tarball...")
                response = requests.get(arxiv_source, timeout=30)
                if response.status_code == 200:
                    output_file = self.output_dir / f"arxiv_{arxiv_id}_source.tar.gz"
                    with open(output_file, 'wb') as f:
                        f.write(response.content)
                    print(f"  ✓ Downloaded: {output_file}")
                    print(f"    Size: {len(response.content) / 1024:.1f} KB")
                    print(f"    Extract with: tar -xzf {output_file.name}")
            else:
                print(f"  - arXiv source not available ({response.status_code})")

        except Exception as e:
            print(f"  ✗ Error downloading arXiv source: {e}")

    def check_direct_apj_download(self):
        """Try to access ApJ machine-readable tables."""
        print("\n[ApJ DATA] Checking for machine-readable tables...")

        # ApJ machine-readable table URLs typically:
        # https://iopscience.iop.org/article/10.3847/2041-8213/ac5c5b/suppdata
        doi = self.paper_info['doi']
        suppdata_url = f"https://iopscience.iop.org/article/{doi}/suppdata"

        try:
            response = requests.get(suppdata_url, timeout=10, allow_redirects=True)
            if response.status_code == 200:
                print(f"  ✓ Supplementary data page exists")
                print(f"    URL: {suppdata_url}")
                print(f"    Visit this URL in browser to download files")

                # Save HTML for inspection
                html_file = self.output_dir / "apj_suppdata_page.html"
                with open(html_file, 'w') as f:
                    f.write(response.text)
                print(f"  ✓ Saved page HTML: {html_file}")
                print(f"    Inspect to find direct download links")
            else:
                print(f"  - No supplementary data page found ({response.status_code})")

        except Exception as e:
            print(f"  ✗ Error accessing ApJ data: {e}")

    def generate_data_request_template(self):
        """Generate email template to request data from authors."""
        print("\n[DATA REQUEST] Generating email template...")

        template = f"""
Subject: Request for MCMC Posterior Samples - Riess et al. 2022 (ApJ, 934, L7)

Dear Dr. Riess and SH0ES Team,

I am a PhD applicant working on observer-domain uncertainty propagation
methods applied to the Hubble tension. I have developed a framework that
achieves 91% reduction using published aggregate H₀ values.

To validate the full methodology with empirical covariance structures,
I would greatly appreciate access to:

1. MCMC posterior samples for H₀ and nuisance parameters from your
   2022 ApJ paper (934, L7)
2. Covariance matrices between:
   - Anchor distances (NGC 4258, LMC, etc.)
   - Cepheid period-luminosity parameters
   - Final H₀ measurement

If these are already publicly available (e.g., as journal supplementary
data, on Zenodo, or the SH0ES website), please direct me to the location.

I will properly cite the data source and acknowledge the SH0ES team in
any resulting publications.

Thank you for your pioneering work on the distance ladder!

Best regards,
[Your Name]
[Your Institution]
[Your Email]

---
Paper reference:
{self.paper_info['authors']} {self.paper_info['year']}
{self.paper_info['journal']}, {self.paper_info['volume']}, {self.paper_info['page']}
DOI: {self.paper_info['doi']}
---
"""

        template_file = self.output_dir / "data_request_email_template.txt"
        with open(template_file, 'w') as f:
            f.write(template.strip())

        print(f"  ✓ Saved template: {template_file}")
        print(f"\n  Contact info:")
        print(f"    Adam Riess: ariess@stsci.edu")
        print(f"    Stefano Casertano: casertano@stsci.edu")

    def save_search_summary(self):
        """Save summary of data source search."""
        summary = {
            'search_date': datetime.now().isoformat(),
            'paper': self.paper_info,
            'data_sources_checked': [
                'ApJ supplementary data portal',
                'Zenodo repository',
                'SH0ES team website',
                'VizieR catalog',
                'arXiv source files'
            ],
            'recommended_actions': [
                'Visit ApJ suppdata URL in browser',
                'Check VizieR catalog J/ApJ/934/L7',
                'Extract arXiv source tarball (if contains data)',
                'Contact authors directly using template'
            ],
            'urls': {
                'doi': f"https://doi.org/{self.paper_info['doi']}",
                'apj': f"https://iopscience.iop.org/article/{self.paper_info['doi']}",
                'arxiv': f"https://arxiv.org/abs/2112.04510",
                'vizier': "https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/ApJ/934/L7",
                'shoes': "https://sites.google.com/view/shoes-h0"
            }
        }

        summary_file = self.output_dir / "data_search_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✓ Saved search summary: {summary_file}")


def main():
    """Main execution."""
    print("=" * 80)
    print("RIESS ET AL. 2022 DATA RETRIEVAL")
    print("=" * 80)
    print(f"Target: ApJ, 934, L7 (2022)")
    print(f"DOI: 10.3847/2041-8213/ac5c5b")
    print(f"H₀ = 73.04 ± 1.04 km/s/Mpc")
    print("=" * 80)

    fetcher = Riess2022DataFetcher()

    # Check all possible data sources
    fetcher.check_direct_apj_download()
    fetcher.check_vizier_catalog()
    fetcher.check_zenodo_deposit()
    fetcher.check_shoes_website()
    fetcher.download_arxiv_source()

    # Generate request template
    fetcher.generate_data_request_template()

    # Save summary
    fetcher.save_search_summary()

    print("\n" + "=" * 80)
    print("SEARCH COMPLETE")
    print("=" * 80)
    print("\nRecommended next steps:")
    print("  1. Visit ApJ suppdata URL in web browser")
    print("  2. Check VizieR catalog for published tables")
    print("  3. Extract arXiv source (may contain ancillary data)")
    print("  4. If no data found, contact authors using template")
    print("=" * 80)


if __name__ == "__main__":
    main()
