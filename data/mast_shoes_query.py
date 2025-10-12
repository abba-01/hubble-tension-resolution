#!/usr/bin/env python3
"""
MAST Query Script for SH0ES Program Data

Queries the Mikulski Archive for Space Telescopes (MAST) for SH0ES-related
HST observations and downloads available datasets.

SH0ES (Supernovae and H0 for the Equation of State) Key Programs:
- GO-12880: SH0ES SNe Ia (PI: Riess, Cycle 19)
- GO-13691: Anchoring the Distance Scale with Cepheids (PI: Riess, Cycle 22)
- GO-14216: SH0ES: Reducing H0 Uncertainty (PI: Riess, Cycle 23)
- GO-15698: SH0ES: Completing the Legacy (PI: Riess, Cycle 26)
- GO-16664: SH0ES: Precision Distance Ladder (PI: Riess, Cycle 28)

Author: Eric D. Martin
Date: 2025-10-11
"""

from astroquery.mast import Observations, Mast
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

class SHOESDataQuery:
    """Query and download SH0ES program data from MAST."""

    def __init__(self, output_dir: str = "./shoes_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Known SH0ES program IDs
        self.shoes_programs = {
            'GO-12880': 'SH0ES SNe Ia observations (Cycle 19)',
            'GO-13691': 'Anchoring Distance Scale with Cepheids (Cycle 22)',
            'GO-14216': 'Reducing H0 Uncertainty (Cycle 23)',
            'GO-15698': 'Completing the Legacy (Cycle 26)',
            'GO-16664': 'Precision Distance Ladder (Cycle 28)'
        }

    def query_shoes_programs(self):
        """Query all known SH0ES programs and return observation table."""
        print("=" * 80)
        print("QUERYING MAST FOR SH0ES PROGRAM DATA")
        print("=" * 80)

        all_observations = []

        for program_id, description in self.shoes_programs.items():
            print(f"\n[{program_id}] {description}")
            print("-" * 80)

            try:
                # Query by proposal_id
                obs = Observations.query_criteria(proposal_id=program_id.split('-')[1])

                if len(obs) > 0:
                    print(f"  ✓ Found {len(obs)} observations")
                    all_observations.append(obs)

                    # Show sample
                    if len(obs) > 0:
                        print(f"    Instruments: {set(obs['instrument_name'])}")
                        print(f"    Targets: {len(set(obs['target_name']))} unique targets")
                else:
                    print(f"  ⚠ No observations found")

            except Exception as e:
                print(f"  ✗ Error querying {program_id}: {e}")

        if all_observations:
            from astropy.table import vstack
            combined = vstack(all_observations)
            print(f"\n{'=' * 80}")
            print(f"TOTAL: {len(combined)} observations across {len(self.shoes_programs)} programs")
            print(f"{'=' * 80}")
            return combined
        else:
            print("\n⚠ No observations found for any SH0ES program")
            return None

    def query_by_pi(self, pi_name: str = "Riess"):
        """Query by Principal Investigator name (broader search)."""
        print(f"\n{'=' * 80}")
        print(f"QUERYING ALL OBSERVATIONS BY PI: {pi_name}")
        print(f"{'=' * 80}\n")

        try:
            obs = Observations.query_criteria(
                obs_collection="HST",
                pi_name=pi_name
            )

            print(f"✓ Found {len(obs)} observations by {pi_name}")

            # Filter for relevant instruments (WFC3, ACS for Cepheids/SNe)
            relevant_instruments = ['WFC3/UVIS', 'WFC3/IR', 'ACS/WFC', 'ACS/HRC']
            relevant_obs = obs[
                [inst in relevant_instruments for inst in obs['instrument_name']]
            ]

            print(f"✓ {len(relevant_obs)} with relevant instruments (WFC3/ACS)")

            return relevant_obs

        except Exception as e:
            print(f"✗ Error querying by PI: {e}")
            return None

    def get_available_products(self, observations):
        """Get data products for a set of observations."""
        print(f"\n{'=' * 80}")
        print("QUERYING AVAILABLE DATA PRODUCTS")
        print(f"{'=' * 80}\n")

        try:
            products = Observations.get_product_list(observations)
            print(f"✓ Found {len(products)} data products")

            # Filter for science data
            science_products = products[products['productType'] == 'SCIENCE']
            print(f"✓ {len(science_products)} science products")

            # Show product types
            print(f"\nProduct types available:")
            for ptype in set(products['productSubGroupDescription']):
                count = sum(products['productSubGroupDescription'] == ptype)
                print(f"  - {ptype}: {count}")

            return products

        except Exception as e:
            print(f"✗ Error getting products: {e}")
            return None

    def save_observation_summary(self, observations, filename: str = "shoes_observations_summary.json"):
        """Save observation metadata to JSON."""
        if observations is None or len(observations) == 0:
            print("⚠ No observations to save")
            return

        # Convert to pandas for easier manipulation
        df = observations.to_pandas()

        summary = {
            'query_date': datetime.now().isoformat(),
            'total_observations': len(df),
            'programs': list(self.shoes_programs.keys()),
            'instruments': list(df['instrument_name'].unique()),
            'targets': {
                'total': len(df['target_name'].unique()),
                'sample': list(df['target_name'].unique()[:20])  # First 20
            },
            'filters': list(df['filters'].unique()) if 'filters' in df.columns else [],
            'date_range': {
                'start': str(df['t_min'].min()) if 't_min' in df.columns else None,
                'end': str(df['t_max'].max()) if 't_max' in df.columns else None
            }
        }

        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✓ Saved summary: {output_path}")

        # Also save full table as CSV
        csv_path = self.output_dir / "shoes_observations_full.csv"
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved full table: {csv_path}")

    def query_cepheid_hosts(self):
        """Query specific Cepheid host galaxies used in SH0ES distance ladder."""
        print(f"\n{'=' * 80}")
        print("QUERYING CEPHEID HOST GALAXIES")
        print(f"{'=' * 80}\n")

        # Key SH0ES Cepheid hosts
        hosts = [
            'NGC-4258',  # Megamaser anchor
            'NGC-4536', 'NGC-4639', 'NGC-3370', 'NGC-5584',  # SN Ia hosts
            'NGC-3021', 'NGC-3982', 'NGC-1309', 'NGC-2442',
            'M101', 'NGC-3972', 'NGC-1015', 'UGC-9391'
        ]

        all_host_obs = []

        for host in hosts:
            try:
                obs = Observations.query_criteria(
                    target_name=host,
                    obs_collection="HST",
                    instrument_name="WFC3/UVIS"
                )

                if len(obs) > 0:
                    print(f"  ✓ {host}: {len(obs)} WFC3 observations")
                    all_host_obs.append(obs)
                else:
                    print(f"  - {host}: No WFC3 data")

            except Exception as e:
                print(f"  ✗ {host}: Error - {e}")

        if all_host_obs:
            from astropy.table import vstack
            combined = vstack(all_host_obs)
            print(f"\n✓ Total: {len(combined)} observations of {len(hosts)} Cepheid hosts")
            return combined
        else:
            return None

    def download_sample_data(self, observations, max_files: int = 10):
        """Download a sample of data products (for testing)."""
        print(f"\n{'=' * 80}")
        print(f"DOWNLOADING SAMPLE DATA (max {max_files} files)")
        print(f"{'=' * 80}\n")

        try:
            products = Observations.get_product_list(observations[:5])  # First 5 observations

            # Filter for calibrated images only
            cal_products = products[
                (products['productType'] == 'SCIENCE') &
                (products['productSubGroupDescription'].isin(['FLC', 'FLT', 'DRZ']))
            ]

            if len(cal_products) == 0:
                print("⚠ No calibrated products found")
                return

            # Download first N
            download_products = cal_products[:max_files]

            print(f"Downloading {len(download_products)} calibrated science products...")
            manifest = Observations.download_products(
                download_products,
                download_dir=str(self.output_dir / "downloads")
            )

            print(f"\n✓ Downloaded {len(manifest)} files")
            print(f"✓ Location: {self.output_dir / 'downloads'}")

        except Exception as e:
            print(f"✗ Download error: {e}")


def main():
    """Main execution."""
    print("=" * 80)
    print("MAST QUERY FOR SH0ES PROGRAM DATA")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    query = SHOESDataQuery(output_dir="./shoes_data")

    # Method 1: Query by known program IDs
    print("\n[METHOD 1] Querying by SH0ES program IDs...")
    obs_programs = query.query_shoes_programs()

    # Method 2: Query by PI name (broader)
    print("\n[METHOD 2] Querying by PI name (Riess)...")
    obs_pi = query.query_by_pi("Riess")

    # Method 3: Query specific Cepheid hosts
    print("\n[METHOD 3] Querying Cepheid host galaxies...")
    obs_hosts = query.query_cepheid_hosts()

    # Save results
    if obs_programs is not None:
        query.save_observation_summary(obs_programs, "shoes_programs_summary.json")

    if obs_pi is not None:
        query.save_observation_summary(obs_pi, "shoes_pi_summary.json")

    if obs_hosts is not None:
        query.save_observation_summary(obs_hosts, "shoes_cepheid_hosts_summary.json")

    # Ask about downloading
    print("\n" + "=" * 80)
    print("DATA DOWNLOAD")
    print("=" * 80)
    print("\nTo download actual data products, use:")
    print("  query.download_sample_data(observations, max_files=10)")
    print("\nNote: HST data files are large (100s of MB each)")
    print("Recommendation: Query first, then download specific datasets")

    print("\n" + "=" * 80)
    print("QUERY COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
