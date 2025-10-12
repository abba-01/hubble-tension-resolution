#!/usr/bin/env python3
"""
VizieR Query for Riess et al. 2022 (J/ApJ/934/L7) Tables

Downloads and parses all published tables from the VizieR catalog for
Riess et al. 2022 ApJ paper on the Hubble constant measurement.

Author: Eric D. Martin
Date: 2025-10-12
"""

from astroquery.vizier import Vizier
from astropy.io import ascii
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

class VizieRRiess2022:
    """Query and download VizieR catalog J/ApJ/934/L7."""

    def __init__(self, output_dir: str = "./vizier_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.catalog_id = "J/ApJ/934/L7"

        # Configure Vizier to get all rows
        Vizier.ROW_LIMIT = -1  # No limit

    def list_available_tables(self):
        """List all tables in the catalog."""
        print("=" * 80)
        print(f"VIZIER CATALOG: {self.catalog_id}")
        print("=" * 80)
        print("\nQuerying catalog structure...\n")

        try:
            # Get catalog metadata
            catalog = Vizier.get_catalogs(self.catalog_id)

            print(f"✓ Found {len(catalog)} table(s) in catalog\n")

            table_info = []
            for i, table in enumerate(catalog):
                info = {
                    'index': i,
                    'name': table.meta.get('name', 'Unknown'),
                    'description': table.meta.get('description', 'No description'),
                    'nrows': len(table),
                    'columns': list(table.colnames)
                }
                table_info.append(info)

                print(f"[Table {i+1}] {info['name']}")
                print(f"  Description: {info['description']}")
                print(f"  Rows: {info['nrows']}")
                print(f"  Columns ({len(info['columns'])}): {', '.join(info['columns'][:10])}")
                if len(info['columns']) > 10:
                    print(f"              ... and {len(info['columns']) - 10} more")
                print()

            return catalog, table_info

        except Exception as e:
            print(f"✗ Error querying catalog: {e}")
            return None, None

    def download_all_tables(self, catalog):
        """Download all tables to CSV files."""
        print("=" * 80)
        print("DOWNLOADING TABLES")
        print("=" * 80)

        if catalog is None:
            print("✗ No catalog provided")
            return []

        downloaded = []

        for i, table in enumerate(catalog):
            try:
                table_name = table.meta.get('name', f'table_{i+1}')

                # Clean filename
                safe_name = table_name.replace('/', '_').replace(' ', '_')
                csv_path = self.output_dir / f"{safe_name}.csv"

                # Convert to pandas and save
                df = table.to_pandas()
                df.to_csv(csv_path, index=False)

                print(f"\n✓ Table {i+1}: {table_name}")
                print(f"  Saved: {csv_path}")
                print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                print(f"  Size: {csv_path.stat().st_size / 1024:.1f} KB")

                # Save metadata
                meta = {
                    'table_name': table_name,
                    'description': table.meta.get('description', ''),
                    'nrows': len(df),
                    'ncols': len(df.columns),
                    'columns': list(df.columns),
                    'file': str(csv_path.name)
                }

                # Show first few rows
                print(f"\n  Preview (first 3 rows):")
                print(df.head(3).to_string(index=False))

                downloaded.append(meta)

            except Exception as e:
                print(f"\n✗ Error downloading table {i+1}: {e}")

        return downloaded

    def analyze_tables(self, table_info):
        """Analyze tables for relevant data."""
        print("\n" + "=" * 80)
        print("TABLE ANALYSIS")
        print("=" * 80)

        # Look for key information
        keywords = {
            'distance': ['dist', 'modulus', 'mu', 'dm'],
            'hubble': ['h0', 'h_0', 'hubble'],
            'cepheid': ['ceph', 'period', 'pl'],
            'uncertainty': ['err', 'unc', 'sigma', 'e_', 'cov', 'corr'],
            'galaxy': ['gal', 'host', 'ngc', 'ic'],
            'anchor': ['anchor', 'lmc', 'n4258']
        }

        for info in table_info:
            print(f"\n[{info['name']}]")

            # Check columns for keywords
            cols_lower = [c.lower() for c in info['columns']]

            matches = {}
            for category, terms in keywords.items():
                matching_cols = []
                for col, col_lower in zip(info['columns'], cols_lower):
                    if any(term in col_lower for term in terms):
                        matching_cols.append(col)
                if matching_cols:
                    matches[category] = matching_cols

            if matches:
                print("  Relevant columns found:")
                for category, cols in matches.items():
                    print(f"    {category.upper()}: {', '.join(cols)}")
            else:
                print("  No obviously relevant columns for H0 analysis")

    def search_covariance_data(self, catalog):
        """Search for covariance or correlation matrices."""
        print("\n" + "=" * 80)
        print("SEARCHING FOR COVARIANCE/CORRELATION DATA")
        print("=" * 80)

        found_cov = False

        for i, table in enumerate(catalog):
            table_name = table.meta.get('name', f'table_{i+1}')

            # Check column names
            cols_lower = [c.lower() for c in table.colnames]

            cov_indicators = ['cov', 'corr', 'rho', 'correlation', 'covariance']
            has_cov = any(any(ind in col for ind in cov_indicators) for col in cols_lower)

            if has_cov:
                print(f"\n✓ Table {i+1}: {table_name}")
                print(f"  Contains potential covariance/correlation data")
                print(f"  Columns: {', '.join(table.colnames)}")
                found_cov = True

                # Show sample
                df = table.to_pandas()
                print(f"\n  Sample data:")
                print(df.head(5).to_string(index=False))

        if not found_cov:
            print("\n⚠ No explicit covariance/correlation matrices found in tables")
            print("  Covariance data may be:")
            print("    - In journal supplementary files (not VizieR)")
            print("    - Included as error columns (σ values)")
            print("    - Published in a different paper")

    def extract_h0_measurements(self, catalog):
        """Extract H0 measurements from tables."""
        print("\n" + "=" * 80)
        print("EXTRACTING H0 MEASUREMENTS")
        print("=" * 80)

        h0_data = []

        for i, table in enumerate(catalog):
            table_name = table.meta.get('name', f'table_{i+1}')
            df = table.to_pandas()

            # Look for H0 columns
            h0_cols = [c for c in df.columns if 'h0' in c.lower() or 'h_0' in c.lower()]

            if h0_cols:
                print(f"\n✓ Table {i+1}: {table_name}")
                print(f"  H0 columns: {', '.join(h0_cols)}")

                # Extract data
                for col in h0_cols:
                    if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                        values = df[col].dropna()
                        if len(values) > 0:
                            print(f"\n  {col}:")
                            print(f"    Mean: {values.mean():.2f}")
                            print(f"    Std: {values.std():.2f}")
                            print(f"    Min: {values.min():.2f}")
                            print(f"    Max: {values.max():.2f}")
                            print(f"    N: {len(values)}")

                            h0_data.append({
                                'table': table_name,
                                'column': col,
                                'mean': float(values.mean()),
                                'std': float(values.std()),
                                'n': int(len(values)),
                                'values': values.tolist()
                            })

        return h0_data

    def save_summary(self, table_info, h0_data, downloaded):
        """Save comprehensive summary."""
        summary = {
            'query_date': datetime.now().isoformat(),
            'catalog_id': self.catalog_id,
            'paper': {
                'authors': 'Riess et al.',
                'year': 2022,
                'journal': 'ApJ',
                'volume': 934,
                'page': 'L7',
                'doi': '10.3847/2041-8213/ac5c5b',
                'title': 'A Comprehensive Measurement of the Local Value of the Hubble Constant'
            },
            'tables': table_info,
            'h0_measurements': h0_data,
            'downloaded_files': downloaded,
            'urls': {
                'vizier': f'https://vizier.cds.unistra.fr/viz-bin/VizieR?-source={self.catalog_id}',
                'doi': 'https://doi.org/10.3847/2041-8213/ac5c5b'
            }
        }

        summary_path = self.output_dir / "vizier_catalog_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✓ Saved summary: {summary_path}")


def main():
    """Main execution."""
    print("=" * 80)
    print("VIZIER QUERY: RIESS ET AL. 2022 (J/ApJ/934/L7)")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    query = VizieRRiess2022(output_dir="./vizier_data")

    # List tables
    catalog, table_info = query.list_available_tables()

    if catalog is None:
        print("\n✗ Failed to retrieve catalog")
        return

    # Analyze
    query.analyze_tables(table_info)

    # Download all
    downloaded = query.download_all_tables(catalog)

    # Search for covariance
    query.search_covariance_data(catalog)

    # Extract H0
    h0_data = query.extract_h0_measurements(catalog)

    # Save summary
    query.save_summary(table_info, h0_data, downloaded)

    print("\n" + "=" * 80)
    print("QUERY COMPLETE")
    print("=" * 80)
    print(f"\nDownloaded {len(downloaded)} table(s)")
    print(f"Output directory: {query.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
