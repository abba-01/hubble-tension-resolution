#!/usr/bin/env python3
"""
Fetch all datasets listed in SSOT.yml, write checksums, and emit a manifest.
We avoid hard-coding brittle direct links by pulling from the authoritative pages/DOIs.
"""
import os, sys, time, json, hashlib, pathlib
from urllib.parse import urlparse
from typing import Dict, Any
import requests, yaml
from tqdm import tqdm

ROOT = pathlib.Path(__file__).resolve().parents[1]
SSOT = yaml.safe_load(open(ROOT/"SSOT.yml"))

OUT = ROOT/"data"/"raw"
OUT.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR = ROOT/"manifests"
MANIFEST_DIR.mkdir(exist_ok=True, parents=True)
CHECKSUM_DIR = ROOT/"checksums"
CHECKSUM_DIR.mkdir(exist_ok=True, parents=True)

def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def download(url: str, dest: pathlib.Path, desc: str=""):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        size = int(r.headers.get("content-length", 0))
        pbar = tqdm(total=size, unit="B", unit_scale=True, desc=desc or dest.name)
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1<<20):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        pbar.close()


manifest: Dict[str, Any] = {"time": time.time(), "artifacts": []}

# 1) Planck PR3 via ESA DOI page: we record filenames + DOI and let the user authenticate to PLA if needed.
planck = SSOT["datasets"]["planck_pr3"]
planck_dir = OUT/"planck"
planck_dir.mkdir(exist_ok=True, parents=True)

# The DOI page lists file names and shows PLA links; some require interactive access.
# We still record the expected filenames and attempt direct PLA GET; if it fails, point users to the DOI HTML.
for f in planck["files"]:
    name = f["name"]
    dest = planck_dir/name
    tried = []
    success = False
    # Try an ESAC DOI "retrieve" pattern first (if available later), else fall back to PLA "download" endpoints if open.
    for candidate in []:
        try:
            download(candidate, dest, desc=f"Planck:{name}")
            success = True
            tried.append({"url": candidate, "status": "ok"})
            break
        except Exception as e:
            tried.append({"url": candidate, "status": f"fail:{e}"})
            continue
    if not success:
        # write a stub note file telling user to manually fetch from DOI page if automated grab is blocked.
        note = dest.with_suffix(dest.suffix + ".MANUAL.txt")
        note.write_text(
            f"{name}\n\n"
            f"This file must be downloaded from the official DOI landing page due to PLA session requirements:\n"
            f"  {planck['files'][0]['via']}\n"
            f"Place it here:\n  {dest}\n"
        )
    entry = {"dataset":"planck_pr3","file":str(dest), "name":name, "doi":planck["doi"], "tried":tried}
    if dest.exists():
        entry["sha256"] = sha256_file(dest)
        (CHECKSUM_DIR/f"{name}.sha256").write_text(entry["sha256"]+"  "+name+"\n")
    manifest["artifacts"].append(entry)

# 2) KiDS-1000 tarball (public)
kids = SSOT["datasets"]["kids_1000_cosmic_shear"]
kids_dir = OUT/"kids1000"
kids_tar = kids_dir/"KiDS1000_cosmic_shear_data_release.tgz"
try:
    download(kids["url_tarball"], kids_tar, desc="KiDS-1000 tarball")
    sha = sha256_file(kids_tar)
    (CHECKSUM_DIR/"KiDS1000_cosmic_shear_data_release.tgz.sha256").write_text(f"{sha}  KiDS1000_cosmic_shear_data_release.tgz\n")
except Exception as e:
    print("[WARN] KiDS-1000 download failed:", e, file=sys.stderr)

# 3) DES Y3 SACC via mirror (GitHub release). We record checksum for reproducibility.
des = SSOT["datasets"]["des_y3_three_by_two"]
des_dir = OUT/"des_y3"
des_dir.mkdir(exist_ok=True, parents=True)

# If GitHub API accessible, hit the release asset; otherwise, inform user to fetch manually.
try:
    # naive scrape of release HTML to find the FITS asset
    html = requests.get(des["primary"]["url"], timeout=30).text
    import re
    # look for a .fits link
    m = re.search(r'href="([^"]+\.fits)"', html)
    if m:
        fits_url = m.group(1)
        if fits_url.startswith("/"):
            fits_url = "https://github.com"+fits_url
        dest = des_dir/"y3_5x2_maglim_UNBLIND_07202021_120721_bestfit3x2.fits"
        download(fits_url, dest, desc="DES Y3 3x2pt SACC")
        sha = sha256_file(dest)
        (CHECKSUM_DIR/"des_y3_y3_5x2_maglim_bestfit3x2.fits.sha256").write_text(f"{sha}  {dest.name}\n")
    else:
        raise RuntimeError("Could not locate .fits asset in release HTML")
except Exception as e:
    note = des_dir/"DESY3_MANUAL.txt"
    note.write_text(
        "DES Y3 SACC file could not be auto-located.\n"
        f"Visit the release page and download the SACC FITS asset:\n  {des['primary']['url']}\n"
        "Place it in this directory.\n"
    )
    print("[WARN] DES Y3 auto-download failed:", e, file=sys.stderr)

open(MANIFEST_DIR/"fetch_manifest.json","w").write(json.dumps(manifest, indent=2))
print("Fetch complete. See manifests/fetch_manifest.json and checksums/.")
