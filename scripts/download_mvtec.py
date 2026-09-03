"""Download and extract MVTec AD categories.

Usage:
    python scripts/download_mvtec.py bottle capsule pill
    python scripts/download_mvtec.py --all

Downloads go to a temp file under data/.downloads/ and are extracted into
data/<category>/{train,test,ground_truth}/. Source: mvtec.com/company/research/datasets/mvtec-ad
(license: CC BY-NC-SA 4.0, non-commercial research use).
"""
import argparse
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

# Direct download links scraped from the official MVTec AD downloads page
# (https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads).
CATEGORY_URLS = {
    "bottle": "https://www.mydrive.ch/shares/150452/132a93367fb17cdf968dfb5c4013f6e7/download/420937370-1629958698/bottle.tar.xz",
    "cable": "https://www.mydrive.ch/shares/150453/10f960e07fec2838b2cd512586633a32/download/420937413-1629958794/cable.tar.xz",
    "capsule": "https://www.mydrive.ch/shares/150454/e0ce6dd74eb150f46c0d98131b3703f2/download/420937454-1629958872/capsule.tar.xz",
    "carpet": "https://www.mydrive.ch/shares/150455/eac7fbce84d93a5094e13f391170eca4/download/420937484-1629959013/carpet.tar.xz",
    "grid": "https://www.mydrive.ch/shares/150456/bb0b2e3dc804ccb8b4485e01f8e4493b/download/420937487-1629959044/grid.tar.xz",
    "hazelnut": "https://www.mydrive.ch/shares/150457/51d49f65e84bdc100ff8035d7dd783ea/download/420937545-1629959162/hazelnut.tar.xz",
    "leather": "https://www.mydrive.ch/shares/150458/923030ce14e10a7d147e95d0f8885f6b/download/420937607-1629959262/leather.tar.xz",
    "metal_nut": "https://www.mydrive.ch/shares/150459/c68856a21dca589b0f8ff6d4ee0f18f4/download/420937637-1629959294/metal_nut.tar.xz",
    "pill": "https://www.mydrive.ch/shares/150460/d4f1c04da67034ccc8d8b5fa3e73f244/download/420938129-1629960351/pill.tar.xz",
    "screw": "https://www.mydrive.ch/shares/150461/242f454cc6385e5693c4fd4b94567d1e/download/420938130-1629960389/screw.tar.xz",
    "tile": "https://www.mydrive.ch/shares/150462/5479f0fdc97bc6fa16eab0cb0cf0109f/download/420938133-1629960456/tile.tar.xz",
    "toothbrush": "https://www.mydrive.ch/shares/150463/895a1a5fb84b958f07417784b060434b/download/420938134-1629960477/toothbrush.tar.xz",
    "transistor": "https://www.mydrive.ch/shares/150464/99b8ea0332438d64fcc2475ee73f9c29/download/420938166-1629960554/transistor.tar.xz",
    "wood": "https://www.mydrive.ch/shares/150465/d5b4115b720cdb54d217e75636e6e374/download/420938383-1629960649/wood.tar.xz",
    "zipper": "https://www.mydrive.ch/shares/150466/bd155b557520edaf692d9bfdb915c24a/download/420938385-1629960680/zipper.tar.xz",
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def download_and_extract(category: str) -> None:
    if category not in CATEGORY_URLS:
        raise ValueError(f"Unknown category '{category}'. Choices: {sorted(CATEGORY_URLS)}")

    target_dir = DATA_DIR / category
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"[skip] {category}: data/{category}/ already populated")
        return

    downloads_dir = DATA_DIR / ".downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    archive_path = downloads_dir / f"{category}.tar.xz"

    if not archive_path.exists():
        print(f"[download] {category} -> {archive_path}")
        urllib.request.urlretrieve(CATEGORY_URLS[category], archive_path)
    else:
        print(f"[reuse] {category}: found existing archive at {archive_path}")

    print(f"[extract] {category}")
    with tarfile.open(archive_path, "r:xz") as tar:
        tar.extractall(DATA_DIR)

    archive_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("categories", nargs="*", help="One or more MVTec AD category names")
    parser.add_argument("--all", action="store_true", help="Download all 15 categories")
    args = parser.parse_args()

    categories = list(CATEGORY_URLS) if args.all else args.categories
    if not categories:
        parser.error("Specify at least one category or use --all")

    for category in categories:
        download_and_extract(category)

    downloads_dir = DATA_DIR / ".downloads"
    if downloads_dir.exists() and not any(downloads_dir.iterdir()):
        shutil.rmtree(downloads_dir)


if __name__ == "__main__":
    sys.exit(main())
