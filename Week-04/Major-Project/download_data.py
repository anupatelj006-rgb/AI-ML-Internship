"""Download TrashNet and arrange the 5 classes into data/<class>/ (skips the 'trash' class).

    python download_data.py
"""
import shutil
import sys
import urllib.request
import zipfile

import config

URL = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"


def main():
    zip_path = config.ROOT / "trashnet.zip"
    tmp = config.ROOT / "_trashnet_tmp"
    try:
        print("Downloading TrashNet (about 40 MB)...")
        urllib.request.urlretrieve(URL, zip_path)
    except Exception as e:
        print(f"Download failed: {e}")
        print("Download dataset-resized.zip manually from https://github.com/garythung/trashnet (data folder),")
        print("unzip it, and copy the cardboard, glass, metal, paper and plastic folders into data/.")
        sys.exit(1)

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp)
    src = next((p for p in tmp.rglob("cardboard") if p.is_dir()), None)
    if src is None:
        print("Could not find the class folders inside the zip. Copy them into data/ manually.")
        sys.exit(1)

    for cls in config.CLASSES:
        dest = config.DATA_DIR / cls
        dest.mkdir(parents=True, exist_ok=True)
        for f in (src.parent / cls).iterdir():
            if f.is_file():
                shutil.copy2(f, dest / f.name)

    shutil.rmtree(tmp, ignore_errors=True)
    zip_path.unlink(missing_ok=True)
    for cls in config.CLASSES:
        n = sum(1 for f in (config.DATA_DIR / cls).iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
        print(f"{cls}: {n} images")
    print("Done. Next: python main.py")


if __name__ == "__main__":
    main()