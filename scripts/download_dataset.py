import kagglehub
import shutil
from pathlib import Path

DATASET_ID = "ismailpromus/skin-diseases-image-dataset"
DEST_ROOT = Path("./data")
RAW_DIR = DEST_ROOT / "raw"

def main():
    print("Downloading dataset from Kaggle...")
    downloaded_path = Path(kagglehub.dataset_download(DATASET_ID))

    print(f"Downloaded to: {downloaded_path}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for item in downloaded_path.iterdir():
        target = RAW_DIR / item.name

        if target.exists():
            print(f"Skipping existing: {target}")
            continue

        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)

    print(f"Dataset copied to: {RAW_DIR.resolve()}")
    print("Done.")

if __name__ == "__main__":
    main()
