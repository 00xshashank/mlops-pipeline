import random
import shutil
from pathlib import Path

RAW_DIR = Path("./data/raw/IMG_CLASSES")
OUTPUT_DIR = Path("./data")

IMAGES_PER_CLASS = 1250
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def get_images(class_dir):
    return [
        p for p in class_dir.iterdir()
        if p.suffix.lower() in VALID_EXTENSIONS and p.is_file()
    ]

def main():
    random.seed(RANDOM_SEED)

    class_dirs = [d for d in RAW_DIR.iterdir() if d.is_dir()]
    assert len(class_dirs) == 10, f"Expected 10 class folders, found {len(class_dirs)}"

    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)

    for class_dir in class_dirs:
        class_name = class_dir.name
        images = get_images(class_dir)

        if len(images) < IMAGES_PER_CLASS:
            raise ValueError(
                f"Class '{class_name}' has only {len(images)} images, "
                f"but {IMAGES_PER_CLASS} are required."
            )

        sampled_images = random.sample(images, IMAGES_PER_CLASS)

        n_train = int(IMAGES_PER_CLASS * TRAIN_RATIO)
        n_val = int(IMAGES_PER_CLASS * VAL_RATIO)

        train_imgs = sampled_images[:n_train]
        val_imgs = sampled_images[n_train:n_train + n_val]
        test_imgs = sampled_images[n_train + n_val:]

        splits = {
            "train": train_imgs,
            "val": val_imgs,
            "test": test_imgs,
        }

        for split, imgs in splits.items():
            split_class_dir = OUTPUT_DIR / split / class_name
            split_class_dir.mkdir(parents=True, exist_ok=True)

            for img in imgs:
                shutil.copy2(img, split_class_dir / img.name)

        print(
            f"{class_name}: "
            f"train={len(train_imgs)}, "
            f"val={len(val_imgs)}, "
            f"test={len(test_imgs)}"
        )

    print("\nDataset preparation complete.")

if __name__ == "__main__":
    main()
