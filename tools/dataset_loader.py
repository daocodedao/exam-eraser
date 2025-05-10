import shutil
import random
from pathlib import Path

SOURCE_DIR = "" 
OUTPUT_DIR = ""                   
SPLIT_RATIO = 0.8                          # 80% train, 20% test
LABELS = {"Handwritten", "Printed"}  # Expected labels

random.seed(None) #use system time
source_path = Path(SOURCE_DIR)
output_path = Path(OUTPUT_DIR)

all_images = []
for file in source_path.iterdir():
    if file.is_file() and "_" in file.stem:
        try:
            label = file.stem.split("_")[-1]
            if label in LABELS:
                all_images.append((file, label))
        except IndexError:
            continue

# Shuffle and split
random.shuffle(all_images)
split_idx = int(len(all_images) * SPLIT_RATIO)
train_images = all_images[:split_idx]
test_images = all_images[split_idx:]

def distribute(images, split):
    for img_path, label in images:
        dest_dir = output_path / split / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(img_path, dest_dir / img_path.name)

distribute(train_images, "train")
distribute(test_images, "test")

print(f"{len(train_images)} images in train")
print(f"{len(test_images)} images in test")
