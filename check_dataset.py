import os

BASE = "dataset"

for split in ["train", "val"]:
    print(f"\n📂 {split.upper()} CLASSES:")
    path = os.path.join(BASE, split)
    for cls in sorted(os.listdir(path)):
        print("  ", cls)
