import os

DATASET_DIR = "original_dataset"

for folder in os.listdir(DATASET_DIR):
    old_path = os.path.join(DATASET_DIR, folder)

    if not os.path.isdir(old_path):
        continue

    # Replace double underscore with triple underscore
    if "__" in folder and "___" not in folder:
        new_name = folder.replace("__", "___")
        new_path = os.path.join(DATASET_DIR, new_name)

        os.rename(old_path, new_path)
        print(f"✅ Renamed: {folder} → {new_name}")
