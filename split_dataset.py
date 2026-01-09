import os
import shutil
import random

SOURCE_DIR = "original_dataset"
OUT_DIR = "dataset"
TRAIN_SPLIT = 0.8

random.seed(42)

def is_image(f):
    return f.lower().endswith((".jpg", ".png", ".jpeg"))

print("🔹 Processing dataset...")

for folder in os.listdir(SOURCE_DIR):
    folder_path = os.path.join(SOURCE_DIR, folder)

    if not os.path.isdir(folder_path):
        continue

    # skip junk folders
    if "___" not in folder:
        print(f"⚠ Skipped: {folder}")
        continue

    plant, disease = folder.split("___", 1)
#eg :Apple_apple_scab apple-plant , apple_scab- disease
    images = [f for f in os.listdir(folder_path) if is_image(f)]
    if len(images) == 0:
        continue
#splitting the images
    random.shuffle(images)
    split = int(len(images) * TRAIN_SPLIT)

    train_imgs = images[:split]
    val_imgs = images[split:]
#looping through train and val, makeing a directory for output
    for split_name, img_list in [("train", train_imgs), ("val", val_imgs)]:
        out_dir = os.path.join(OUT_DIR, split_name, folder)
        os.makedirs(out_dir, exist_ok=True)
#defining source and destination paths
        for img in img_list:
            src = os.path.join(folder_path, img)
            dst = os.path.join(out_dir, img)
#if the image does not exist in dst folder copy and put it there
            if not os.path.exists(dst):
                shutil.copy(src, dst)

print("\n✅ Dataset split SUCCESSFUL")
