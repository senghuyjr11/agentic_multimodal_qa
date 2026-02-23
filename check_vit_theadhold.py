import os
import shutil

SRC_DIR = "ImageNet-Mini/images"
DST_DIR = "dataset_reject/images"
os.makedirs(DST_DIR, exist_ok=True)

# Start count after existing reject images
existing = len(os.listdir(DST_DIR))
print(f"Existing reject images: {existing}")

count = 0
for cls in os.listdir(SRC_DIR):
    cls_path = os.path.join(SRC_DIR, cls)
    if not os.path.isdir(cls_path):
        continue
    for fname in os.listdir(cls_path):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            src = os.path.join(cls_path, fname)
            dst = os.path.join(DST_DIR, f"reject_{existing + count:04d}.jpg")
            shutil.copy(src, dst)
            count += 1

print(f"✓ Added {count} images")
print(f"Total reject images: {existing + count}")