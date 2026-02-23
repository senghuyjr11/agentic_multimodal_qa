from datasets import load_dataset

ds = load_dataset("timm/mini-imagenet", split="train", streaming=True)

# Check first item structure
item = next(iter(ds))
print(item.keys())
print("label:", item.get("label") or item.get("cls"))

# Print first 20 labels to see what classes exist
count = 0
labels = set()
for item in ds:
    labels.add(item.get("label") or item.get("cls"))
    count += 1
    if count >= 5000:
        break

print(f"Sample of classes found: {list(labels)[:30]}")