from pathlib import Path
from datasets import load_dataset, DownloadConfig

# 1) Pick a project-local cache (hidden folder is common)
cache_dir = Path("./.hf_cache").resolve()

# 2) Load the dataset with a project-local cache
ds = load_dataset(
    "flaviagiammarino/path-vqa",
    cache_dir=str(cache_dir),
    download_config=DownloadConfig(cache_dir=str(cache_dir)),
)

print("Cache dir:", cache_dir)
print("Train cache files:", ds["train"].cache_files)
