from collections import defaultdict
from pathlib import Path

from safetensors import safe_open

models_dir = Path("models")
for sf_path in sorted(models_dir.glob("*.safetensors")):
    print("=" * 80)
    print(f"File: {sf_path.name} ({sf_path.stat().st_size / (1024**3):.2f} GB)")
    with safe_open(sf_path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        keys = list(f.keys())
        print(f"Total keys: {len(keys)}")
        if metadata:
            print(f"Metadata keys: {list(metadata.keys())}")
        prefix_counts = defaultdict(int)
        for k in keys:
            prefix = k.split(".", 1)[0]
            prefix_counts[prefix] += 1
        print("Top-level prefixes:", dict(sorted(prefix_counts.items())))
        print("Sample keys (first 15):")
        for k in keys[:15]:
            slice_info = f.get_slice(k)
            print(
                f"  {k}: shape={slice_info.get_shape()}, dtype={slice_info.get_dtype()}"
            )
        print("Sample keys (last 10):")
        for k in keys[-10:]:
            slice_info = f.get_slice(k)
            print(
                f"  {k}: shape={slice_info.get_shape()}, dtype={slice_info.get_dtype()}"
            )
