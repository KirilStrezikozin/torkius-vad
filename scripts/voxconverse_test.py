from datasets import load_dataset
from pathlib import Path
import shutil

# Pick splits you want (e.g., "dev" and "test")
splits = [
    # "dev",
    "test",
]

output_base = Path("voxconverse_audio")
output_base.mkdir(exist_ok=True)

for split in splits:
    print(f"Processing split: {split}")

    # streaming=False to download files locally in cache
    ds = load_dataset("diarizers-community/voxconverse", split=split)

    split_dir = output_base / split
    split_dir.mkdir(exist_ok=True)

    for i, example in enumerate(ds):
        # cached audio file path
        audio_path = example["audio"]["path"]

        # copy out
        dst = split_dir / f"{i:05d}_{Path(audio_path).name}"
        shutil.copy(audio_path, dst)

    print(f"Copied {len(ds)} audio files to {split_dir}")
