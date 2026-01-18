from datasets import load_dataset
from pathlib import Path
import shutil

# Split to download
split = "validation"

# Output folder
output_dir = Path("librispeech_val_clean_audio")
output_dir.mkdir(exist_ok=True)

# Load the dataset (audio is automatically downloaded)
dataset = load_dataset("librispeech_asr", "clean", split=split)

# Copy audio files from Hugging Face cache to your folder
for i, sample in enumerate(dataset):
    audio_path = sample["audio"]["path"]  # cached path
    dst = output_dir / f"{i:05d}.wav"  # rename sequentially
    shutil.copy(audio_path, dst)
