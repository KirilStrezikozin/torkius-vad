from datasets import load_dataset
from pathlib import Path
import soundfile as sf
import os

# Load the MUSAN dataset
dataset = load_dataset("FluidInference/musan", split="train")

# Output folder
out_root = Path("musan_audio")
classes = ["speech", "music", "noise"]
for c in classes:
    (out_root / c).mkdir(parents=True, exist_ok=True)

# Iterate and save
for i, sample in enumerate(dataset):
    label = sample["label"]

    # Get audio array and sample rate
    audio_array = sample["audio"]["array"]
    sample_rate = sample["audio"]["sampling_rate"]

    (out_root / str(label)).mkdir(exist_ok=True)

    # Destination file
    filename = out_root / str(label) / f"{i:06d}.wav"

    # Save audio
    sf.write(filename, audio_array, sample_rate)

print("Done — audio files organized by class in musan_audio/")
