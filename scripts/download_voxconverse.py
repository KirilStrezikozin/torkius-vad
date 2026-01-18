from datasets import load_dataset
from pathlib import Path
import soundfile as sf

# Load German CallHome
dataset = load_dataset("diarizers-community/voxconverse", split="test")

out_dir = Path("voxconverse_test_audio")
out_dir.mkdir(parents=True, exist_ok=True)

for i, sample in enumerate(dataset):
    audio = sample["audio"]

    audio_array = audio["array"]
    sample_rate = audio["sampling_rate"]

    out_path = out_dir / f"voxconverse_test_{i:05d}.wav"
    sf.write(out_path, audio_array, sample_rate)

print("Done. Saved audio to:", out_dir)
