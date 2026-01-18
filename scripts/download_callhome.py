from datasets import load_dataset
from pathlib import Path
import soundfile as sf

# Load German CallHome
dataset = load_dataset("diarizers-community/callhome", "deu", split="data")

out_dir = Path("callhome_deu_audio")
out_dir.mkdir(parents=True, exist_ok=True)

for i, sample in enumerate(dataset):
    audio = sample["audio"]

    audio_array = audio["array"]
    sample_rate = audio["sampling_rate"]

    out_path = out_dir / f"callhome_deu_{i:05d}.wav"
    sf.write(out_path, audio_array, sample_rate)

print("Done. Saved audio to:", out_dir)
