import soundata
from pathlib import Path
from time import time
from datasets import load_dataset
from huggingface_hub import snapshot_download

import urllib.request

SAVE_PATH = Path("../data/sets/")


def soundata_loader(dataset_name: str) -> None:
    dataset = soundata.initialize(dataset_name)
    dataset.download()


def huggingface_loader(dataset_name: str, *args, **kwargs) -> None:
    dataset = load_dataset(dataset_name, *args, **kwargs)
    # Force download by iterating through the dataset:
    for _ in dataset:
        pass


def huggingface_snapshot_loader(dataset_name: str, *args, **kwargs) -> None:
    snapshot_download(
        dataset_name, local_dir=SAVE_PATH / dataset_name, repo_type="dataset"
    )


if __name__ == "__main__":
    datasets = [
        # ("tau2021sse_nigens", (), {}, soundata_loader),
        # ("urbansound8k", (), {}, soundata_loader),
        # ("diarizers-community/callhome", (), {"name": "deu"}, huggingface_loader),
        # (
        #     "librispeech_asr",
        #     (),
        #     {"name": "clean", "split": "validation"},
        #     huggingface_loader,
        # ),
        (
            "diarizers-community/voxconverse",
            (),
            {"name": "default", "split": "test"},
            huggingface_loader,
        ),
        # ("nccratliri/vad-human-ava-speech", (), {}, huggingface_snapshot_loader),
        # ("ashraq/esc50", (), {}, huggingface_snapshot_loader),
    ]

    for dataset_name, args, kwargs, loader in datasets:
        try:
            print(f"Downloading {dataset_name} dataset...")
            t0 = time()
            loader(dataset_name, *args, **kwargs)
            t1 = time()
            print(
                f"Downloaded {dataset_name} dataset to {SAVE_PATH} in {t1 - t0:.3f}s."
            )
        except Exception as e:
            print(f"Failed to download {dataset_name} dataset. Error: {e}")
