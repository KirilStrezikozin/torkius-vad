import soundata
from pathlib import Path
from time import time

# SAVE_PATH = Path("../data/sets/")

import urllib.request


def soundata_loader(dataset_name: str, *args, **kwargs) -> None:
    dataset = soundata.initialize(dataset_name)
    dataset.download(*args, **kwargs)


if __name__ == "__main__":
    datasets = [
        ("urbansound8k", (), {}, soundata_loader),
    ]

    for dataset_name, args, kwargs, loader in datasets:
        print(f"Downloading {dataset_name} dataset...")
        t0 = time()
        loader(dataset_name, *args, **kwargs)
        t1 = time()
        print(f"Downloaded {dataset_name} dataset in {t1 - t0:.3f}s.")
