# %% [markdown]
"""
# Torkius VAD

Streaming, weakly-supervised voice activity detection (VAD) model for telephony
and audio processing applications. Robust enough to various acoustic conditions,
noise types, tones, music, and recording devices.

![img](./0133_boosting.png)
_(On the plot: blue is ideal, red is predicted)_

This notebook includes full pipelines for:
- Dataset metadata orchestration.
- Dataset audio loading.
- Pseudo-labelling (teaching).
- Feature extraction and context aggregation.
- Partial (online) and offline learning methods.
- Evaluation and metrics.
- Visualization of audio waveforms, VAD probabilities, and audio features.

The pipeline is designed to be modular and extensible, allowing for easy
experimentation with different datasets, teaching methods, features, and models.

This notebook presents the core base for the Torkius VAD project. It has not
reached its full potential yet, but it is a solid foundation for further development
and experimentation. Check the [GitHub repository](https://github.com/KirilStrezikozin/torkius-vad)
for the latest updates and code.
"""

# %% [markdown]
"""
## Background and motivation

This project aims to build and train a model to classify audio segments into speech
and non-speech categories, with freedom to additionally classify into speech/music+tones/rest
using heuristics.

## VAD pipeline strategy

Audio data is chunked into fixed-size segments (e.g., 10ms). Various temporal and
spectral features are then extracted from these segments. Features are accumulated
over a context window (e.g., 1s) to capture temporal dependencies. A machine learning
model is trained on these features to predict the probability of speech presence in each segment.

## Weakly-supervised learning approach

As there was no single VAD-ready dataset with labelled 10ms speech and non-speech segments available,
a weakly-supervised learning approach was adopted. A probability teacher (pseudo-labelling)
method was used to generate labels for the training data. The [Silero VAD](https://github.com/snakers4/silero-vad)
model was used as the primary teacher to provide initial pseudo-labels for the audio segments in
datasets containing speech and mix of speech and non-speech. For datasets containing only non-speech,
a simple heuristic teacher was implemented to label all segments as non-speech.

## Before training

Before training the model, the following steps take place:

1. **Dataset metadata orchestration**: The datasets are organized and their metadata is computed, including total hours, number of files, and size.
2. **Audio loading**: Audio files are loaded and re-sampled to a common sample rate if necessary.
3. **Probability teaching**: The Silero VAD model is used to generate pseudo-labels for the audio segments, which serve as the target probabilities for training.
4. **Feature extraction and context aggregation**: Various features are extracted from the audio segments, and context is aggregated to capture temporal dependencies.

Each stage optionally supports caching to disk to speed up subsequent runs and avoid redundant computations.

After these steps, the final feature vectors and true labels for 120h of audio data
from datasets occupy about 2GB of disk space. The datasets themselves occupy about 20GB of disk space.

## Training

The model was trained on a combination of speech, non-speech, and mix datasets.
The total duration of audio used for training is roughly 120h.

While the final set of features and model architecture were still being experimented with,
an online training pipeline with `SGDClassifier` from `sklearn` was implemented and used.

The reason for this is that the datasets are huge (20 GB) in size, which deems training
on the entire dataset at once infeasible. The online training pipeline allows for training
the model incrementally on batches of samples (per one audio file), which is more memory-efficient
and allows for faster iterations during development.

## Evaluation

The following models were trained and evaluated:
- `SGDClassifier` with different regularization settings and parameters and features.
- Ensemble of `SGDClassifier` models.
- `XGBClassifier` with different regularization settings and parameters.

The model was evaluated on a separate test set containing a mix of speech and non-speech audio files.
Evaluation metrics included accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC.

## Results and future work

The `SGDClassifier` model achieved an accuracy of 78% and recall of 75% on the test set.
The `XGBClassifier` model achieved an accuracy of 89% and recall of 89% on the test set.
ROC-AUC was 0.85 for the `SGDClassifier` and 0.96 for the `XGBClassifier`.

These are excellent results for this first iteration of the project. However, there is
still room for improvement and further experimentation with:

- Features: revisiting the feature set.
- Context window: finding the optimal smallest window size for the best performance.
- Custom thresholding heuristics for speech/music+tones/rest.
- Increasing the amount of training data and diversity of datasets for better generalization.
- Re-training on self-predicted labels for further boosting performance.

The trained model is quite robust to music and tones and detects speech in multiple languages
in noisy conditions well. It is not perfect yet and there are situations where it
fails and either predicts a low speech probability for speech segments or a high speech
probability for non-speech segments.
"""

# %% [markdown]
"""
### Dataset selection

Crossed-out datasets mainly complement the others and only make sense if future training with
increased dataset size is planned.

Non-speech:

1. ~[UrbanSound8K](https://soundata.readthedocs.io/en/latest/source/quick_reference.html): A dataset containing 8,732 labeled sound excerpts (<=4s) of urban sounds from 10 classes, such as air conditioner, car horn, children playing, dog bark, drilling, engine idling, gunshot, jackhammer, siren, and street music.~
2. [ESC-50](https://github.com/karolpiczak/ESC-50): A labeled collection of 2,000 environmental audio recordings (5s) organized into 50 classes, including animals, natural soundscapes, human non-speech sounds, interior/domestic sounds, and exterior/urban noises.
3. ~[TAU NIGENS SSE 2021](https://soundata.readthedocs.io/en/latest/source/quick_reference.html): Spatial sound-scene recordings, consisting of sound events of distinct categories in a variety of acoustical spaces, and from multiple source directions and distances, at varying signal-to-noise ratios (SNR) ranging from noiseless (30dB) to noisy (6dB) conditions.~

Speech (clean, noisy):

1. ~[LibriSpeech Clean](https://www.openslr.org/12): 100 hours of clean read English speech derived from audiobooks from the LibriVox project, suitable for training and evaluating speech recognition systems.~
2. [Callhome German](https://huggingface.co/datasets/talkbank/callhome): A dataset of telephone conversations in German.
3. [VoxConverse test](https://github.com/joonson/voxconverse): A dataset for speaker diarization in real-world scenarios, containing multi-speaker conversations with overlapping speech and background noise.

Speech and non-speech:

1. [AVA-Speech for VAD](https://huggingface.co/datasets/nccratliri/vad-human-ava-speech): AVA-Speech dataset customized for Human Speech Voice Activity Detection in WhisperSeg. The audio files were extracted from films.
2. [MUSAN](https://huggingface.co/datasets/FluidInference/musan): A corpus of music, speech, and noise recordings suitable for training and evaluating voice activity detection (VAD) systems.
3. Private telephony: A collection of telephony audio recordings containing both speech and non-speech segments, used for training and evaluating VAD systems in telecommunication applications.
"""

# %% [markdown]
"""
### References

- [Silero VAD Demonstration](https://thegradient.pub/one-voice-detector-to-rule-them-all/)
- [Weak supervision, Wikipedia](https://en.wikipedia.org/wiki/Weak_supervision)
- [Whisper datasets, Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)

"""

# %% [markdown]
"""
## Checklist of steps implemented in this notebook:

- [x] Interactive audio player.
- [x] Loading and re-sampling audio files with caching capability.
- [x] Probability teacher (pseudo-labelling) with Silero VAD, with caching capability.
- [x] Static and interactive plotting of audio waveforms and VAD probabilities.
- [x] Find and download datasets.
- [x] Compute dataset metadata and statistics.
- [x] Audio feature extraction with caching capability.
- [x] Feature context stacking and aggregation.
- [x] Model training pipeline.
- [x] Model evaluation pipeline.
"""

# %% [markdown]
"""
### Notebook imports.

Import of required libraries and modules to run cells in this notebook.
"""

# %%
def _configure_plotly_classic() -> None:
    import plotly.io as pio 

    pio.renderers.default = "notebook_connected"


_configure_plotly_classic()  # If plotly charts don't render.

# %%
import torch  # noqa: E402

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# %%
from abc import ABC, abstractmethod  # noqa: E402
from collections import deque  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from enum import StrEnum  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import (  # noqa: E402
    Any,
    Callable,
    Generator,
    Iterable,
    Literal,
    NamedTuple,
    Protocol,
    cast,
)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from IPython.display import display  # noqa: E402

# %%
!pip install git+https://github.com/KirilStrezikozin/torkius-vad.git
!pip install silero-vad

# %%
from torkius_vad.plotting.widgets import play  # noqa: E402

# %% [markdown]
"""
### Datasets metadata orchestration and handling.

This section includes classes and methods for handling datasets metadata,
including loading, caching metadata, and displaying statistics about the datasets.
The `DatasetsMeta` class is responsible for orchestrating the metadata of all datasets,
while the `DatasetMeta` class handles the metadata of individual datasets.

### Note on Python patterns used
This notebook makes heavy use of abstract base classes to define interfaces
for implementations. There are not always multiple interchangeable implementations
for each interface, but the pattern is enforced globally for consistency.

Immutable data classes are utilized to represent audio data and its associated metadata.

Generators and iterators are used for efficient data processing, especially when
dealing with large datasets that cannot fit into memory.
"""

# %%
class AbstractVisualizer(ABC):
    """
    Abstract visualizer for data that can display
    any kind of visualization.
    """

    @abstractmethod
    def show(self, *args, **kwargs) -> None: ...


# %%
class AbstractDatasetsMeta(ABC):
    """
    Abstract datasets metadata handler.
    """

    @property
    @abstractmethod
    def datasets_meta(self) -> pd.DataFrame: ...

    @property
    @abstractmethod
    def grouped_meta(self) -> pd.DataFrame: ...

    @property
    @abstractmethod
    def dataset_names(self) -> list: ...

    @property
    @abstractmethod
    def datasets_path(self) -> Path: ...

    @property
    @abstractmethod
    def datasets_meta_path(self) -> Path: ...


class DatasetsMeta(AbstractVisualizer, AbstractDatasetsMeta):
    default_path: Path = Path().resolve().parent / "data" / "sets"
    default_meta_path: Path = default_path / "datasets_meta_raw.csv"

    def __init__(
        self,
        *,
        datasets_path: Path = default_path,
        meta_path: Path = default_meta_path,
        use_disk_cache: bool = True,
        print_stats: bool = True,
    ) -> None:
        import pickle
        from time import time

        self._check_datasets_path(path=datasets_path)
        self._datasets_path = datasets_path

        if use_disk_cache:
            try:
                s0 = time()

                datasets_meta = pd.read_csv(
                    self._datasets_path / "datasets_meta.csv",
                )
                datasets_meta.set_index("Dataset Name", inplace=True)
                grouped_meta = pd.read_csv(
                    self._datasets_path / "grouped_datasets_meta.csv"
                )
                grouped_meta.set_index("Type", inplace=True)
                dataset_names = pickle.load(
                    open(self._datasets_path / "dataset_names_meta.pkl", "rb"),
                )

                self._datasets_meta_path = meta_path
                self._datasets_meta = datasets_meta
                self._grouped_meta = grouped_meta
                self._dataset_names = dataset_names

                s1 = time()
                if print_stats:
                    print(
                        f"Dataset statistics loaded from disk cache in {s1 - s0:.3f}s."
                    )
                return
            except (FileNotFoundError, pd.errors.EmptyDataError, EOFError):
                if print_stats:
                    print("Disk cache not found or invalid. Rebuilding metadata...")

        s0 = time()

        df = pd.read_csv(meta_path)
        self._check_meta_df(df=df)
        self._datasets_meta_path = meta_path

        (
            self._datasets_meta,
            self._grouped_meta,
            self._dataset_names,
        ) = self._build_meta(datasets_meta=df)

        s1 = time()
        if print_stats:
            print(f"Dataset statistics loaded and built in {s1 - s0:.3f}s.")

        if use_disk_cache:
            s0 = time()
            self._datasets_meta.to_csv(self._datasets_path / "datasets_meta.csv")
            self._grouped_meta.to_csv(self._datasets_path / "grouped_datasets_meta.csv")
            pickle.dump(
                self._dataset_names,
                open(self._datasets_path / "dataset_names_meta.pkl", "wb"),
            )
            s1 = time()
            if print_stats:
                print(f"Dataset statistics cached to disk in {s1 - s0:.3f}s.")

    @property
    def datasets_meta(self) -> pd.DataFrame:
        return self._datasets_meta

    @property
    def grouped_meta(self) -> pd.DataFrame:
        return self._grouped_meta

    @property
    def dataset_names(self) -> list:
        return self._dataset_names

    @property
    def datasets_path(self) -> Path:
        return self._datasets_path

    @property
    def datasets_meta_path(self) -> Path:
        return self._datasets_meta_path

    def _build_meta(
        self, *, datasets_meta: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, list]:
        datasets_meta = datasets_meta.round({"total_hours": 1})
        datasets_meta["total_size_gb"] = (datasets_meta["total_size_mb"] / 1024).round(
            1
        )
        datasets_meta.drop(columns=["total_seconds", "total_size_mb"], inplace=True)
        datasets_meta.rename(
            columns={
                "directory": "Dataset Name",
                "total_hours": "Total Hours",
                "total_files": "Total Audio Files",
                "total_size_gb": "Total Size (GB)",
            },
            inplace=True,
        )

        datasets_meta["Type"] = datasets_meta["Dataset Name"].str.extract(
            r"^(speech|nonspeech|mix)_"
        )

        dataset_names = datasets_meta["Dataset Name"].tolist()

        grouped = cast(
            pd.DataFrame,
            datasets_meta.dropna(subset=["Type"])
            .groupby("Type", as_index=True)
            .sum(numeric_only=True),
        )

        datasets_meta.loc["Total"] = datasets_meta.sum(numeric_only=True)
        datasets_meta.at["Total", "Dataset Name"] = "Total"
        datasets_meta.at["Total", "Type"] = len(dataset_names)

        datasets_meta["Total Audio Files"] = datasets_meta["Total Audio Files"].astype(
            int
        )

        datasets_meta.set_index("Dataset Name", inplace=True)

        return datasets_meta, grouped, dataset_names

    def _check_meta_df(self, *, df: pd.DataFrame) -> None:
        expected_columns = {
            "directory",
            "total_seconds",
            "total_hours",
            "total_files",
            "total_size_mb",
        }
        if set(df.columns) != expected_columns:
            raise ValueError(
                f"Invalid dataset metadata columns. "
                f"Expected: {expected_columns}, "
                f"Found: {set(df.columns)}.",
            )

    def _check_datasets_path(self, *, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"Datasets path '{path}' does not exist.",
            )
        if not path.is_dir():
            raise NotADirectoryError(
                f"Datasets path '{path}' is not a directory.",
            )

    def show(self, *, groups: bool = True) -> None:
        display(self._datasets_meta)
        if groups:
            display(self._grouped_meta)

# %% [markdown]
"""
### Display datasets metadata and statistics.

Shows the metadata and statistics of the datasets, including total hours,
number of files, and size of each dataset.
"""

# %%
datasets_meta = DatasetsMeta()
datasets_meta.show()

# %% [markdown]
"""
### Orchestration of individual dataset metadata.

This section includes the `DatasetMeta` class, which handles the metadata of individual datasets.
It loads the metadata of a specific dataset, either from disk cache or by building it from the audio files.
"""

# %%
class DatasetType(StrEnum):
    """
    Enum for dataset types.
    """
    SPEECH = "speech"
    NONSPEECH = "nonspeech"
    MIX = "mix"


class AbstractDatasetMeta(ABC):
    """
    Abstract dataset metadata handler.
    """

    @property
    @abstractmethod
    def dataset_name(self) -> str: ...

    @property
    @abstractmethod
    def dataset_meta(self) -> pd.DataFrame | pd.Series: ...

    @property
    @abstractmethod
    def dataset_path(self) -> Path: ...

    @property
    @abstractmethod
    def dataset_type(self) -> DatasetType: ...

    @abstractmethod
    def shuffled(self, *, random_state: int | None = None) -> "AbstractDatasetMeta": ...


class DatasetMeta(AbstractVisualizer, AbstractDatasetMeta):
    def __init__(
        self,
        *,
        dataset_name: str,
        datasets_meta: AbstractDatasetsMeta,
        use_disk_cache: bool = True,
        dataset_mask: Any | None = None,
        print_stats: bool = True,
    ) -> None:
        from time import time

        self._datasets_meta = datasets_meta
        self._use_disk_cache = use_disk_cache
        self._print_stats = print_stats

        self._check_dataset_name(dataset_name=dataset_name)
        self._dataset_name = dataset_name

        self._dataset_type = self._get_dataset_type(dataset_name=dataset_name)
        self._dataset_mask = dataset_mask

        dataset_path = self._datasets_meta.datasets_path / dataset_name
        self._check_dataset_path(path=dataset_path)
        self._dataset_path = dataset_path

        if self._use_disk_cache:
            try:
                s0 = time()
                self._dataset_meta = pd.read_csv(
                    self._dataset_path / "dataset_meta.csv",
                )
                s1 = time()
                if self._print_stats:
                    print(
                        f"Dataset '{dataset_name}' metadata loaded from disk "
                        f"cache in {s1 - s0:.3f}s.",
                    )
                return
            except (FileNotFoundError, pd.errors.EmptyDataError):
                if self._print_stats:
                    print(
                        f"Disk cache for dataset '{dataset_name}' not found or "
                        f"invalid. Rebuilding metadata...",
                    )

        s0 = time()
        self._dataset_meta = self._build_meta()
        s1 = time()

        if self._print_stats:
            print(f"Dataset '{dataset_name}' metadata built in {s1 - s0:.3f}s.")

        if use_disk_cache:
            s0 = time()
            self._dataset_meta.to_csv(
                self._dataset_path / "dataset_meta.csv", index=False
            )
            s1 = time()
            if self._print_stats:
                print(
                    f"Dataset '{dataset_name}' metadata cached to disk in "
                    f"{s1 - s0:.3f}s.",
                )

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def dataset_meta(self) -> pd.DataFrame | pd.Series:
        if self._dataset_mask is not None:
            return self._dataset_meta[self._dataset_mask]
        return self._dataset_meta

    @property
    def dataset_path(self) -> Path:
        return self._dataset_path

    @property
    def dataset_type(self) -> DatasetType:
        return self._dataset_type

    def shuffled(self, *, random_state: int | None = None) -> "DatasetMeta":
        _dataset_meta_shuffled = self._dataset_meta.sample(
            frac=1.0,
            random_state=random_state,
        ).reset_index(drop=True)

        dataset_meta = DatasetMeta(
            dataset_name=self._dataset_name,
            datasets_meta=self._datasets_meta,
            use_disk_cache=self._use_disk_cache,
            dataset_mask=self._dataset_mask,
            print_stats=self._print_stats,
        )

        dataset_meta._dataset_meta = _dataset_meta_shuffled
        return dataset_meta

    def _check_dataset_name(self, *, dataset_name: str) -> None:
        if dataset_name not in self._datasets_meta.dataset_names:
            raise ValueError(
                f"Dataset '{dataset_name}' not found in metadata. "
                f"Available datasets: "
                f"{self._datasets_meta.dataset_names}.",
            )

    def _get_dataset_type(self, *, dataset_name: str) -> DatasetType:
        if dataset_name.startswith("speech_"):
            return DatasetType.SPEECH
        elif dataset_name.startswith("nonspeech_"):
            return DatasetType.NONSPEECH
        elif dataset_name.startswith("mix_"):
            return DatasetType.MIX
        else:
            raise ValueError(
                f"Cannot determine dataset type from name '{dataset_name}'.",
            )

    def _build_meta(self) -> pd.DataFrame:
        paths: list[Path] = []

        for path in self._dataset_path.rglob("*"):
            if path.is_file() and path.suffix.lower() in {
                ".wav",
                ".flac",
                ".mp3",
                ".ogg",
                ".m4a",
                ".aac",
            }:
                paths.append(path)

        dataset_meta = pd.DataFrame(
            {
                "Slug": [str(p.relative_to(self._dataset_path.parent)) for p in paths],
                "Type": self._dataset_type,
                "File Size (MB)": [p.stat().st_size / (1024 * 1024) for p in paths],
            }
        )

        dataset_meta = dataset_meta.round({"file_size_mb": 1})
        dataset_meta.reset_index(drop=True, inplace=True)

        return dataset_meta

    def _check_dataset_path(self, *, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset path '{path}' does not exist.",
            )
        if not path.is_dir():
            raise NotADirectoryError(
                f"Dataset path '{path}' is not a directory.",
            )

    def show(self) -> None:
        display(self.dataset_meta)

    def show_player(self, *, random_n: int = 2) -> None:
        import random

        sample_slugs = random.sample(
            self._dataset_meta["Slug"].tolist(),
            k=min(random_n, len(self._dataset_meta)),
        )

        for slug in sample_slugs:
            file_path = self._datasets_meta.datasets_path / slug
            play(
                file_path.as_posix(),
                title=file_path.relative_to(
                    self._datasets_meta.datasets_path
                ).as_posix(),
            )


# %% [markdown]
"""
### Building metadata for individual datasets.

This section builds the metadata for each individual dataset by loading the audio files
and computing the necessary statistics. The metadata is then cached to disk for faster
loading in subsequent runs.
"""

# %%
dataset_metas: dict[str, DatasetMeta] = {}

for dataset_name in datasets_meta.dataset_names:
    dataset_meta = DatasetMeta(
        dataset_name=dataset_name,
        datasets_meta=datasets_meta,
        print_stats=True,
    )
    dataset_metas[dataset_name] = dataset_meta

print(f"Total dataset metas built: {len(dataset_metas)}.")
for name, meta in dataset_metas.items():
    print(f"- {name}: {len(meta.dataset_meta)} audio files.")

# %% [markdown]
"""
### Displaying metadata and several random audio samples

This section displays the metadata for each individual dataset and shows an interactive audio player
for several random audio samples from each dataset.
"""

# %%
import random  # noqa: E402

sample_dataset_names = random.sample(datasets_meta.dataset_names, k=3)
for dataset_name in sample_dataset_names:
    dataset_meta = dataset_metas[dataset_name]
    print(f"\nShowing metadata for '{dataset_name}' dataset:")
    dataset_meta.show()

    print("Showing audio player for 2 random samples:")
    dataset_meta.show_player(random_n=2)

# %% [markdown]
"""
### Audio data representation and processing pipeline.

This section defines the `AudioData` data class, which represents the audio data and its associated metadata.

The `AbstractAudioLoader` and `AbstractProbabilityTeacher` classes define the interfaces for loading audio data and teaching probabilities, respectively.

`AudioLoader` is an implementation of `AbstractAudioLoader` that loads audio files, converts them to mono, and resamples them to a target sample rate.

`NonSpeechProbabilityTeacher` is an implementation of `AbstractProbabilityTeacher` that generates pseudo-labels for non-speech audio segments.

`SileroProbabilityTeacher` is an implementation of `AbstractProbabilityTeacher` that uses the Silero VAD model to generate pseudo-labels for speech and non-speech audio segments.
"""

# %%
@dataclass(frozen=True)
class AudioData:
    file_path: str
    target_sr: int

    chunk_size: int
    """
    Size of audio chunk for which inference is made.
    """

    audio: np.ndarray | None = None
    sr: int | None = None

    taught_probas: np.ndarray | None = None

    feat_vectors: np.ndarray | None = None

    predicted_probas: np.ndarray | None = None

    def with_audio(self, *, audio: np.ndarray, sr: int) -> "AudioData":
        return AudioData(
            file_path=self.file_path,
            target_sr=self.target_sr,
            chunk_size=self.chunk_size,
            audio=audio,
            sr=sr,
            taught_probas=self.taught_probas,
        )

    def with_taught_probas(self, *, taught_probas: np.ndarray) -> "AudioData":
        return AudioData(
            file_path=self.file_path,
            target_sr=self.target_sr,
            chunk_size=self.chunk_size,
            audio=self.audio,
            sr=self.sr,
            taught_probas=taught_probas,
        )

    def with_feat_vectors(self, *, feat_vectors: np.ndarray) -> "AudioData":
        return AudioData(
            file_path=self.file_path,
            target_sr=self.target_sr,
            chunk_size=self.chunk_size,
            audio=self.audio,
            sr=self.sr,
            taught_probas=self.taught_probas,
            feat_vectors=feat_vectors,
        )

    def with_predicted_probas(self, *, predicted_probas: np.ndarray) -> "AudioData":
        return AudioData(
            file_path=self.file_path,
            target_sr=self.target_sr,
            chunk_size=self.chunk_size,
            audio=self.audio,
            sr=self.sr,
            taught_probas=self.taught_probas,
            feat_vectors=self.feat_vectors,
            predicted_probas=predicted_probas,
        )


class AbstractAudioLoader(ABC):
    @abstractmethod
    def load(self, *, audio_data: AudioData) -> AudioData: ...


class NoopAudioLoader(AbstractAudioLoader):
    def __init__(self, *, print_stats: bool = False) -> None:
        self._print_stats = print_stats

    def load(self, *, audio_data: AudioData) -> AudioData:
        if self._print_stats:
            print(
                f"NoopAudioLoader: Skipping loading for file '{audio_data.file_path}'.",
            )
        return audio_data.with_audio(
            audio=np.array([], dtype=np.float32),
            sr=audio_data.target_sr,
        )


class AudioLoader(AbstractAudioLoader):
    def __init__(self, *, print_stats: bool = False) -> None:
        self._print_stats = print_stats
        self._load_n = 0
        self._avg_load_time = 0.0

    def load(self, *, audio_data: AudioData) -> AudioData:
        from time import time

        import soundfile as sf

        file_path = audio_data.file_path
        target_sr = audio_data.target_sr

        s0 = time()
        audio, sample_rate = sf.read(file_path)

        # Convert to mono.
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        audio = audio.astype(np.float32)
        if sample_rate != target_sr:
            # Re-sample audio to target sample rate
            import librosa

            audio = librosa.resample(
                audio,
                orig_sr=sample_rate,
                target_sr=target_sr,
            )
            sample_rate = target_sr

        s1 = time()

        # Statistics.
        self._load_n += 1
        self._avg_load_time = (
            (self._load_n - 1) * self._avg_load_time + (s1 - s0)
        ) / self._load_n

        if self._print_stats:
            print(f"Loaded audio file '{file_path}' in {s1 - s0:.3f}s.")
            print(
                f"Average loading time over {self._load_n} runs: "
                f"{self._avg_load_time:.3f}s.",
            )

        return audio_data.with_audio(audio=audio, sr=sample_rate)


class AbstractProbabilityTeacher(ABC):
    @abstractmethod
    def teach(self, *, audio_data: AudioData) -> AudioData: ...


class NonSpeechProbabilityTeacher(AbstractProbabilityTeacher):
    def __init__(self, *, print_stats: bool = False) -> None:
        self._print_stats = print_stats
        self._avg_teach_n = 0
        self._avg_teach_time = 0.0

    def teach(self, *, audio_data: AudioData) -> AudioData:
        from time import time

        import numpy as np

        fmt_err = "Audio data must contain {} for teaching probabilities."
        if audio_data.audio is None:
            raise ValueError(fmt_err.format("audio samples"))
        elif audio_data.sr is None:
            raise ValueError(fmt_err.format("sampling rate"))

        s0 = time()
        num_chunks = len(audio_data.audio) // audio_data.chunk_size
        taught_probas = np.zeros(shape=(num_chunks,), dtype=np.float32)
        s1 = time()

        # Statistics.
        self._avg_teach_n += 1
        self._avg_teach_time = (
            (self._avg_teach_n - 1) * self._avg_teach_time + (s1 - s0)
        ) / self._avg_teach_n

        if self._print_stats:
            print(f"Teaching with NonSpeech completed in {s1 - s0:.3f}s.")
            print(
                f"Average teaching time over {self._avg_teach_n} runs: "
                f"{self._avg_teach_time:.3f}s.",
            )

        return audio_data.with_taught_probas(taught_probas=taught_probas)


class SileroProbabilityTeacher(AbstractProbabilityTeacher):
    """
    It is not safe to use a single instance of this teacher in multi-threaded
    environments as the Silero VAD model is stateful. Ensure you use separate
    independent instances of this class per thread or that the teacher is only
    used for a single audio file at a time.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        max_speech_duration_s: float = float("inf"),
        min_silence_duration_ms: int = 200,
        speech_pad_ms: int = 30,
        neg_threshold: float | None = None,
        min_silence_at_max_speech: int = 98,
        use_max_poss_sil_at_max_speech: bool = True,
        pad_end_chunk_offset: int = -400,
        print_stats: bool = False,
        print_init_stats: bool = False,
    ) -> None:
        """
        Args:
            threshold: Speech threshold. If model's current state is NON-SPEECH, values ABOVE this value are considered as SPEECH.
            min_speech_duration_ms: Minimum duration of speech chunks in milliseconds.
            max_speech_duration_s: Maximum duration of speech chunks in seconds.
            min_silence_duration_ms: Minimum duration of silence in milliseconds to separate speech chunks.
            speech_pad_ms: Padding in milliseconds to add to each side of the final speech chunks.
            neg_threshold: Negative threshold (noise or exit threshold). If model's current state is SPEECH, values BELOW this value are considered as NON-SPEECH.
            min_silence_at_max_speech: Minimum silence duration in ms used to avoid abrupt cuts when max_speech_duration_s is reached.
            use_max_poss_sil_at_max_speech: Whether to use the maximum possible silence at max_speech_duration_s or not. If not, the last silence is used.
            pad_end_chunk_offset: Offset in samples to pad the end of the audio chunk. Default is -400 samples to avoid trailing noise.
            print_stats: Whether to print statistics about the VAD processing.
            print_init_stats: Whether to print statistics about the VAD model initialization.
        """
        from time import time

        import torch

        self._threshold = threshold
        self._min_speech_duration_ms = min_speech_duration_ms
        self._max_speech_duration_s = max_speech_duration_s
        self._min_silence_duration_ms = min_silence_duration_ms
        self._speech_pad_ms = speech_pad_ms
        self._neg_threshold = neg_threshold
        self._min_silence_at_max_speech = min_silence_at_max_speech
        self._use_max_poss_sil_at_max_speech = use_max_poss_sil_at_max_speech
        self._pad_end_chunk_offset = pad_end_chunk_offset

        # Statistics.
        self._print_stats = print_stats
        self._avg_init_n = 0
        self._avg_init_time = 0.0
        self._avg_teach_n = 0
        self._avg_teach_time = 0.0

        s0 = time()
        self._model, self._utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
        )  # type: ignore
        (
            self._get_speech_timestamps,
            self._save_audio,
            self._read_audio,
            self._VADIterator,
            self._collect_chunks,
        ) = self._utils

        # Move model to appropriate device.
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")

        self._model.to(self.device)

        self._proba_steps = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        s1 = time()

        # Statistics.
        self._avg_init_n += 1
        self._avg_init_time = (
            (self._avg_init_n - 1) * self._avg_init_time + (s1 - s0)
        ) / self._avg_init_n

        if self._print_stats or print_init_stats:
            print(f"Silero VAD model and teacher loaded in {s1 - s0:.3f}s.")
            print(
                f"Average initialization time over {self._avg_init_n} runs: "
                f"{self._avg_init_time:.3f}s.",
            )

    def teach(self, *, audio_data: AudioData) -> AudioData:
        from time import time

        import numpy as np
        import torch
        from silero_vad import get_speech_timestamps

        fmt_err = "Audio data must contain {} for teaching probabilities."
        if audio_data.audio is None:
            raise ValueError(fmt_err.format("audio samples"))
        elif audio_data.sr is None:
            raise ValueError(fmt_err.format("sampling rate"))
        elif self._pad_end_chunk_offset % audio_data.chunk_size != 0:
            raise ValueError(
                "pad_end_chunk_offset must be multiple of chunk_size "
                f"({self._pad_end_chunk_offset} % {audio_data.chunk_size} != 0).",
            )

        s0 = time()
        audio = audio_data.audio
        audio_tensor = torch.from_numpy(audio).to(self.device)

        speech_ts = get_speech_timestamps(
            audio_tensor,
            self._model,
            threshold=self._threshold,
            sampling_rate=audio_data.sr,
            min_speech_duration_ms=self._min_speech_duration_ms,
            max_speech_duration_s=self._max_speech_duration_s,
            min_silence_duration_ms=self._min_silence_duration_ms,
            speech_pad_ms=self._speech_pad_ms,
            neg_threshold=self._neg_threshold,  # type: ignore
            min_silence_at_max_speech=self._min_silence_at_max_speech,
            use_max_poss_sil_at_max_speech=self._use_max_poss_sil_at_max_speech,
        )

        taught_probas = np.zeros(shape=(len(audio),), dtype=np.float32)
        for ts in speech_ts:
            t0, t1 = ts["start"], ts["end"]
            c0 = t0 // audio_data.chunk_size * audio_data.chunk_size
            c1 = t1 // audio_data.chunk_size * audio_data.chunk_size

            # Normalized offsets.
            offset0 = (t0 - c0) / audio_data.chunk_size
            offset1 = (t1 - c1) / audio_data.chunk_size

            # Snap to closest probability step.
            c0_proba = self._proba_steps[np.argmin(np.abs(self._proba_steps - offset0))]
            c1_proba = self._proba_steps[np.argmin(np.abs(self._proba_steps - offset1))]

            if c0_proba == 0.0:
                c0 += audio_data.chunk_size
            if c1_proba == 0.0:
                c1 -= audio_data.chunk_size

            c1 += self._pad_end_chunk_offset

            taught_probas[c0:c1] = 1.0

        num_chunks = len(audio) // audio_data.chunk_size
        taught_probas = taught_probas[: num_chunks * audio_data.chunk_size]
        taught_probas = taught_probas[:: audio_data.chunk_size]

        s1 = time()

        # Statistics.
        self._avg_teach_n += 1
        self._avg_teach_time = (
            (self._avg_teach_n - 1) * self._avg_teach_time + (s1 - s0)
        ) / self._avg_teach_n

        if self._print_stats:
            print(f"Teaching with Silero VAD completed in {s1 - s0:.3f}s.")
            print(
                f"Average teaching time over {self._avg_teach_n} runs: "
                f"{self._avg_teach_time:.3f}s.",
            )

        return audio_data.with_taught_probas(taught_probas=taught_probas)

class UnthresholdedSileroProbabilityTeacher(SileroProbabilityTeacher):
    def teach(self, *, audio_data: AudioData) -> AudioData:
        from time import time

        import numpy as np
        import torch
        from silero_vad import get_speech_timestamps

        fmt_err = "Audio data must contain {} for teaching probabilities."
        if audio_data.audio is None:
            raise ValueError(fmt_err.format("audio samples"))
        elif audio_data.sr is None:
            raise ValueError(fmt_err.format("sampling rate"))

        s0 = time()
        audio = audio_data.audio
        audio_tensor = torch.from_numpy(audio).to(self.device)

        speech_probs = []
        window_size_samples = 512 if audio_data.sr == 16000 else 256
        for current_start_sample in range(0, len(audio), window_size_samples):
            chunk = audio_tensor[current_start_sample: current_start_sample + window_size_samples]
            if len(chunk) < window_size_samples:
                break
            speech_prob = self._model(chunk, audio_data.sr).item()
            speech_probs.append(speech_prob)


        aligned_probs = []
        for i in range(0, len(audio), audio_data.chunk_size):
            frame_center = i + audio_data.chunk_size // 2
            silero_idx = frame_center // window_size_samples

            if silero_idx < len(speech_probs):
                aligned_probs.append(speech_probs[silero_idx])

        s1 = time()

        # Statistics.
        self._avg_teach_n += 1
        self._avg_teach_time = (
            (self._avg_teach_n - 1) * self._avg_teach_time + (s1 - s0)
        ) / self._avg_teach_n

        if self._print_stats:
            print(f"Teaching with Silero VAD completed in {s1 - s0:.3f}s.")
            print(
                f"Average teaching time over {self._avg_teach_n} runs: "
                f"{self._avg_teach_time:.3f}s.",
            )

        return audio_data.with_taught_probas(taught_probas=aligned_probs)


class MixAvaDatasetProbabilityTeacher(AbstractProbabilityTeacher):
    """
    An alternative to Silero VAD teacher that uses the Mix-AVA dataset
    annotations to generate pseudo-labels for speech and non-speech audio
    segments.

    Note: This teacher is specifically designed for the Mix-AVA dataset
    and expects the audio files to have corresponding JSON metadata files with
    "onset" and "offset" annotations for speech segments. It will generate
    binary pseudo-labels based on these annotations, snapping to the closest
    probability step (0.0, 0.5, 1.0) for the start and end of speech segments.
    """

    def __init__(
        self,
        *,
        mix_ava_dataset_meta: DatasetMeta,
        print_stats: bool = False,
    ) -> None:
        if mix_ava_dataset_meta.dataset_type != DatasetType.MIX:
            raise ValueError(
                "mix_ava_dataset_meta must be of type 'mix'. "
                f"Found: '{mix_ava_dataset_meta.dataset_type}'.",
            )
        elif mix_ava_dataset_meta.dataset_name != "mix_ava":
            raise ValueError(
                "mix_ava_dataset_meta must be for 'mix_ava' dataset. "
                f"Found: '{mix_ava_dataset_meta.dataset_name}'.",
            )

        self._mix_ava_dataset_meta = mix_ava_dataset_meta
        self._print_stats = print_stats
        self._avg_teach_n = 0
        self._avg_teach_time = 0.0

        self._proba_steps = np.array([0.0, 0.5, 1.0], dtype=np.float32)

    def teach(self, *, audio_data: AudioData) -> AudioData:
        import json
        from time import time

        import numpy as np

        fmt_err = "Audio data must contain {} for teaching probabilities."
        if audio_data.audio is None:
            raise ValueError(fmt_err.format("audio samples"))
        elif audio_data.sr is None:
            raise ValueError(fmt_err.format("sampling rate"))

        s0 = time()

        clip_meta_path = Path(audio_data.file_path).with_suffix(".json")
        if not clip_meta_path.exists():
            raise FileNotFoundError(
                f"Mix-AVA clip metadata file '{clip_meta_path}' not found.",
            )

        with open(clip_meta_path, "r", encoding="utf-8") as f:
            clip_meta = json.load(f)

        audio = audio_data.audio
        taught_probas = np.zeros(shape=(len(audio),), dtype=np.float32)

        for onset_s, offset_s in zip(clip_meta["onset"], clip_meta["offset"]):
            t0 = int(onset_s * audio_data.sr)
            t1 = int(offset_s * audio_data.sr)
            c0 = t0 // audio_data.chunk_size * audio_data.chunk_size
            c1 = t1 // audio_data.chunk_size * audio_data.chunk_size

            # Normalized offsets.
            offset0 = (t0 - c0) / audio_data.chunk_size
            offset1 = (t1 - c1) / audio_data.chunk_size

            # Snap to closest probability step.
            c0_proba = self._proba_steps[np.argmin(np.abs(self._proba_steps - offset0))]
            c1_proba = self._proba_steps[np.argmin(np.abs(self._proba_steps - offset1))]

            if c0_proba == 0.0:
                c0 += audio_data.chunk_size
            if c1_proba == 0.0:
                c1 -= audio_data.chunk_size

            taught_probas[c0:c1] = 1.0

        num_chunks = len(audio) // audio_data.chunk_size
        taught_probas = taught_probas[: num_chunks * audio_data.chunk_size]
        taught_probas = taught_probas[:: audio_data.chunk_size]

        s1 = time()

        # Statistics.
        self._avg_teach_n += 1
        self._avg_teach_time = (
            (self._avg_teach_n - 1) * self._avg_teach_time + (s1 - s0)
        ) / self._avg_teach_n

        if self._print_stats:
            print(f"Teaching with Mix Ava Dataset Teacher completed in {s1 - s0:.3f}s.")
            print(
                f"Average teaching time over {self._avg_teach_n} runs: "
                f"{self._avg_teach_time:.3f}s.",
            )

        return audio_data.with_taught_probas(taught_probas=taught_probas)

# %% [markdown]
"""
### Dataset orchestration

After implementing audio loaders and teachers for one audio file, the next
sections define orchestration classes that handle the loading and teaching of
audio data for entire datasets.
"""

# %% [markdown]
"""
### Dataset audio loading pipeline.

This section defines the `AbstractDatasetAudioLoader` and `DatasetAudioLoader` classes, which handle the loading of audio data for each dataset.

The `DatasetAudioLoader` class attempts to load pre-processed audio data from disk cache for faster loading. If the cache is not available or invalid, it builds the audio data from the source audio files using the provided `AbstractAudioLoader` implementation. The loaded audio data is then cached to disk for future runs if disk caching is enabled.

In practice, one would typically not use caching for audio loading, since
loading and re-sampling it is usually not a bottleneck and caching will consume
the same amount of disk space as the original audio files.
"""

# %%
class AbstractDatasetAudioLoader(ABC):
    @abstractmethod
    def load(
        self, *, dataset_meta: DatasetMeta
    ) -> Generator[AudioData, None, None]: ...


class DatasetAudioLoader(AbstractDatasetAudioLoader):
    default_cache_dir: Path = Path().resolve().parent / "data" / "processed"

    def __init__(
        self,
        *,
        audio_loader: AbstractAudioLoader,
        target_sr: int = 8000,
        chunk_size: int = int(0.01 * 8000),  # 10 ms chunks
        use_disk_cache: bool = True,
        cache_dir: Path = default_cache_dir,
        mmap_mode: Literal["r", "r+", "w+", "c"] | None = None,
        print_stats: bool = True,
    ) -> None:
        self._audio_loader = audio_loader
        self._target_sr = target_sr
        self._chunk_size = chunk_size
        self._use_disk_cache = use_disk_cache
        self._cache_dir = cache_dir
        self._mmap_mode: Literal["r", "r+", "w+", "c"] | None = mmap_mode
        self._print_stats = print_stats

    def load(self, *, dataset_meta: DatasetMeta) -> Generator[AudioData, None, None]:
        from time import time

        datasets_dir = dataset_meta.dataset_path.parent
        build = True

        s0 = time()
        if self._use_disk_cache:
            build = False

            if not self._cache_dir.exists():
                self._cache_dir.mkdir(parents=True, exist_ok=True)

            if self._print_stats:
                print(
                    f"Attempting to load {len(dataset_meta.dataset_meta)} audio files "
                    f"from dataset '{dataset_meta.dataset_name}' cache...",
                )

            for slug in dataset_meta.dataset_meta["Slug"]:
                cached_audio_path = self._cache_dir / Path(slug).with_suffix(".npy")
                if not cached_audio_path.exists():
                    build = True
                    if self._print_stats:
                        print(
                            f"Cache file '{cached_audio_path}' not found. "
                            f"Rebuilding audio data from source files...",
                        )
                    break

        took = time() - s0
        if self._use_disk_cache and not build:
            i = 0
            for slug in dataset_meta.dataset_meta["Slug"]:
                s0 = time()
                cached_audio_path = self._cache_dir / Path(slug).with_suffix(".npy")
                cached_audio = np.load(cached_audio_path, mmap_mode=self._mmap_mode)
                audio_data = AudioData(
                    file_path=(datasets_dir / slug).as_posix(),
                    target_sr=self._target_sr,
                    chunk_size=self._chunk_size,
                    audio=cached_audio,
                    sr=self._target_sr,
                )

                fmt_err = "Cached audio data must contain {}."
                if audio_data.audio is None:
                    raise ValueError(fmt_err.format("audio samples"))
                elif audio_data.sr != self._target_sr:
                    raise ValueError(
                        f"Cached audio data from file '{cached_audio_path}' "
                        f"has invalid sampling rate "
                        f"({audio_data.sr} != {self._target_sr}).",
                    )

                i += 1
                if self._print_stats:
                    print(
                        f"Loaded audio (CACHE): {slug} ({i}/{len(dataset_meta.dataset_meta)}) "
                        f"shape={audio_data.audio.shape}"
                    )

                took += time() - s0
                yield audio_data

            if not build and self._print_stats:
                print(
                    f"Loaded {i} audio files from dataset "
                    f"'{dataset_meta.dataset_name}' cache in {took:.3f}s."
                )

        if not build:
            return

        if self._print_stats:
            print(
                f"Loading {len(dataset_meta.dataset_meta)} audio files from dataset "
                f"'{dataset_meta.dataset_name}'...",
            )

        took = 0.0
        i = 0
        for slug in dataset_meta.dataset_meta["Slug"]:
            s0 = time()
            file_path = datasets_dir / slug
            audio_data = AudioData(
                file_path=file_path.as_posix(),
                target_sr=self._target_sr,
                chunk_size=self._chunk_size,
            )
            audio_data = self._audio_loader.load(audio_data=audio_data)

            if audio_data.audio is None:
                raise ValueError(
                    f"Loaded audio data from file '{file_path}' "
                    f"does not contain audio samples.",
                )
            elif audio_data.sr != self._target_sr:
                raise ValueError(
                    f"Loaded audio data from file '{file_path}' "
                    f"has invalid sampling rate "
                    f"({audio_data.sr} != {self._target_sr}).",
                )

            if self._use_disk_cache:
                cached_audio_path = self._cache_dir / Path(slug).with_suffix(".npy")
                if not cached_audio_path.parent.exists():
                    cached_audio_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cached_audio_path, audio_data.audio, allow_pickle=False)

            i += 1
            if self._print_stats:
                print(
                    f"Loaded audio: {slug} ({i}/{len(dataset_meta.dataset_meta)}) "
                    f"shape={audio_data.audio.shape}"
                )

            took += time() - s0
            yield audio_data

        if self._print_stats:
            print(
                f"Loaded {i} audio files from dataset "
                f"'{dataset_meta.dataset_name}' in {took:.3f}s."
            )

        return

# %% [markdown]
"""
### Dataset audio teaching pipeline.

This section defines the `AbstractDatasetAudioTeacher` and `DatasetAudioTeacher` classes, which handle the teaching of probabilities for each dataset.

The `DatasetAudioTeacher` class attempts to load pre-computed taught probabilities from disk cache for faster loading. If the cache is not available or invalid, it builds the taught probabilities from the source audio files using the provided `AbstractProbabilityTeacher` implementation. The taught probabilities are then cached to disk for future runs if disk caching is enabled.

The caching is quite useful here and avoids the need to run the expensive teaching
process.
"""

# %%
class AbstractDatasetAudioTeacher(ABC):
    @abstractmethod
    def teach(
        self,
        *,
        dataset_meta: DatasetMeta,
        audio_data_producer: Generator[AudioData, None, None],
    ) -> Generator[AudioData, None, None]: ...


class DatasetAudioTeacher(AbstractDatasetAudioTeacher):
    default_cache_dir: Path = Path().resolve().parent / "data" / "processed"

    def __init__(
        self,
        *,
        probability_teacher: AbstractProbabilityTeacher,
        use_disk_cache: bool = True,
        cache_dir: Path = default_cache_dir,
        mmap_mode: Literal["r", "r+", "w+", "c"] | None = None,
        fix_shape: bool = False,
        print_stats: bool = True,
    ) -> None:
        self._probability_teacher = probability_teacher
        self._use_disk_cache = use_disk_cache
        self._cache_dir = cache_dir
        self._mmap_mode: Literal["r", "r+", "w+", "c"] | None = mmap_mode
        self._fix_shape = fix_shape
        self._print_stats = print_stats

    def teach(
        self,
        *,
        dataset_meta: DatasetMeta,
        audio_data_producer: Generator[AudioData, None, None],
    ) -> Generator[AudioData, None, None]:
        from time import time

        datasets_dir = dataset_meta.dataset_path.parent
        build = True

        s0 = time()
        if self._use_disk_cache:
            build = False

            if not self._cache_dir.exists():
                self._cache_dir.mkdir(parents=True, exist_ok=True)

            if self._print_stats:
                print(
                    f"Attempting to load taught probabilities for "
                    f"{len(dataset_meta.dataset_meta)} audio files "
                    f"from dataset '{dataset_meta.dataset_name}' cache...",
                )

            for slug in dataset_meta.dataset_meta["Slug"]:
                slug_path = Path(slug)
                cached_probas_path = self._cache_dir / slug_path.with_stem(
                    f"{slug_path.stem}_probas"
                ).with_suffix(".npy")
                if not cached_probas_path.exists():
                    build = True
                    if self._print_stats:
                        print(
                            f"Cache file '{cached_probas_path}' not found. "
                            f"Rebuilding taught probabilities from source files...",
                        )
                    break

        took = time() - s0
        if self._use_disk_cache and not build:
            i = 0
            for audio_data in audio_data_producer:
                s0 = time()
                slug = Path(audio_data.file_path).relative_to(datasets_dir)
                cached_probas_path = self._cache_dir / slug.with_stem(
                    f"{slug.stem}_probas"
                ).with_suffix(".npy")

                cached_probas = np.load(cached_probas_path, mmap_mode=self._mmap_mode)
                taught_audio_data = audio_data.with_taught_probas(
                    taught_probas=cached_probas,
                )

                fmt_err = "Cached taught audio data must contain {}."
                if taught_audio_data.audio is None:
                    raise ValueError(fmt_err.format("audio samples"))
                elif taught_audio_data.taught_probas is None:
                    raise ValueError(fmt_err.format("taught probabilities"))

                i += 1
                if self._print_stats:
                    print(
                        f"Loaded taught (CACHE): {slug} ({i}/{len(dataset_meta.dataset_meta)})"
                        f" shape={taught_audio_data.taught_probas.shape}"
                    )

                num_chunks = (
                    len(taught_audio_data.audio) // taught_audio_data.chunk_size
                )
                if self._fix_shape and taught_audio_data.taught_probas.shape != (
                    num_chunks,
                ):
                    taught_audio_data = taught_audio_data.with_taught_probas(
                        taught_probas=taught_audio_data.taught_probas[:num_chunks]
                    )
                    assert taught_audio_data.taught_probas is not None

                    # Re-save adjusted probabilities to cache.
                    np.save(
                        cached_probas_path,
                        taught_audio_data.taught_probas,
                        allow_pickle=False,
                    )

                    # Always print adjustment info in cache loading mode.
                    print(
                        f"Adjusted cached taught probabilities shape for "
                        f"file '{cached_probas_path}' to "
                        f"({num_chunks},)."
                    )

                took += time() - s0
                yield taught_audio_data

            if not build and self._print_stats:
                print(
                    f"Loaded taught probabilities for "
                    f"{i} audio files in dataset "
                    f"'{dataset_meta.dataset_name}' cache in {took:.3f}s."
                )

        if not build:
            return

        if self._print_stats:
            print(
                f"Teaching probabilities for {len(dataset_meta.dataset_meta)} audio files "
                f"in dataset '{dataset_meta.dataset_name}'...",
            )

        took = 0.0
        i = 0
        for audio_data in audio_data_producer:
            s0 = time()
            taught_audio_data = self._probability_teacher.teach(audio_data=audio_data)
            if taught_audio_data.taught_probas is None:
                raise ValueError(
                    f"Taught audio data from file '{audio_data.file_path}' "
                    f"does not contain taught probabilities.",
                )

            if self._use_disk_cache:
                slug = Path(audio_data.file_path).relative_to(datasets_dir)
                cached_probas_path = self._cache_dir / slug.with_stem(
                    f"{slug.stem}_probas"
                ).with_suffix(".npy")
                if not cached_probas_path.parent.exists():
                    cached_probas_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(
                    cached_probas_path,
                    taught_audio_data.taught_probas,
                    allow_pickle=False,
                )

            i += 1
            if self._print_stats:
                print(
                    f"Taught: {audio_data.file_path} ({i}/{len(dataset_meta.dataset_meta)}) "
                    f"shape={taught_audio_data.taught_probas.shape}"
                )

            took += time() - s0
            yield taught_audio_data

        if self._print_stats:
            print(
                f"Taught probabilities for {i} audio files "
                f"in dataset '{dataset_meta.dataset_name}' in {took:.3f}s."
            )

        return

# %% [markdown]
"""
### Audio frame generation and feature extraction.

The next step in the pipeline is to generate audio frames from the loaded audio data and extract features from those frames. The following classes handle these steps:
- `AbstractAudioFrameGenerator` and `AudioFrameGenerator`: These classes define the interface and implementation for generating audio frames from the loaded audio data. The `AudioFrameGenerator` class divides the audio samples into non-overlapping frames based on the specified chunk size.
- `AbstractAudioFeatureExtractor` and `AudioFeatureExtractor`: These classes define the interface and implementation for extracting features from the generated audio frames.

The `AudioFeatureExtractor` class applies a Hamming window to each frame, computes the FFT spectrum, along with the following features:
- Zero-crossing rate (ZCR) - represents the rate at which the signal changes sign, which can indicate the presence of speech.
- Centroid - represents the "center of mass" of the spectrum, which can indicate the brightness of the sound.
- Tonality - represents the degree to which the sound is tonal (harmonic) versus noisy, which can help distinguish speech from noise.
- Peaks - represents the number of peaks in the spectrum, which can indicate the complexity of the sound.
- Flux - represents the amount of spectral change between consecutive frames, which can indicate the presence of speech.
- Dominant frequency ratio - represents the ratio of the dominant frequency to the total energy in the spectrum, which can indicate the presence of speech or pure tones.
- Log energy - represents the logarithm of the total energy in the frame, which can indicate the presence of speech.

There are MFCCs and Mel-spectrogram features as well, but they are currently disabled to save time and because they did not seem to improve the results in preliminary experiments. They can be easily re-enabled by uncommenting the relevant lines in the `_calc_feat_vec` method of the `AudioFeatureExtractor` class.

### Early normalization

The log energy and flux features are normalized during extraction, using
running mean and variance and `fahn` normalization. This is done because the
scale of these features varies significantly and is order of magnitude higher than
the other features, which makes training completely unstable, even with
fitted scalers.
"""

# %%
class AbstractAudioFrameGenerator(ABC):
    @abstractmethod
    def generate(
        self, *, audio_data: AudioData
    ) -> Generator[np.ndarray, None, None]: ...


class AudioFrameGenerator(AbstractAudioFrameGenerator):
    def generate(self, *, audio_data: AudioData) -> Generator[np.ndarray, None, None]:
        fmt_err = "Audio data must contain {} for frame generation."
        if audio_data.audio is None:
            raise ValueError(fmt_err.format("audio samples"))
        elif audio_data.sr is None:
            raise ValueError(fmt_err.format("sampling rate"))

        audio = audio_data.audio
        chunk_size = audio_data.chunk_size

        num_chunks = len(audio) // chunk_size
        print(f"Generating {num_chunks} frames from audio of shape {audio.shape}.")
        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size
            yield audio[start:end]


class AbstractAudioFeatureExtractor(ABC):
    @abstractmethod
    def extract(
        self,
        *,
        audio_data: AudioData,
        frame_generator: AbstractAudioFrameGenerator,
    ) -> AudioData: ...


class AudioFeatureExtractor(AbstractAudioFeatureExtractor):
    def __init__(
        self,
        hamming_window_size: int = 80,
        sr: int = 8000,
        n_fft: int = 80,
        n_mels: int = 12,
        eps: float = 1e-10,
        print_stats: bool = False,
    ) -> None:
        """
        Args:
            hamming_window_size: Size of the Hamming window to apply to each frame.
            n_fft: Number of FFT sample points.
            eps: Small value to avoid log(0) and division by zero.
            n_mfcc: Number of MFCC coefficients to extract.
            print_stats: Whether to print statistics about the feature extraction.
        """

        import librosa

        self._hamming_window_size = hamming_window_size
        self._window = np.hamming(self._hamming_window_size)
        self._n_fft = n_fft
        self._n_mels = n_mels
        self._n_mfcc = 13
        self._eps = eps
        self._print_stats = print_stats

        self._avg_extract_n = 0
        self._avg_extract_time = 0.0

        self._mel_filter_bank = librosa.filters.mel(
            sr=sr,
            n_fft=self._n_fft,
            n_mels=self._n_mels,
            fmin=50.0,
            fmax=sr / 2,
        ).astype(np.float32)

    def extract(
        self,
        *,
        audio_data: AudioData,
        frame_generator: AbstractAudioFrameGenerator,
    ) -> AudioData:
        from time import time

        fmt_err = "Audio data must contain {} for feature extraction."
        if audio_data.audio is None:
            raise ValueError(fmt_err.format("audio samples"))
        elif audio_data.sr is None:
            raise ValueError(fmt_err.format("sampling rate"))

        s0 = time()
        feat_vectors = []

        prev_fft_spectrum = None
        mean_energy = 0.0
        mean2_energy = 0.0
        mean_flux = 0.0
        mean2_flux = 0.0
        i = 1
        for frame in frame_generator.generate(audio_data=audio_data):
            (
                feat_vec,
                prev_fft_spectrum,
                mean_energy,
                mean2_energy,
                mean_flux,
                mean2_flux,
            ) = self._calc_feat_vec(
                frame=frame,
                sr=audio_data.sr,
                prev_fft_spectrum=prev_fft_spectrum,
                mean_energy=mean_energy,
                mean2_energy=mean2_energy,
                mean_flux=mean_flux,
                mean2_flux=mean2_flux,
                i=i,
            )
            feat_vectors.append(feat_vec)
            i += 1

        feat_vectors_array = np.vstack(feat_vectors).astype(np.float32)

        s1 = time()
        if self._print_stats:
            print(
                f"Extracted {len(feat_vectors_array)} feature vectors in {s1 - s0:.3f}s.",
            )

        # Statistics.
        self._avg_extract_n += 1
        self._avg_extract_time = (
            (self._avg_extract_n - 1) * self._avg_extract_time + (s1 - s0)
        ) / self._avg_extract_n

        if self._print_stats:
            print(
                f"Average feature extraction time over "
                f"{self._avg_extract_n} runs: "
                f"{self._avg_extract_time:.3f}s.",
            )

        return audio_data.with_feat_vectors(feat_vectors=feat_vectors_array)

    def _calc_feat_vec(
        self,
        *,
        frame: np.ndarray,
        sr: int,
        prev_fft_spectrum: np.ndarray | None,
        mean_energy: float,
        mean2_energy: float,
        mean_flux: float,
        mean2_flux: float,
        i: int,
    ) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
        from scipy.fft import rfft

        frame = frame * self._window

        log_energy = np.log(np.sum(frame**2) + self._eps)
        delta_energy = log_energy - mean_energy
        mean_energy += delta_energy / i
        mean2_energy += delta_energy * (log_energy - mean_energy)
        var_energy = mean2_energy / i
        std_energy = np.sqrt(var_energy)
        log_energy = np.tanh((log_energy - mean_energy) / (std_energy + self._eps))
        log_energy = (log_energy * 2) + 1.0

        zcr = np.mean(frame[:-1] * frame[1:] < 0.0)

        fft_spectrum = np.abs(rfft(frame, n=self._n_fft)) + self._eps

        if prev_fft_spectrum is None:
            flux = 0.0
        else:
            diff = np.maximum(fft_spectrum - prev_fft_spectrum, 0.0)
            flux = np.sum(diff)

        delta_flux = flux - mean_flux
        mean_flux += delta_flux / i
        mean2_flux += delta_flux * (flux - mean_flux)
        var_flux = mean2_flux / i
        std_flux = np.sqrt(var_flux)
        flux = np.tanh((flux - mean_flux) / (std_flux + self._eps))
        # flux = (flux * 2) + 1.0

        freqs = np.linspace(0, sr / 2, num=len(fft_spectrum))

        # lf_mask = (freqs >= 2) & (freqs <= 16)
        # lf_power = np.sum(fft_spectrum[lf_mask])

        # rms = np.sqrt(np.mean(frame**2)) + self._eps
        dominant_energy = np.max(fft_spectrum)
        total_energy = np.sum(fft_spectrum) + self._eps
        mean_energy = total_energy / len(fft_spectrum) + self._eps

        tonality = np.log(dominant_energy / mean_energy)
        peaks = np.sum(fft_spectrum > 0.1 * dominant_energy)
        flatness = np.exp(np.mean(np.log(fft_spectrum))) / mean_energy
        centroid = np.sum(freqs * fft_spectrum) / total_energy

        # dominant frequency ratio
        dfr = 0.0
        if total_energy > 0.0:
            dfr = dominant_energy / total_energy

        # rolloff_idx = np.searchsorted(np.cumsum(fft_spectrum), 0.85)
        # rolloff_idx = min(rolloff_idx, len(freqs) - 1)
        # rolloff = freqs[rolloff_idx]
        # lf_energy = np.sum(fft_spectrum[freqs <= 300.0])
        # hf_energy = np.sum(fft_spectrum[freqs >= 300.0])
        # lf_ratio = lf_energy / (hf_energy + self._eps)

        # fft_spectrum /= total_energy
        # mel_energy = self._mel_filter_bank @ fft_spectrum
        # mel_energy = np.maximum(mel_energy, self._eps)
        # mean_log_mel_energy = np.mean(np.log(mel_energy + self._eps))

        # hnr = tonality / (flux + self._eps)

        # mfcc = librosa.feature.mfcc(
        #     y=frame,
        #     sr=sr,
        #     n_mfcc=5,
        #     n_fft=self._n_fft,
        #     hop_length=len(frame),
        # )
        # mfcc_mean = np.mean(mfcc[:, 0])

        feat_vec = np.hstack(
            [
                # log_energy,
                zcr,
                centroid,
                tonality,
                peaks,
                flux,
                dfr,
                flatness,
                # mean_log_mel_energy,
                log_energy,
                # rms,
                # lf_power,
                # hnr,
                # mfcc_mean,
            ]
        ).astype(np.float32)

        return feat_vec, fft_spectrum, mean_energy, mean2_energy, mean_flux, mean2_flux

# %% [markdown]
"""
### Feature scaling and contextual statistics.

The next step in the pipeline is to apply feature scaling and compute contextual statistics for the extracted features. The following classes handle these steps:
- `ScalerProtocol`: A protocol that defines the interface for feature scalers, which can be used to normalize the features before feeding them into the model.
- `AbstractFeatureCompute` and `FeatureWithContextStats`: These classes define the interface and implementation for computing contextual statistics for the extracted features. The `FeatureWithContextStats` class maintains a buffer of the most recent feature vectors and computes statistics such as mean and standard deviation over this buffer to capture the temporal context of the features. The computed statistics are then concatenated with the original feature vector to form an augmented feature vector that can provide the model with more information about the temporal dynamics of the audio signal.
- `FeatureWithVariableContextStats`: An extension of `FeatureWithContextStats` that allows for computing statistics over variable context sizes for different subsets of features. This can be useful if certain features benefit from longer or shorter context windows.
- `FeatureSelector`: A class that selects specific features from the computed feature vector based on a provided selection dictionary. This can be used to reduce the dimensionality of the feature vector and to filter out less relevant features without modifying the extraction pipeline.

### Observations

During experimentation, it was observed that mean and standard deviation statistics were sufficient.
The context window size significantly affects the trained model performance. A smaller 
window size results in higher reaction to signal changes, but the model misses longer-term signal patterns and trends, thus performing poorly. A larger window size captures more temporal context and results in excellent performance, but it slows down the reaction to signal changes.
"""

# %%
class ScalerProtocol(Protocol):
    def partial_fit(self, X: np.ndarray) -> None: ...

    def transform(self, X: np.ndarray) -> np.ndarray: ...

    def fit_transform(self, X: np.ndarray) -> np.ndarray: ...

# %%
class AbstractFeatureCompute(ABC):
    @abstractmethod
    def compute(self, *, feat_vec: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def reset(self) -> None: ...


class FeatureWithContextStats(AbstractFeatureCompute):
    def __init__(self, *, context_size: int = 5) -> None:
        from collections import deque

        from scipy.signal import butter

        self._context_size = context_size
        self._buf = deque(maxlen=context_size + 1)

    def compute(self, *, feat_vec: np.ndarray) -> np.ndarray:
        import numpy as np
        from scipy.signal import sosfilt

        self._buf.append(feat_vec)
        buf_array = np.vstack(self._buf)

        mean_vec = np.mean(buf_array, axis=0)
        std_vec = np.std(buf_array, axis=0)

        feat_with_stats = np.hstack(
            [
                mean_vec,
                std_vec,
            ]
        ).astype(np.float32)
        return feat_with_stats

    def reset(self) -> None:
        self._buf.clear()


class FeatureWithVariableContextStats(AbstractFeatureCompute):
    def __init__(self, *, context_sizes: dict[Iterable[tuple[str, int]], int]) -> None:
        self._context_sizes = context_sizes
        self._bufs = [
            ([p[1] for p in key], deque(maxlen=size + 1))
            for key, size in context_sizes.items()
        ]

        for key, buf in self._bufs:
            print(
                f"Initialized context stats buffer for features {key} with size {buf.maxlen}."
            )

    def compute(self, *, feat_vec: np.ndarray) -> np.ndarray:
        import numpy as np

        means, _vars = [], []
        for idx, buf in self._bufs:
            assert buf.maxlen is not None
            buf.append(feat_vec[idx])
            buf_array = np.stack(buf)

            if len(buf) < buf.maxlen:
                zeros = np.zeros_like(feat_vec[idx], dtype=np.float32)
                means.append(zeros)
                _vars.append(zeros)
                continue

            mean_vec = np.mean(buf_array, axis=0)
            var_vec = np.var(buf_array, axis=0, ddof=0)

            means.append(mean_vec)
            _vars.append(var_vec)

        feat_with_stats = np.hstack([*means, *_vars]).astype(np.float32)
        feat_with_stats = np.nan_to_num(
            feat_with_stats,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return feat_with_stats

    def reset(self) -> None:
        for _, buf in self._bufs:
            buf.clear()


class FeatureSelector(AbstractFeatureCompute):
    def __init__(
        self,
        *,
        feature_compute: AbstractFeatureCompute,
        selection: dict[str, int],
    ) -> None:
        self.selection = selection

        self._feature_compute = feature_compute
        self._selected_indices = sorted(selection.values())

    def compute(self, *, feat_vec: np.ndarray) -> np.ndarray:
        feat_vec = self._feature_compute.compute(feat_vec=feat_vec)
        return feat_vec[self._selected_indices]

    def reset(self) -> None:
        self._feature_compute.reset()

# %% [markdown]
"""
### Dataset audio feature extraction pipeline.

This section defines the `AbstractDatasetAudioFeatureExtractor` and `DatasetAudioFeatureExtractor` classes, which handle the feature extraction for each dataset.

The `DatasetAudioFeatureExtractor` class attempts to load pre-extracted feature vectors from disk cache for faster loading. If the cache is not available or invalid, it builds the feature vectors from the source audio files using the provided `AbstractAudioFeatureExtractor` implementation. The extracted feature vectors are then cached to disk for future runs if disk caching is enabled.
"""

# %%
class AbstractDatasetAudioFeatureExtractor(ABC):
    @abstractmethod
    def extract(
        self,
        *,
        dataset_meta: DatasetMeta,
        audio_data_producer: Generator[AudioData, None, None],
    ) -> Generator[AudioData, None, None]: ...


class DatasetAudioFeatureExtractor(AbstractDatasetAudioFeatureExtractor):
    default_cache_dir: Path = Path().resolve().parent / "data" / "processed"

    def __init__(
        self,
        *,
        feature_extractor: AbstractAudioFeatureExtractor,
        frame_generator: AbstractAudioFrameGenerator,
        use_disk_cache: bool = True,
        cache_dir: Path = default_cache_dir,
        mmap_mode: Literal["r", "r+", "w+", "c"] | None = None,
        print_stats: bool = True,
    ) -> None:
        self._feature_extractor = feature_extractor
        self._frame_generator = frame_generator
        self._use_disk_cache = use_disk_cache
        self._cache_dir = cache_dir
        self._mmap_mode: Literal["r", "r+", "w+", "c"] | None = mmap_mode
        self._print_stats = print_stats

    def extract(
        self,
        *,
        dataset_meta: DatasetMeta,
        audio_data_producer: Generator[AudioData, None, None],
    ) -> Generator[AudioData, None, None]:
        from time import time

        datasets_dir = dataset_meta.dataset_path.parent
        build = True

        s0 = time()
        if self._use_disk_cache:
            build = False

            if not self._cache_dir.exists():
                self._cache_dir.mkdir(parents=True, exist_ok=True)

            if self._print_stats:
                print(
                    f"Attempting to load extracted features for "
                    f"{len(dataset_meta.dataset_meta)} audio files "
                    f"from dataset '{dataset_meta.dataset_name}' cache...",
                )

            for slug in dataset_meta.dataset_meta["Slug"]:
                slug_path = Path(slug)
                cached_feats_path = self._cache_dir / slug_path.with_stem(
                    f"{slug_path.stem}_feats"
                ).with_suffix(".npy")
                if not cached_feats_path.exists():
                    build = True
                    if self._print_stats:
                        print(
                            f"Cache file '{cached_feats_path}' not found. "
                            f"Rebuilding extracted features from source files...",
                        )
                    break

        took = time() - s0
        if self._use_disk_cache and not build:
            i = 0
            for audio_data in audio_data_producer:
                s0 = time()
                slug = Path(audio_data.file_path).relative_to(datasets_dir)
                cached_feats_path = self._cache_dir / slug.with_stem(
                    f"{slug.stem}_feats"
                ).with_suffix(".npy")

                cached_feats = np.load(cached_feats_path, mmap_mode=self._mmap_mode)
                feat_audio_data = audio_data.with_feat_vectors(
                    feat_vectors=cached_feats,
                )

                fmt_err = "Cached feature-extracted audio data must contain {}."
                if feat_audio_data.feat_vectors is None:
                    raise ValueError(fmt_err.format("feature vectors"))

                i += 1
                if self._print_stats:
                    print(
                        f"Loaded features (CACHE): {slug} ({i}/{len(dataset_meta.dataset_meta)}) "
                        f"shape={feat_audio_data.feat_vectors.shape}"
                    )

                took += time() - s0
                yield feat_audio_data

            if not build and self._print_stats:
                print(
                    f"Loaded extracted features for "
                    f"{i} audio files in dataset "
                    f"'{dataset_meta.dataset_name}' cache in {took:.3f}s."
                )

        if not build:
            return

        if self._print_stats:
            print(
                f"Extracting features for {len(dataset_meta.dataset_meta)} audio files "
                f"in dataset '{dataset_meta.dataset_name}'...",
            )

        took = 0.0
        i = 0
        for audio_data in audio_data_producer:
            s0 = time()
            feat_audio_data = self._feature_extractor.extract(
                audio_data=audio_data,
                frame_generator=self._frame_generator,
            )
            if feat_audio_data.feat_vectors is None:
                raise ValueError(
                    f"Feature-extracted audio data from file '{audio_data.file_path}' "
                    f"does not contain feature vectors.",
                )

            if self._use_disk_cache:
                slug = Path(audio_data.file_path).relative_to(datasets_dir)
                cached_feats_path = self._cache_dir / slug.with_stem(
                    f"{slug.stem}_feats"
                ).with_suffix(".npy")
                if not cached_feats_path.parent.exists():
                    cached_feats_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(
                    cached_feats_path,
                    feat_audio_data.feat_vectors,
                    allow_pickle=False,
                )

            i += 1
            if self._print_stats:
                print(
                    f"Extracted: {audio_data.file_path} ({i}/{len(dataset_meta.dataset_meta)}) "
                    f"shape={feat_audio_data.feat_vectors.shape}"
                )

            took += time() - s0
            yield feat_audio_data

        if self._print_stats:
            print(
                f"Extracted features for {i} audio files "
                f"in dataset '{dataset_meta.dataset_name}' in {took:.3f}s."
            )

        return

# %% [markdown]
"""
### Dataset audio processing pipelines.

The `DatasetAudioPipeline` class defines a simple pipeline that processes the dataset by loading the audio data, teaching probabilities, and extracting features. It iterates through the processed audio data but does not perform any further operations on it.

The `DatasetPartialLearningPipeline` class extends the functionality of the `DatasetAudioPipeline` by allowing for partial learning. It processes the dataset in a similar way but yields tuples of feature vectors, taught probabilities, and metadata for each audio file. This allows for more flexible training and evaluation, as the yielded data can be used to train models incrementally or to perform analysis on specific subsets of the dataset. The `DatasetPartialLearningPipelineY` class is a variant of the `DatasetPartialLearningPipeline` that yields `None` for the feature vectors, which can be useful for scenarios where only the taught probabilities and metadata are needed, such as in certain evaluation or analysis tasks.
"""

# %%
class AbstractDatasetAudioPipeline(ABC):
    @abstractmethod
    def process(
        self,
        *,
        dataset_meta: DatasetMeta,
        dataset_loader: AbstractDatasetAudioLoader,
        dataset_teacher: AbstractDatasetAudioTeacher,
        dataset_feature_extractor: AbstractDatasetAudioFeatureExtractor,
    ) -> None: ...


class DatasetAudioPipeline(AbstractDatasetAudioPipeline):
    def __init__(self, *, print_stats: bool = True) -> None:
        self._print_stats = print_stats

    def process(
        self,
        *,
        dataset_meta: DatasetMeta,
        dataset_loader: AbstractDatasetAudioLoader,
        dataset_teacher: AbstractDatasetAudioTeacher,
        dataset_feature_extractor: AbstractDatasetAudioFeatureExtractor,
    ) -> None:
        if self._print_stats:
            print(f"\nProcessing dataset '{dataset_meta.dataset_name}':")

        online_loader = dataset_loader.load(dataset_meta=dataset_meta)
        online_teacher = dataset_teacher.teach(
            dataset_meta=dataset_meta,
            audio_data_producer=online_loader,
        )
        online_feature_extractor = dataset_feature_extractor.extract(
            dataset_meta=dataset_meta,
            audio_data_producer=online_teacher,
        )

        for _ in online_feature_extractor:
            pass


# %%
class DatasetPartialLearningSamplesMetadata(NamedTuple):
    samples_slug: str
    samples_index: int
    samples_total: int
    is_train: bool
    is_next_train: bool
    audio_data: AudioData


class DatasetPartialLearningPipeline:
    def __init__(self, *, print_stats: bool = True) -> None:
        self._print_stats = print_stats

    def process(
        self,
        *,
        dataset_meta: DatasetMeta,
        dataset_loader: AbstractDatasetAudioLoader,
        dataset_teacher: AbstractDatasetAudioTeacher,
        dataset_feature_extractor: AbstractDatasetAudioFeatureExtractor,
        feature_compute: AbstractFeatureCompute,
        train_split: float,  # like 0.8 for 80% training data
        skip_first: float = 0.0,  # like 0.1 to skip first 10% of data
    ) -> Generator[
        tuple[
            np.ndarray,
            np.ndarray,
            DatasetPartialLearningSamplesMetadata,
        ],
        None,
        None,
    ]:
        if self._print_stats:
            print(f"\nProcessing dataset '{dataset_meta.dataset_name}':")

        online_loader = dataset_loader.load(dataset_meta=dataset_meta)
        online_teacher = dataset_teacher.teach(
            dataset_meta=dataset_meta,
            audio_data_producer=online_loader,
        )
        online_feature_extractor = dataset_feature_extractor.extract(
            dataset_meta=dataset_meta,
            audio_data_producer=online_teacher,
        )

        n_total = len(dataset_meta.dataset_meta)  # 632
        first_i = int(skip_first * n_total)  # 0
        n_effective = n_total - first_i  # 632
        n_train = int(train_split * n_effective)  # 505
        print(
            f"Dataset '{dataset_meta.dataset_name}': "
            f"total={n_total}, "
            f"skip_first={first_i}, "
            f"effective={n_effective}, "
            f"train={n_train}, "
            f"test={n_effective - n_train}.",
        )

        for i, audio_data in enumerate(online_feature_extractor):
            if i < first_i:
                feature_compute.reset()  # Reset context stats for next audio file.
                continue

            fmt_err = "Audio data must contain {} for final processing."
            if audio_data.taught_probas is None:
                raise ValueError(fmt_err.format("taught probabilities"))
            elif audio_data.feat_vectors is None:
                raise ValueError(fmt_err.format("feature vectors"))

            feature_compute.reset()
            X = np.array(
                [
                    feature_compute.compute(feat_vec=feat_vec)
                    for feat_vec in audio_data.feat_vectors
                ],
                dtype=np.float32,
            )
            feature_compute.reset()
            y = audio_data.taught_probas

            effective_i = i - first_i
            is_train = effective_i < n_train
            is_next_train = (effective_i + 1) < n_train
            sample_metadata = DatasetPartialLearningSamplesMetadata(
                samples_slug=Path(audio_data.file_path)
                .relative_to(dataset_meta.dataset_path.parent)
                .as_posix(),
                samples_index=effective_i,
                samples_total=n_effective,
                is_train=is_train,
                is_next_train=is_next_train,
                audio_data=audio_data,
            )

            yield X, y, sample_metadata


class DatasetPartialLearningPipelineY:
    def __init__(self, *, print_stats: bool = True) -> None:
        self._print_stats = print_stats

    def process(
        self,
        *,
        dataset_meta: DatasetMeta,
        dataset_loader: AbstractDatasetAudioLoader,
        dataset_teacher: AbstractDatasetAudioTeacher,
        dataset_feature_extractor: AbstractDatasetAudioFeatureExtractor,
        feature_compute: AbstractFeatureCompute,
        train_split: float,  # like 0.8 for 80% training data
        skip_first: float = 0.0,  # like 0.1 to skip first 10% of data
    ) -> Generator[
        tuple[
            None,
            np.ndarray,
            DatasetPartialLearningSamplesMetadata,
        ],
        None,
        None,
    ]:
        if self._print_stats:
            print(f"\nProcessing dataset '{dataset_meta.dataset_name}':")

        online_loader = dataset_loader.load(dataset_meta=dataset_meta)
        online_teacher = dataset_teacher.teach(
            dataset_meta=dataset_meta,
            audio_data_producer=online_loader,
        )

        n_total = len(dataset_meta.dataset_meta)  # 632
        first_i = int(skip_first * n_total)  # 0
        n_effective = n_total - first_i  # 632
        n_train = int(train_split * n_effective)  # 505
        print(
            f"Dataset '{dataset_meta.dataset_name}': "
            f"total={n_total}, "
            f"skip_first={first_i}, "
            f"effective={n_effective}, "
            f"train={n_train}, "
            f"test={n_effective - n_train}.",
        )

        for i, audio_data in enumerate(online_teacher):
            if i < first_i:
                continue

            fmt_err = "Audio data must contain {} for final processing."
            if audio_data.taught_probas is None:
                raise ValueError(fmt_err.format("taught probabilities"))

            y = audio_data.taught_probas

            effective_i = i - first_i
            is_train = effective_i < n_train
            is_next_train = (effective_i + 1) < n_train
            sample_metadata = DatasetPartialLearningSamplesMetadata(
                samples_slug=Path(audio_data.file_path)
                .relative_to(dataset_meta.dataset_path.parent)
                .as_posix(),
                samples_index=effective_i,
                samples_total=n_effective,
                is_train=is_train,
                is_next_train=is_next_train,
                audio_data=audio_data,
            )

            yield None, y, sample_metadata


# %% [markdown]
"""
### Offline prediction with SGDClassifier.

The `AbstractOfflinePredictor` class defines the interface for offline predictors, which take in audio data and produce predictions based on the extracted features. The `AbstractPredictionModel` class defines the interface for prediction models, which can compute decision functions and predicted probabilities from feature vectors.

The `SGDClassifierModel` class is a concrete implementation of the `AbstractPredictionModel` interface that wraps around a scikit-learn `SGDClassifier`. It implements the `decision_function` and `predict_proba` methods by calling the corresponding methods on the underlying model.

The `SGDEnsembleModel` class is an ensemble implementation of the `AbstractPredictionModel` interface that takes a list of `SGDClassifier` models and averages their decision functions to produce a final prediction. The `SGDEnsembleModel2` class is a variant that averages the predicted probabilities instead of the decision functions.

The `BaseOfflineSGDPredictor` class is a concrete implementation of the `AbstractOfflinePredictor` interface that uses an `AbstractPredictionModel` to make predictions on audio data. It takes in a model, a scaler for feature normalization, and a feature compute for computing contextual statistics. The `predict` method extracts features from the audio data, applies scaling, and then uses the model to compute predicted probabilities, which are then attached to the audio data for further use.
"""

# %%
from sklearn.linear_model import SGDClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


class AbstractOfflinePredictor(ABC):
    @abstractmethod
    def predict(self, *, audio_data: AudioData) -> AudioData: ...


class AbstractPredictionModel(ABC):
    @abstractmethod
    def decision_function(self, *, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def predict_proba(self, *, X: np.ndarray) -> np.ndarray: ...


class SGDClassifierModel(AbstractPredictionModel):
    def __init__(self, *, model: SGDClassifier) -> None:
        self._model = model

    def decision_function(self, *, X: np.ndarray) -> np.ndarray:
        return self._model.decision_function(X)

    def predict_proba(self, *, X: np.ndarray) -> np.ndarray:
        from scipy.special import expit

        logits = self.decision_function(X=X)
        probas = expit(logits)
        return probas


class SGDEnsembleModel(AbstractPredictionModel):
    def __init__(self, *, models: list[SGDClassifier]) -> None:
        self._models = models

    def decision_function(self, *, X: np.ndarray) -> np.ndarray:
        import numpy as np

        logits = np.stack(
            [model.decision_function(X) for model in self._models],
            axis=0,
        )

        avg_logits = np.mean(logits, axis=0)
        return avg_logits

    def predict_proba(self, *, X: np.ndarray) -> np.ndarray:
        from scipy.special import expit

        logits = self.decision_function(X=X)
        probas = expit(logits)
        return probas


class SGDEnsembleModel2(AbstractPredictionModel):
    def __init__(self, *, models: list[SGDClassifier]) -> None:
        self._models = models

    def predict_proba(self, *, X: np.ndarray) -> np.ndarray:
        import numpy as np

        probas = np.stack(
            [model.predict_proba(X)[:, 1] for model in self._models],
            axis=0,
        )

        return np.mean(probas, axis=0)

    def decision_function(self, *, X: np.ndarray) -> np.ndarray:
        import numpy as np
        from scipy.special import logit

        probas = self.predict_proba(X=X)
        eps = 1e-10
        return logit(np.clip(probas, eps, 1 - eps))


class BaseOfflineSGDPredictor(AbstractOfflinePredictor):
    def __init__(
        self,
        *,
        model: AbstractPredictionModel,
        scaler: ScalerProtocol,
        feature_compute: AbstractFeatureCompute,
        print_stats: bool = True,
    ) -> None:
        self._model = model
        self._scaler = scaler
        self._feature_compute = feature_compute
        self._print_stats = print_stats

        self._avg_predict_n = 0
        self._avg_predict_time = 0.0

    def decision_function(self, *, X: np.ndarray) -> np.ndarray:
        return self._model.decision_function(X=X)

    def predict_proba(self, *, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X=X)

    def predict(self, *, audio_data: AudioData) -> AudioData:
        fmt_err = "Audio data must contain {} for prediction."
        if audio_data.feat_vectors is None:
            raise ValueError(fmt_err.format("feature vectors"))

        self._feature_compute.reset()
        X = np.array(
            [
                self._feature_compute.compute(feat_vec=feat_vec)
                for feat_vec in audio_data.feat_vectors
            ],
            dtype=np.float32,
        )
        self._feature_compute.reset()

        s0 = time()
        X = self._scaler.transform(X)
        y_pred = self.predict_proba(X=X)
        # y_pred = self.decision_function(X=X)
        s1 = time()

        if self._print_stats:
            print(f"Predicted probabilities in {s1 - s0:.3f}s.")

        # Statistics.
        self._avg_predict_n += 1
        self._avg_predict_time = (
            (self._avg_predict_n - 1) * self._avg_predict_time + (s1 - s0)
        ) / self._avg_predict_n

        if self._print_stats:
            print(
                f"Average prediction time over "
                f"{self._avg_predict_n} runs: "
                f"{self._avg_predict_time:.3f}s.",
            )

        return audio_data.with_predicted_probas(predicted_probas=y_pred)

class OfflineXGBoostPredictor(AbstractOfflinePredictor):
    def __init__(
        self,
        *,
        model: AbstractPredictionModel,
        feature_compute: AbstractFeatureCompute,
        print_stats: bool = True,
    ) -> None:
        self._model = model
        self._feature_compute = feature_compute
        self._print_stats = print_stats

    def predict(self, *, audio_data: AudioData) -> AudioData:
        fmt_err = "Audio data must contain {} for prediction."
        if audio_data.feat_vectors is None:
            raise ValueError(fmt_err.format("feature vectors"))

        self._feature_compute.reset()
        X = np.array(
            [
                self._feature_compute.compute(feat_vec=feat_vec)
                for feat_vec in audio_data.feat_vectors
            ],
            dtype=np.float32,
        )
        self._feature_compute.reset()

        s0 = time()
        y_pred = self._model.predict_proba(X=X)[:, 1]
        s1 = time()

        if self._print_stats:
            print(f"Predicted probabilities in {s1 - s0:.3f}s.")

        return audio_data.with_predicted_probas(predicted_probas=y_pred)

# %% [markdown]
"""
### Dataset audio pipeline manager.

The `DatasetAudioPipelineManager` class manages the execution of the audio processing pipelines for multiple datasets. It takes in a dataset audio pipeline, metadata for the datasets, a dataset loader, a dictionary of teacher factories for each dataset, and a dataset feature extractor. The `run` method executes the pipeline for each dataset in parallel using joblib's `Parallel` and `delayed` functions. The `_run_one` method processes an individual dataset by checking if it should be processed based on the provided criteria (such as being in the 'only' list or having an available teacher) and then running the audio pipeline with the appropriate components.
"""

# %%
class DatasetAudioPipelineManager:
    def __init__(
        self,
        *,
        dataset_audio_pipeline: AbstractDatasetAudioPipeline,
        dataset_metas: dict[str, DatasetMeta],
        dataset_loader: AbstractDatasetAudioLoader,
        dataset_teacher_factories: dict[str, Callable[[], AbstractDatasetAudioTeacher]],
        dataset_feature_extractor: AbstractDatasetAudioFeatureExtractor,
        only: set[str] | None = None,
        verbose: int = 50,
        n_jobs: int = -1,
    ) -> None:
        self._dataset_audio_pipeline = dataset_audio_pipeline
        self._dataset_metas = dataset_metas
        self._dataset_loader = dataset_loader
        self._dataset_teacher_factories = dataset_teacher_factories
        self._dataset_feature_extractor = dataset_feature_extractor
        self._only = only
        self._verbose = verbose
        self._n_jobs = n_jobs

    def run(self) -> None:
        from joblib import Parallel, delayed

        Parallel(
            n_jobs=self._n_jobs,
            backend="threading",
            verbose=self._verbose,
        )(
            delayed(self._run_one)(dataset_meta=dataset_meta)
            for dataset_meta in dataset_metas.values()
        )

    def _run_one(self, *, dataset_meta: DatasetMeta) -> None:
        dataset_name = dataset_meta.dataset_name

        if self._only is not None and dataset_name not in self._only:
            print(f"\nSkipping dataset '{dataset_name}': not in 'only' list.")
            return

        if dataset_name not in self._dataset_teacher_factories:
            print(
                f"\nSkipping dataset '{dataset_name}': no teacher available.",
            )
            return

        self._dataset_audio_pipeline.process(
            dataset_meta=dataset_meta,
            dataset_loader=self._dataset_loader,
            dataset_teacher=self._dataset_teacher_factories[dataset_name](),
            dataset_feature_extractor=self._dataset_feature_extractor,
        )

# %% [markdown]
"""
### The beginning. Loaders and teachers

From this section onwards, the actual training and evaluation pipelines are defined.
Here, loaders, teachers, and a feature extractor is created for individual audio files.
"""

# %%
dataset_loader = DatasetAudioLoader(
    audio_loader=AudioLoader(print_stats=True),
    target_sr=8000,
    chunk_size=int(0.01 * 8000),  # 10 ms chunks
    use_disk_cache=False,
    print_stats=True,
)

noop_dataset_loader = DatasetAudioLoader(
    audio_loader=NoopAudioLoader(print_stats=True),
    target_sr=8000,
    chunk_size=int(0.01 * 8000),  # 10 ms chunks
    use_disk_cache=False,
    print_stats=True,
)


def make_nonspeech_dataset_teacher() -> DatasetAudioTeacher:
    print("Creating Non-Speech Dataset Teacher...")
    return DatasetAudioTeacher(
        probability_teacher=NonSpeechProbabilityTeacher(print_stats=False),
        use_disk_cache=True,
        print_stats=True,
    )


def make_mix_dataset_teacher() -> DatasetAudioTeacher:
    print("Creating Mix Dataset Teacher...")
    return DatasetAudioTeacher(
        probability_teacher=SileroProbabilityTeacher(
            print_stats=False, print_init_stats=True
        ),
        use_disk_cache=True,
        print_stats=True,
    )


dataset_feature_extractor = DatasetAudioFeatureExtractor(
    feature_extractor=AudioFeatureExtractor(),
    frame_generator=AudioFrameGenerator(),
    use_disk_cache=True,
    print_stats=True,
)

# %% [markdown]
"""
### Display the metadata once again
"""

# %%
datasets_meta.show(groups=True)

# %% [markdown]
"""
### Run the dataset audio processing pipelines.

This cell will load audio files, ensure the teachers labelled the data,
and ensure the features are extracted. This can take a while.
"""

# %%
dataset_audio_pipeline = DatasetAudioPipeline(print_stats=True)
pipeline_manager = DatasetAudioPipelineManager(
    dataset_audio_pipeline=dataset_audio_pipeline,
    dataset_metas=dataset_metas,
    dataset_loader=dataset_loader,
    dataset_teacher_factories={
        "mix_ava": make_mix_dataset_teacher,
        "mix_private_telephony": make_mix_dataset_teacher,
        "mix_voxconverse_test": make_mix_dataset_teacher,
        "nonspeech_esc_50": make_nonspeech_dataset_teacher,
        "nonspeech_musan_music_rmf": make_nonspeech_dataset_teacher,
        "nonspeech_musan_noise": make_nonspeech_dataset_teacher,
        "speech_callhome_deu": make_mix_dataset_teacher,
        "speech_musan_speech": make_mix_dataset_teacher,
    },
    dataset_feature_extractor=dataset_feature_extractor,
    n_jobs=12,
)

pipeline_manager.run()

# %% [markdown]
"""
### Define models and feature compute for training and evaluation.

Here, a list of `SGDClassifier` models with different regularization strengths (alphas) is created.
This is an experimentationl setup to train multiple models at once to compare their performance.
This is not how the final model is selected.
"""

# %%
from sklearn.linear_model import SGDClassifier  # noqa: E402

alphas = [
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
]
class_weights = [
    # {0.0: 1.0, 1.0: 1.0},
    # {0.0: 1.5, 1.0: 1.0},
    # {0.0: 2.0, 1.0: 1.0},
    # {0.0: 1.0, 1.0: 1.0},
    # {0.0: 1.0, 1.0: 1.5},
    # {0.0: 1.0, 1.0: 2.0},
    # {0.0: 1.0, 1.0: 1.0},
    # {0.0: 1.5, 1.0: 1.0},
    # {0.0: 1.0, 1.0: 1.0},
    # {0.0: 1.0, 1.0: 1.5},
    {0.0: 1.0, 1.0: 1.0},
    {0.0: 1.0, 1.0: 1.0},
    {0.0: 1.0, 1.0: 1.0},
    {0.0: 1.0, 1.0: 1.0},
    {0.0: 1.0, 1.0: 1.0},
    {0.0: 1.0, 1.0: 1.0},
    {0.0: 1.0, 1.0: 1.0},
    {0.0: 1.0, 1.0: 1.0},
    {0.0: 1.0, 1.0: 1.0},
    {0.0: 1.0, 1.0: 1.0},
]
models = [
    SGDClassifier(
        loss="log_loss",
        learning_rate="optimal",
        alpha=alpha,
        penalty="l2",
        # tol=1e-4,
        warm_start=True,
    )
    for _, alpha in enumerate(alphas)
]
for model, class_weight in zip(models, class_weights):
    model.set_params(class_weight=class_weight)

# %% [markdown]
"""
### Feature filtering and learning pipelines

This cell creates feature compute and selectors for filtering.

Then, online learning pipelines are created.
"""

# %%
from time import time  # noqa: E402

feature_compute_ctx = FeatureWithVariableContextStats(
    context_sizes={
        (
            ("zcr", 0),
            ("centroid", 1),
            ("tonality", 2),
            ("peaks", 3),
            # ("flux", 4),
            # ("log_mel_energy", 5),
            # ("log_energy", 6),
            # ("rms", 7),
            # ("dfr", 4),
            # ("flatness", 5),
        ): 100,
        (
            # ("zcr", 0),
            # ("centroid", 1),
            # ("tonality", 2),
            # ("peaks", 3),
            ("flux", 4),
            # ("log_mel_energy", 5),
            # ("log_energy", 7),
            # ("rms", 7),
            # ("dfr", 8),
            # ("flatness", 9),
        ): 5,
        (
            # ("zcr", 0),
            # ("centroid", 1),
            # ("tonality", 2),
            # ("peaks", 3),
            # ("flux", 4),
            # ("log_mel_energy", 5),
            # ("log_energy", 6),
            # ("rms", 7),
            ("dfr", 5),
            ("flatness", 6),
        ): 100,
        (
            # ("zcr", 0),
            # ("centroid", 1),
            # ("tonality", 2),
            # ("peaks", 3),
            # ("flux", 6),
            # ("log_mel_energy", 5),
            ("log_energy", 7),
            # ("rms", 7),
            # ("dfr", 8),
            # ("flatness", 9),
        ): 5,
    },
)  # One instance is safe.
feature_compute_selector = FeatureSelector(
    feature_compute=feature_compute_ctx,
    selection={
        "mean_zcr": 0,
        "mean_centroid": 1,
        "mean_tonality": 2,
        "mean_peaks": 3,
        "mean_flux": 4,
        "mean_dfr": 5,
        "mean_flatness": 6,
        # "mean_log_mel_energy": 5,
        "mean_log_energy": 7,
        # "mean_rms": 7,
        "var_zcr": 8,
        "var_centroid": 9,
        "var_tonality": 10,
        "var_peaks": 11,
        # "var_flux": 14,
        # "var_log_mel_energy": 15,
        # "var_log_energy": 16,
        # "var_rms": 17,
        "var_dfr": 13,
        "var_flatness": 14,
        # "var_zcr": 20,
        # "var_centroid": 21,
        # "var_tonality": 22,
        # "var_peaks": 23,
        # "var_flux": 24,
        # "var_log_mel_energy": 25,
        # "var_log_energy": 26,
        # "var_rms": 27,
        # "var_dfr": 28,
        # "var_flatness": 29,
    },
)
learning_pipeline = DatasetPartialLearningPipeline(print_stats=False)
learning_pipeline_y = DatasetPartialLearningPipelineY(print_stats=True)

SKIP = True
NOT_SKIP = False


def make_learners() -> dict[str, tuple[DatasetMeta, bool, Generator]]:
    return {
        "mix_ava": (
            dataset_metas["mix_ava"],
            NOT_SKIP,
            learning_pipeline_y.process(
                dataset_meta=dataset_metas["mix_ava"].shuffled(
                    random_state=42,
                ),
                dataset_loader=noop_dataset_loader,
                dataset_teacher=make_mix_dataset_teacher(),
                dataset_feature_extractor=dataset_feature_extractor,
                feature_compute=feature_compute_selector,
                train_split=0.9,
                skip_first=0.0,
            ),
        ),
        "mix_private_telephony": (
            dataset_metas["mix_private_telephony"],
            NOT_SKIP,
            learning_pipeline_y.process(
                dataset_meta=dataset_metas["mix_private_telephony"].shuffled(
                    random_state=42,
                ),
                dataset_loader=noop_dataset_loader,
                dataset_teacher=make_mix_dataset_teacher(),
                dataset_feature_extractor=dataset_feature_extractor,
                feature_compute=feature_compute_selector,
                train_split=0.9,
                skip_first=0.0,
            ),
        ),
        "mix_voxconverse_test": (
            dataset_metas["mix_voxconverse_test"],
            NOT_SKIP,
            learning_pipeline_y.process(
                dataset_meta=dataset_metas["mix_voxconverse_test"].shuffled(
                    random_state=42,
                ),
                dataset_loader=noop_dataset_loader,
                dataset_teacher=make_mix_dataset_teacher(),
                dataset_feature_extractor=dataset_feature_extractor,
                feature_compute=feature_compute_selector,
                train_split=0.9,
                skip_first=0.0,
            ),
        ),
        "nonspeech_esc_50": (
            dataset_metas["nonspeech_esc_50"],
            NOT_SKIP,
            learning_pipeline_y.process(
                dataset_meta=dataset_metas["nonspeech_esc_50"].shuffled(
                    random_state=42,
                ),
                dataset_loader=noop_dataset_loader,
                dataset_teacher=make_nonspeech_dataset_teacher(),
                dataset_feature_extractor=dataset_feature_extractor,
                feature_compute=feature_compute_selector,
                train_split=0.9,
                skip_first=0.0,
            ),
        ),
        "nonspeech_musan_music_rmf": (
            dataset_metas["nonspeech_musan_music_rmf"],
            NOT_SKIP,
            learning_pipeline_y.process(
                dataset_meta=dataset_metas["nonspeech_musan_music_rmf"].shuffled(
                    random_state=42,
                ),
                dataset_loader=noop_dataset_loader,
                dataset_teacher=make_nonspeech_dataset_teacher(),
                dataset_feature_extractor=dataset_feature_extractor,
                feature_compute=feature_compute_selector,
                train_split=0.9,
                skip_first=0.0,
            ),
        ),
        "nonspeech_musan_noise": (
            dataset_metas["nonspeech_musan_noise"],
            NOT_SKIP,
            learning_pipeline_y.process(
                dataset_meta=dataset_metas["nonspeech_musan_noise"].shuffled(
                    random_state=42,
                ),
                dataset_loader=noop_dataset_loader,
                dataset_teacher=make_nonspeech_dataset_teacher(),
                dataset_feature_extractor=dataset_feature_extractor,
                feature_compute=feature_compute_selector,
                train_split=0.9,
                skip_first=0.0,
            ),
        ),
        "speech_callhome_deu": (
            dataset_metas["speech_callhome_deu"],
            NOT_SKIP,
            learning_pipeline_y.process(
                dataset_meta=dataset_metas["speech_callhome_deu"].shuffled(
                    random_state=42,
                ),
                dataset_loader=noop_dataset_loader,
                dataset_teacher=make_mix_dataset_teacher(),
                dataset_feature_extractor=dataset_feature_extractor,
                feature_compute=feature_compute_selector,
                train_split=0.9,
                skip_first=0.0,
            ),
        ),
        "speech_musan_speech": (
            dataset_metas["speech_musan_speech"],
            NOT_SKIP,
            learning_pipeline_y.process(
                dataset_meta=dataset_metas["speech_musan_speech"].shuffled(
                    random_state=42,
                ),
                dataset_loader=noop_dataset_loader,
                dataset_teacher=make_mix_dataset_teacher(),
                dataset_feature_extractor=dataset_feature_extractor,
                feature_compute=feature_compute_selector,
                train_split=0.9,
                skip_first=0.0,
            ),
        ),
    }


def make_training_learners() -> dict[str, tuple[DatasetMeta, bool, Generator]]:
    learners = make_learners()
    return {
        "mix_ava": learners["mix_ava"],
        "nonspeech_esc_50_0": learners["nonspeech_esc_50"],
        "mix_private_telephony": learners["mix_private_telephony"],
        "nonspeech_esc_50_1": learners["nonspeech_esc_50"],
        "nonspeech_musan_noise_0": learners["nonspeech_musan_noise"],
        "mix_voxconverse_test": learners["mix_voxconverse_test"],
        "nonspeech_esc_50_2": learners["nonspeech_esc_50"],
        "nonspeech_musan_music_rmf": learners["nonspeech_musan_music_rmf"],
        "speech_callhome_deu": learners["speech_callhome_deu"],
        "nonspeech_musan_noise_1": learners["nonspeech_musan_noise"],
        "speech_musan_speech": learners["speech_musan_speech"],
        "nonspeech_esc_50_3": learners["nonspeech_esc_50"],
    }


def make_fine_tuning_learners() -> dict[str, tuple[DatasetMeta, bool, Generator]]:
    return make_training_learners()

# %% [markdown]
"""
### Round-robin sampling from multiple learners.

The datasets are large, such that will very likely not not fit into RAM.
Hence, the model is trained partially on one audio file at a time. To improve
the balance and better fitting, the samples from all datasets are interleaved
in a round-robin manner. This improves performance and class balance.
"""

# %%
def round_robin_sampling(
    learners_dict: dict[str, tuple[DatasetMeta, bool, Generator]],
    *,
    stop_on_first_exhausted: bool = False,
) -> Generator:
    from collections import deque

    learners_queue = deque(learners_dict.values())
    while learners_queue:
        dataset_meta, skip_dataset, learner = learners_queue.popleft()
        if skip_dataset:
            continue
        try:
            X, y, sample_metadata = next(learner)
            yield dataset_meta, X, y, sample_metadata
            learners_queue.append((dataset_meta, skip_dataset, learner))
        except StopIteration:
            if stop_on_first_exhausted:
                break
            continue

# %% [markdown]
"""
### Scaler fitting.

The next two cells perform the fitting of the `StandardScaler` on the training data. This is done in an online manner. `round_robin_sampling` is here for consistency with actual training, but
can be skipped since the scaler will ultimately fit all training data.
"""

# %%
scaler = StandardScaler()

# %%
s0 = time()
XX = []
for dataset_meta, X, y, meta in round_robin_sampling(
    make_training_learners(),
    stop_on_first_exhausted=True,
):
    if not meta.is_train:
        continue
    t0 = time()
    scaler.partial_fit(X)
    XX.append(X) # if one plans to store X samples.

    print(f"Fitted in {time() - t0:.3f}s.")
    print(
        f"{dataset_meta.dataset_name}: "
        f"Sample {meta.samples_index + 1}/"
        f"{meta.samples_total} ",
    )
print(f"Total scaler fitting time: {time() - s0:.3f}s.")
print(len(XX))

# %% [markdown]
"""
### Online learning.

The next cell performs the online learning of the `SGDClassifier` models. The samples are taken in a round-robin manner from all datasets to improve balance and performance. The features are transformed using the fitted scaler before being used for training.
"""

# %%
s0 = time()
# to_predict = []
i = -1
yy = []
for dataset_meta, _, y, meta in round_robin_sampling(
    make_training_learners(),
    stop_on_first_exhausted=True,
):
    if not meta.is_train:
        # to_predict.append((dataset_meta, X, y, meta))
        continue
    i += 1
    X = XX[i] # if X samples are already loaded and stored.
    yy.append(y) # If one plans to store y samples.
    print(y.shape, X.shape)
    t0 = time()
    Xs = scaler.transform(X)
    for clf in models:
        clf.partial_fit(Xs, y, classes=[0.0, 1.0])

    print(f"Fitted in {time() - t0:.3f}s.")
    print(
        f"{dataset_meta.dataset_name}: "
        f"Sample {meta.samples_index + 1}/"
        f"{meta.samples_total} ",
    )
print(f"Total learning time: {time() - s0:.3f}s.")


# %%
import pickle

# with open("./XX1.pkl", "wb") as f:
#     pickle.dump(XX, f)

# with open("./YY1.pkl", "wb") as f:
#     pickle.dump(yy, f)

with open("./XX1.pkl", "rb") as f:
    XX = pickle.load(f)
with open("./YY1.pkl", "rb") as f:
    yy = pickle.load(f)

# %% [markdown]
"""
### Offline learning.

Once the feature set is fixed, different models can be evaluated much quicker
and easily in an offline manner. For this, `X` and `y` samples are collected in the previous online learning step to avoid recomputing them again, sicne context aggregation is not cached and takes a while.
"""

# %%
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.linear_model import SGDClassifier  # noqa: E402
from sklearn.ensemble import VotingClassifier  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

# %% [markdown]
"""
### Pipelines.

The following pipelines are defined for evaluation:
1. A single `SGDClassifier` with a specific set of hyperparameters.
2. An ensemble of `SGDClassifier` models with different regularization strengths (alphas) and class weights, using soft voting.
3. An `XGBClassifier` with specific hyperparameters.

`RandomForestClassifier` was not selected due to long training time.
"""

# %%
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "classifier",
            SGDClassifier(
                loss="log_loss",
                learning_rate="optimal",
                alpha=1e-4,
                penalty="l2",
                class_weight="balanced",
                n_jobs=-1,
            ),
        ),
    ],
)
# %%
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "classifier",
            VotingClassifier(
                estimators=[
                    (f"sgd_{i}", model)
                    for i, model in enumerate(
                        [
                            SGDClassifier(
                                loss="log_loss",
                                learning_rate="optimal",
                                alpha=alpha,
                                penalty="l2",
                                class_weight="balanced",
                                n_jobs=-1,
                            )
                            for alpha in alphas
                        ]
                    )
                ],
                voting="soft",
                n_jobs=-1,
            ),
        ),
    ],
)
# %%
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "classifier",
            XGBClassifier(
                max_depth=8,
                n_jobs=12,
                n_estimators=500,
                eval_metric="logloss",
                tree_method="hist",
                class_weight="balanced",
            ),
        ),
    ],
)

# %%
XX_ = np.vstack(XX)
print(len(XX), XX_.shape, XX[0].shape)
# %%
yy_ = np.concatenate(yy)
print(len(yy), yy_.shape, yy[0].shape)
# %%
from sklearn.model_selection import train_test_split  # noqa: E402
X_train, X_test, y_train, y_test = train_test_split(
    XX_,
    yy_,
    test_size=0.2,
    random_state=42,
    stratify=yy_,
)
# %%
pipeline.fit(X_train, y_train)

# %% [markdown]
"""
### Evaluation.

The following metrics are computed on the test set:
1. Classification report (precision, recall, f1-score).
2. Confusion matrix.
3. ROC AUC score.
"""

# %%
from sklearn.metrics import classification_report  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
# %%

y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# %%
print(roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1]))

# %%
# Plot a ROC curve
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import RocCurveDisplay  # noqa: E402

RocCurveDisplay.from_estimator(
    model,
    X_test,
    y_test,
)
plt.show()

# %% [markdown]
"""
### Results

```
SGDClasifier - baseline, log_loss.
========================================================

              precision    recall  f1-score   support

         0.0       0.78      0.64      0.70  13524017
         1.0       0.80      0.88      0.84  21570252

    accuracy                           0.79  35094269
   macro avg       0.79      0.76      0.77  35094269
weighted avg       0.79      0.79      0.78  35094269

[[ 8644813  4879204]
 [ 2505718 19064534]]

roc=0.845849912069585


SGDClasifier - log_loss, l2, balanced class weight
========================================================

              precision    recall  f1-score   support

         0.0       0.70      0.75      0.72  13524017
         1.0       0.83      0.79      0.81  21570252

    accuracy                           0.78  35094269
   macro avg       0.77      0.77      0.77  35094269
weighted avg       0.78      0.78      0.78  35094269

[[10132205  3391812]
 [ 4438565 17131687]]

roc=0.8459882135678315


10 SGDClasifiers with soft voting.
========================================================

              precision    recall  f1-score   support

         0.0       0.69      0.75      0.72  13524017
         1.0       0.84      0.79      0.81  21570252

    accuracy                           0.77  35094269
   macro avg       0.76      0.77      0.77  35094269
weighted avg       0.78      0.77      0.78  35094269

[[10187874  3336143]
 [ 4588512 16981740]]

roc=0.845650607462749


XGBoostClassifier - max_depth=6
========================================================

              precision    recall  f1-score   support

         0.0       0.86      0.80      0.83   2704803
         1.0       0.88      0.92      0.90   4314051

    accuracy                           0.87   7018854
   macro avg       0.87      0.86      0.86   7018854
weighted avg       0.87      0.87      0.87   7018854

[[2168549  536254]
 [ 364334 3949717]]

roc=0.9421431769028902


XGBoostClassifier - max_depth=8, 500 estimators, slower
========================================================

              precision    recall  f1-score   support

         0.0       0.88      0.84      0.86   2704803
         1.0       0.90      0.93      0.92   4314051

    accuracy                           0.89   7018854
   macro avg       0.89      0.88      0.89   7018854
weighted avg       0.89      0.89      0.89   7018854

[[2271541  433262]
 [ 304510 4009541]]

roc=0.959909106387729
```

![img](./0135_boosting_roc.png)
"""

# %% [markdown]
"""
### Progress towards these results

#### Small context, tried fine-tuning:

![img](./0127_tuned_bad.png)

#### MFCCs and bad scaling:

![img](./0128_features_bad.png)

#### Large context with certain weights scaled badly, obliterating the decision function:

![img](./0129_large_context_weights_bad.png)

#### Isolated well-scaled features to analyze the decision function:

![img](./0130_large_context_removed_energy_look_at_decision_func.png)
![img](./0131_large_context_removed_energy_look_at_decision_func.png)

#### First successful model with large context, slightly underfitted for speech:

![img](./0131_baseline.png)
![img](./0132_baseline.png)

#### Second successful model with large context and boosting, slightly overfitted for speech:

![img](./0133_boosting.png)
![img](./0134_boosting.png)
"""

# %% [markdown]
"""
### Model saving.
"""

# %%
import pickle  # noqa: E402
from datetime import datetime  # noqa: E402

with open(f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl", "wb") as f:
    pickle.dump(
        {
            "pipeline": pipeline,
            "comment": (
                "XGBClassifier with 500 estimators, max depth 8, "
                "roc_auc_score 0.95 on test set"
            ),
        },
        f,
    )

# %%
import pickle  # noqa: E402
from datetime import datetime  # noqa: E402

with open(f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl", "wb") as f:
    pickle.dump(
        {
            "scaler": scaler,
            "classifier": models,
            "comment": (
                "TODO"
            ),
        },
        f,
    )

# %% [markdown]
"""
### Model loading.
"""

# %%
import pickle

with open("model_20260205_223523.pkl", "rb") as f:
    saved_data = pickle.load(f)
    scaler = saved_data["scaler"]
    models = saved_data["classifier"]
    print(saved_data["comment"])

# %%
import pickle

with open("model_20260206_022323.pkl", "rb") as f:
    saved_data = pickle.load(f)
    model = saved_data["pipeline"]
    print(saved_data["comment"])

# %% [markdown]
"""
### Predictors.

The following predictors are defined for evaluation:
1. `BaseOfflineSGDPredictor` for the `SGDClassifier` models.
2. `OfflineXGBoostPredictor` for the `XGBClassifier` model.

The predictors take care of the feature computation and scaling before making predictions. They can also print statistics about the predictions if needed.
"""

# %%
predictor = BaseOfflineSGDPredictor(
    # Some other variants of models that can be used for prediction:
    # model=SGDEnsembleModel(models=models),
    # model=SGDClassifierModel(model=models[4]),
    model=model,
    scaler=scaler,
    feature_compute=feature_compute_selector,
    print_stats=False,
)

# %%
predictor = OfflineXGBoostPredictor(
    model=model,
    feature_compute=feature_compute_selector,
    print_stats=False,
)

# %% [markdown]
"""
### Visualizers.

The following visualizers are defined for visualizing the audio data, teacher probabilities, and predicted probabilities:
1. `StaticPlotAudioVisualizer` using Matplotlib for static plots.
2. `InteractivePlotAudioVisualizer` using Plotly for interactive plots.
3. `InteractiveScaledFeatureVisualizer` for visualizing scaled feature vectors in an interactive manner.
"""

# %%
class AbstractAudioVisualizer(AbstractVisualizer):
    """
    Specialized abstract visualizer for audio data.
    """

    @abstractmethod
    def show(self) -> None: ...


class StaticPlotAudioVisualizer(AbstractAudioVisualizer):
    def __init__(self, *, audio_data: AudioData) -> None:
        self._audio_data = audio_data

    def show(self) -> None:
        import matplotlib.pyplot as plt

        audio_data = self._audio_data
        speech = audio_data.audio
        sr = audio_data.sr
        taught_probas = audio_data.taught_probas

        assert speech is not None, "Audio data must contain audio samples."
        assert sr is not None, "Audio data must contain sampling rate."
        assert taught_probas is not None, (
            "Audio data must contain taught probabilities."
        )

        t_proba = (np.arange(len(taught_probas)) * audio_data.chunk_size) / sr

        _, ax1 = plt.subplots(figsize=(12, 3), dpi=300)

        # Audio waveform.
        ax1.plot(
            np.arange(len(speech)) / sr,
            speech,
            color="black",
            linewidth=1.2,
            alpha=0.8,
        )
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")

        # Taught probabilities.
        ax2 = ax1.twinx()
        ax2.step(
            t_proba,
            taught_probas,
            color="blue",
            linewidth=1.2,
            alpha=0.8,
        )
        ax2.get_yaxis().set_visible(False)
        # ax2.set_ylabel("Speech Probability from Teacher")

        # Predicted probabilities.
        if audio_data.predicted_probas is not None:
            ax3 = ax1.twinx()
            t_pred_proba = (
                np.arange(len(audio_data.predicted_probas)) * audio_data.chunk_size
            ) / sr
            ax3.step(
                t_pred_proba,
                audio_data.predicted_probas,
                color="red",
                linewidth=1.2,
                alpha=0.8,
            )
            ax3.set_ylabel("Trained Probability/Decision")

        plt.title(
            f"{self._audio_data.file_path}: Waveform + Teacher VAD + Probabilities"
        )
        plt.tight_layout()
        plt.show()


class InteractivePlotAudioVisualizer(AbstractAudioVisualizer):
    def __init__(self, *, audio_data: AudioData) -> None:
        self._audio_data = audio_data

    def show(self) -> None:
        import plotly.graph_objects as go

        audio_data = self._audio_data
        speech = audio_data.audio
        sr = audio_data.sr
        taught_probas = audio_data.taught_probas

        assert speech is not None, "Audio data must contain audio samples."
        assert sr is not None, "Audio data must contain sampling rate."
        assert taught_probas is not None, (
            "Audio data must contain taught probabilities."
        )

        t_proba = (np.arange(len(taught_probas)) * audio_data.chunk_size) / sr

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=np.arange(len(speech)) / sr,
                y=speech,
                mode="lines",
                name="Waveform",
                line=dict(color="black", width=1.2),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=t_proba,
                y=taught_probas,
                mode="lines",
                name="Speech Probability from Teacher",
                line=dict(shape="hv", color="blue", width=1.2),
                yaxis="y2",
            )
        )

        if audio_data.predicted_probas is not None:
            t_pred_proba = (
                np.arange(len(audio_data.predicted_probas)) * audio_data.chunk_size
            ) / sr
            fig.add_trace(
                go.Scatter(
                    x=t_pred_proba,
                    y=audio_data.predicted_probas,
                    mode="lines",
                    name="Predicted Speech Probability",
                    line=dict(shape="hv", color="red", width=1.2),
                    yaxis="y3",
                )
            )

        fig.update_layout(
            height=500,
            xaxis=dict(title="Time (s)"),
            yaxis=dict(title="Amplitude"),
            yaxis2=dict(
                title="Speech Probability from Teacher", overlaying="y", side="right"
            ),
            yaxis3=dict(
                title="Predicted Speech Probability",
                overlaying="y",
                side="right",
                position=0.95,
                range=[-1, 1],
            ),
            legend=dict(x=0.01, y=500, bordercolor="Black", borderwidth=1),
        )

        fig.show()


# %%
class InteractiveScaledFeatureVisualizer(AbstractAudioVisualizer):
    def __init__(
        self, *, audio_data: AudioData, scaled_feat_vectors: np.ndarray
    ) -> None:
        fmt_err = "Audio data must contain {} for visualization."
        if audio_data.audio is None:
            raise ValueError(fmt_err.format("audio samples"))
        elif audio_data.sr is None:
            raise ValueError(fmt_err.format("sampling rate"))
        self._audio_data = audio_data
        self._scaled_feat_vectors = scaled_feat_vectors

    def show(self) -> None:
        import numpy as np
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        audio = self._audio_data.audio
        assert audio is not None
        sr = self._audio_data.sr
        assert sr is not None
        feat = self._scaled_feat_vectors
        frames, feat_dim = feat.shape
        frame_duration = self._audio_data.chunk_size / sr

        t_audio = np.arange(len(audio)) / sr
        t_frames = np.arange(frames) * frame_duration

        # --- feature indexing ---
        scalar_names = [
            # "Log Energy",
            "ZCR",
            "Centroid",
            # "Flatness",
            "Tonality",
            # "Rolloff",
            # "LF/HF Ratio",
            "Peaks",
            "Flux",
        ]
        n_scalar = len(scalar_names)
        mel = feat[:, n_scalar:]
        print(feat_dim, n_scalar)

        # --- figure with subplots ---
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(
                "Waveform",
                "Scalar Features",
            ),
        )

        # --- waveform ---
        fig.add_trace(
            go.Scatter(
                x=t_audio,
                y=audio,
                mode="lines",
                name="Waveform",
                line=dict(color="black", width=1),
            ),
            row=1,
            col=1,
        )

        # --- energy + ZCR ---
        for idx, name, color in [
            (0, "ZCR", "red"),
            (1, "Centroid", "blue"),
            (2, "Tonality", "green"),
            (3, "Peaks", "orange"),
            (4, "Flux", "purple"),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=t_frames,
                    y=feat[:, idx],
                    mode="lines",
                    name=name,
                    opacity=0.7,
                    line=dict(color=color, width=2),
                ),
                row=2,
                col=1,
            )

        fig.update_layout(
            height=900,
            xaxis_title="Time (s)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0.01,
            ),
            title="Audio + Scaled Scalar Features",
        )

        fig.show()

        # ===== log-mel heatmap =====
        fig_mel = go.Figure()

        fig_mel.add_trace(
            go.Scatter(
                x=t_audio,
                y=audio,
                mode="lines",
                name="Waveform",
                line=dict(color="black", width=1),
            )
        )

        fig_mel.add_trace(
            go.Heatmap(
                z=mel.T,
                x=t_frames,
                y=np.arange(mel.shape[1]),
                colorscale="Viridis",
                colorbar=dict(title="Log-Mel Energy"),
                opacity=0.9,
                yaxis="y2",
            )
        )

        fig_mel.update_layout(
            height=600,
            xaxis=dict(title="Time (s)"),
            yaxis=dict(title="Amplitude"),
            yaxis2=dict(
                title="Mel Band",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
            title="Waveform + Log-Mel Spectrogram",
        )

        fig_mel.show()


class InteractiveFeatureVisualizer(AbstractAudioVisualizer):
    def __init__(
        self,
        *,
        audio_data: AudioData,
        scaled_ctx_feat_vectors: np.ndarray,
        selected_ctx_feat_vectors: np.ndarray,
    ) -> None:
        fmt_err = "Audio data must contain {} for visualization."
        if audio_data.audio is None:
            raise ValueError(fmt_err.format("audio samples"))
        if audio_data.sr is None:
            raise ValueError(fmt_err.format("sampling rate"))
        if audio_data.feat_vectors is None:
            raise ValueError(fmt_err.format("raw feature vectors"))

        self._audio_data = audio_data
        self._scaled_ctx_feat_vectors = scaled_ctx_feat_vectors
        self._selected_ctx_feat_vectors = selected_ctx_feat_vectors

    def show(self) -> None:
        import numpy as np
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        audio = self._audio_data.audio
        sr = self._audio_data.sr
        raw_feat = self._audio_data.feat_vectors

        assert audio is not None
        assert sr is not None
        assert raw_feat is not None

        ctx_feat = self._scaled_ctx_feat_vectors

        n_frames, n_raw_feat = raw_feat.shape

        # ---- base feature names (MUST match extractor order) ----
        feature_names = [
            "ZCR",
            "Centroid",
            "Tonality",
            "Peaks",
            "Flux",
            "DFR",
            "Flatness",
            "LogEnergy",
        ]

        assert n_raw_feat == len(feature_names), (
            f"Expected {len(feature_names)} raw features, got {n_raw_feat}"
        )

        # ---- context layout ----
        # ctx = [ mean(features), var(features), flux_var ]
        n_ctx_feat = ctx_feat.shape[1]
        expected_ctx = 2 * n_raw_feat
        assert n_ctx_feat == expected_ctx, (
            f"Expected {expected_ctx} context features, got {n_ctx_feat}"
        )

        frame_duration = self._audio_data.chunk_size / sr
        t_audio = np.arange(len(audio)) / sr
        t_frames = np.arange(n_frames) * frame_duration

        feature_names_selected = [
            "Mean ZCR",
            "Mean Centroid",
            "Mean Tonality",
            "Mean Peaks",
            "Mean Flux",
            "Mean DFR",
            "Mean Flatness",
            # "Mean Log Mel Energy",
            "Mean Log Energy",
            # "Mean RMS",
            # "Std ZCR",
            # "Std Centroid",
            # "Std Tonality",
            # "Std Peaks",
            # "Std Flux",
            # "Std Log Mel Energy",
            # "Std Log Energy",
            # "Std RMS",
            # "Std DFR",
            # "Std Flatness",
            "Var ZCR",
            "Var Centroid",
            "Var Tonality",
            "Var Peaks",
            # "Var Flux",
            # "Var Log Mel Energy",
            # "Var Log Energy",
            # "Var RMS",
            "Var DFR",
            "Var Flatness",
        ]

        colors_selected = [
            "blue",
            "orange",
            "green",
            "brown",
            "blue",
            "yellow",
            "purple",
            # "red",
            "red",
            # "red",
            # "blue",
            # "orange",
            # "green",
            # "brown",
            # "purple",
            # "red",
            # "red",
            # "red",
            # "blue",
            # "yellow",
            "blue",
            "orange",
            "green",
            "brown",
            "blue",
            "yellow",
            # "purple",
            # "red",
            # "red",
            # "red",
        ]

        fig_selected = make_subplots(
            rows=len(feature_names_selected) + 1,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=["Waveform"] + feature_names_selected,
            specs=[[{}]]
            + [[{"secondary_y": True}] for _ in range(len(feature_names_selected))],
        )

        for i, name, color in zip(
            range(self._selected_ctx_feat_vectors.shape[1]),
            feature_names_selected,
            colors_selected,
        ):
            # waveform
            fig_selected.add_trace(
                go.Scatter(
                    x=t_audio,
                    y=audio,
                    mode="lines",
                    name="Waveform",
                    line=dict(color="black", width=1),
                    opacity=0.2,
                ),
                row=i + 2,
                col=1,
                secondary_y=True,
            )

            # selected context feature
            fig_selected.add_trace(
                go.Scatter(
                    x=t_frames,
                    y=self._selected_ctx_feat_vectors[:, i],
                    mode="lines",
                    name=f"{name} (selected ctx)",
                    line=dict(width=1, color=color),
                ),
                row=i + 2,
                col=1,
                secondary_y=False,
            )

        fig_selected.update_layout(
            height=250 * (len(feature_names_selected) + 1),
            xaxis_title="Time (s)",
            title="Selected Context-Aggregated Features",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.01,
                xanchor="left",
                x=0.01,
            ),
        )

        fig_selected.update_yaxes(title_text="Feature Value", secondary_y=False)
        fig_selected.update_yaxes(title_text="Amplitude", secondary_y=True)

        fig_selected.show()


class PlayerWidgetAudioVisualizer(AbstractAudioVisualizer):
    def __init__(self, *, audio_data: AudioData) -> None:
        self._audio_data = audio_data

    def show(self) -> None:
        from torkius_vad.plotting.widgets import play

        play(self._audio_data.file_path)


class AudioVisualizer(AbstractAudioVisualizer):
    def __init__(
        self,
        visualizer: AbstractAudioVisualizer,
        *rest: AbstractAudioVisualizer,
    ) -> None:
        self._visualizers = (visualizer, *rest)

    def show(self) -> None:
        for visualizer in self._visualizers:
            visualizer.show()


# %% [markdown]
"""
### Local audio data visualization.

The next cell demonstrates the visualization of a single audio file from the dataset, showing the waveform, teacher probabilities, and predicted probabilities (if available). It also visualizes the selected context-aggregated features.

Modify the `file_path` in the `AudioData` initialization to visualize different audio samples from the dataset. The visualizer will display static and interactive plots of the audio data and features, as well as an audio player widget for listening to the sample.
"""

# %%
audio_data = AudioData(
    file_path="./audio_file_550.wav",
    target_sr=8000,
    chunk_size=int(0.01 * 8000),  # 10 ms chunks
)

audio_data = AudioLoader(print_stats=True).load(audio_data=audio_data)
audio_data = SileroProbabilityTeacher(print_stats=True).teach(
    audio_data=audio_data,
)

audio_data = AudioFeatureExtractor(print_stats=True).extract(
    audio_data=audio_data, frame_generator=AudioFrameGenerator()
)

assert audio_data.feat_vectors is not None
print(f"Visualizing sample: {audio_data.file_path}")

feature_compute_ctx.reset()
X = np.array(
    [
        feature_compute_ctx.compute(feat_vec=feat_vec)
        for feat_vec in audio_data.feat_vectors
    ],
    dtype=np.float32,
)
feature_compute_ctx.reset()

feature_compute_selector.reset()
X_selected = np.array(
    [
        feature_compute_selector.compute(feat_vec=feat_vec)
        for feat_vec in audio_data.feat_vectors
    ],
    dtype=np.float32,
)
feature_compute_selector.reset()

corr = np.corrcoef(X_selected, rowvar=False)

feature_names = list(feature_compute_selector.selection.keys())
print(feature_names)

for i in range(len(corr)):
    for j in range(i + 1, len(corr[i])):
        c = corr[i, j]
        if abs(c) > 0.8:
            print(f"{feature_names[i]:20s} <-> {feature_names[j]:20s} : {c:.3f}")

# %%
for model in models:
    print(model.coef_)
    print(model.intercept_)

# %%
audio_data = predictor.predict(audio_data=audio_data)

visualizer = AudioVisualizer(
    StaticPlotAudioVisualizer(audio_data=audio_data),
    InteractivePlotAudioVisualizer(audio_data=audio_data),
    InteractiveFeatureVisualizer(
        audio_data=audio_data,
        scaled_ctx_feat_vectors=X,
        selected_ctx_feat_vectors=scaler.transform(X_selected),
    ),
    PlayerWidgetAudioVisualizer(audio_data=audio_data),
)

visualizer.show()
del visualizer

# %% [markdown]
"""
### Random samples visualization.

The next cell randomly selects a few audio samples from the datasets and visualizes them using the defined visualizers. This allows you to explore different samples and see how the model's predictions compare to the teacher probabilities across various audio files.
"""

# %%
import random  # noqa: E402

sample_dataset_names = random.sample(datasets_meta.dataset_names, k=3)
for dataset_name in sample_dataset_names:
    dataset_meta = dataset_metas[dataset_name]
    print(f"\nShowing metadata for '{dataset_name}' dataset:")
    dataset_meta.show()

    print("Showing audio player for 2 random samples:")
    sample_slugs = random.sample(
        dataset_meta.dataset_meta["Slug"].tolist(),
        k=min(2, len(dataset_meta.dataset_meta)),
    )

    for slug in sample_slugs:
        file_path = datasets_meta.datasets_path / slug

        audio_data = AudioData(
            file_path=file_path.as_posix(),
            target_sr=8000,
            chunk_size=int(0.01 * 8000),  # 10 ms chunks
        )

        audio_data = AudioLoader(print_stats=True).load(audio_data=audio_data)
        audio_data = UnthresholdedSileroProbabilityTeacher(print_stats=True).teach(
            audio_data=audio_data,
        )

        audio_data = AudioFeatureExtractor(print_stats=True).extract(
            audio_data=audio_data, frame_generator=AudioFrameGenerator()
        )

        audio_data = predictor.predict(audio_data=audio_data)

        assert audio_data.feat_vectors is not None
        print(f"Visualizing sample: {audio_data.file_path}")

        feature_compute_ctx.reset()
        X = np.array(
            [
                feature_compute_ctx.compute(feat_vec=feat_vec)
                for feat_vec in audio_data.feat_vectors
            ],
            dtype=np.float32,
        )
        feature_compute_ctx.reset()

        feature_compute_selector.reset()
        X_selected = np.array(
            [
                feature_compute_selector.compute(feat_vec=feat_vec)
                for feat_vec in audio_data.feat_vectors
            ],
            dtype=np.float32,
        )
        feature_compute_selector.reset()

        visualizer = AudioVisualizer(
            StaticPlotAudioVisualizer(audio_data=audio_data),
            InteractivePlotAudioVisualizer(audio_data=audio_data),
            # InteractiveFeatureVisualizer(audio_data=audio_data),
            # InteractiveFeatureVisualizer(
            #     audio_data=audio_data,
            #     scaled_ctx_feat_vectors=X,
            #     selected_ctx_feat_vectors=scaler.transform(X_selected),
            # ),
            PlayerWidgetAudioVisualizer(audio_data=audio_data),
        )

        visualizer.show()
        del visualizer
