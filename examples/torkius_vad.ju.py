# %% [markdown]
"""
# Torkius VAD

WIP.

- [x] Interactive `PlayerWidget` for audio visualization.
- [x] Loading and resampling audio files.
- [x] Probability teacher (pseudo-labelling) with Silero VAD.
- [x] Static and interactive plotting of audio waveforms and VAD probabilities.
- [x] Find and download datasets.
- [x] Save resampled audio array data loaded from datasets into `.npy` files.
- [x] Render all dataset metadata statistics.
- [x] Render individual dataset metadata statistics with player.
- [x] Save taught probabilities into `.npy` files.
- [x] Extract features
- [x] Save features to `.npy` files.
- [x] Stack features.
- [x] Arrange balanced training and testing splits.
- [x] Model training pipeline.
- [ ] Model evaluation pipeline.
"""

# %% [markdown]
"""
### Notebook imports.

Import of required libraries and modules to run cells in this notebook.
"""


# %%
def _configure_plotly_classic() -> None:
    import plotly.io as pio  # noqa: E402

    pio.renderers.default = "notebook_connected"


_configure_plotly_classic()  # If plotly charts don't render.

# %%
import torch  # noqa: E402

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# %%
from abc import ABC, abstractmethod  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from enum import StrEnum  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable, Generator, Literal, NamedTuple, cast  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from IPython.display import display  # noqa: E402

from torkius_vad.plotting.widgets import play  # noqa: E402


# %%
class AbstractVisualizer(ABC):
    """
    Abstract visualizer for data that can display any kind of visualization.
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


# %%
datasets_meta = DatasetsMeta()
datasets_meta.show()


# %%
class DatasetType(StrEnum):
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

    def show_player(self, *, random_n: int = 1) -> None:
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

# %%
import random  # noqa: E402

sample_dataset_names = random.sample(datasets_meta.dataset_names, k=3)
for dataset_name in sample_dataset_names:
    dataset_meta = dataset_metas[dataset_name]
    print(f"\nShowing metadata for '{dataset_name}' dataset:")
    dataset_meta.show()

    print("Showing audio player for 2 random samples:")
    dataset_meta.show_player(random_n=2)


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


class MixAvaDatasetProbabilityTeacher(AbstractProbabilityTeacher):
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
        print_stats: bool = True,
    ) -> None:
        self._probability_teacher = probability_teacher
        self._use_disk_cache = use_disk_cache
        self._cache_dir = cache_dir
        self._mmap_mode: Literal["r", "r+", "w+", "c"] | None = mmap_mode
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
                if taught_audio_data.taught_probas.shape != (num_chunks,):
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
        for frame in frame_generator.generate(audio_data=audio_data):
            feat_vec, prev_fft_spectrum = self._calc_feat_vec(
                frame=frame,
                sr=audio_data.sr,
                prev_fft_spectrum=prev_fft_spectrum,
            )
            feat_vectors.append(feat_vec)

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
    ) -> tuple[np.ndarray, np.ndarray]:
        from scipy.fft import rfft

        frame = frame * self._window

        log_energy = np.log(np.sum(frame**2) + self._eps)
        zcr = np.mean(frame[:-1] * frame[1:] < 0.0)

        fft_spectrum = np.abs(rfft(frame, n=self._n_fft)) + self._eps
        fft_spectrum /= np.sum(fft_spectrum)

        freqs = np.linspace(0, sr / 2, num=len(fft_spectrum))

        centroid = np.sum(freqs * fft_spectrum)
        flatness = np.exp(np.mean(np.log(fft_spectrum))) / (
            np.mean(fft_spectrum) + self._eps
        )
        tonality = np.log(np.max(fft_spectrum) / (np.mean(fft_spectrum) + self._eps))
        rolloff_idx = np.searchsorted(np.cumsum(fft_spectrum), 0.85)
        rolloff_idx = min(rolloff_idx, len(freqs) - 1)
        rolloff = freqs[rolloff_idx]
        lf_energy = np.sum(fft_spectrum[freqs <= 300.0])
        hf_energy = np.sum(fft_spectrum[freqs >= 300.0])
        lf_ratio = lf_energy / (hf_energy + self._eps)
        peaks = np.sum(fft_spectrum > 0.1 * np.max(fft_spectrum))

        if prev_fft_spectrum is None:
            flux = 0.0
        else:
            diff = fft_spectrum - prev_fft_spectrum
            flux = (
                np.sum(diff * diff) / (np.sum(prev_fft_spectrum**2) + self._eps)
                + self._eps
            )

        mel_energy = self._mel_filter_bank @ fft_spectrum
        log_mel_energy = np.log(mel_energy + self._eps)

        feat_vec = np.hstack(
            [
                log_energy,
                zcr,
                centroid,
                flatness,
                tonality,
                rolloff,
                lf_ratio,
                peaks,
                flux,
                log_mel_energy,
            ]
        ).astype(np.float32)

        return feat_vec, fft_spectrum


# %%
class Scaler:
    def __init__(self, n_features: int):
        self.n_features = n_features
        self.count = 0
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.M2 = np.zeros(
            n_features, dtype=np.float64
        )  # Sum of squares of differences

    def partial_fit(self, X: np.ndarray) -> None:
        for x in X:
            self.count += 1
            delta = x - self.mean
            self.mean += delta / self.count
            delta2 = x - self.mean
            self.M2 += delta * delta2

    def transform(self, X: np.ndarray) -> np.ndarray:
        std = np.sqrt(self.M2 / np.maximum(self.count - 1, 1))
        std[std < 1e-8] = 1.0  # avoid division by zero
        return (X - self.mean) / std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.partial_fit(X)
        return self.transform(X)


# %%
class AbstractFeatureWithContextStats(ABC):
    @abstractmethod
    def compute(self, *, feat_vec: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def reset(self) -> None: ...


class FeatureWithContextStats(AbstractFeatureWithContextStats):
    def __init__(self, *, context_size: int = 5) -> None:
        from collections import deque

        self._context_size = context_size
        self._buf = deque(maxlen=context_size + 1)

    def compute(self, *, feat_vec: np.ndarray) -> np.ndarray:
        import numpy as np

        self._buf.append(feat_vec)
        buf_array = np.vstack(self._buf)

        mean_vec = np.mean(buf_array, axis=0)
        std_vec = np.std(buf_array, axis=0)
        min_vec = np.min(buf_array, axis=0)
        max_vec = np.max(buf_array, axis=0)

        feat_with_stats = np.hstack(
            [
                feat_vec,
                mean_vec,
                std_vec,
                min_vec,
                max_vec,
            ]
        ).astype(np.float32)
        return feat_with_stats

    def reset(self) -> None:
        self._buf.clear()


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


# %%
dataset_loader = DatasetAudioLoader(
    audio_loader=AudioLoader(print_stats=False),
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
        X_ctx: AbstractFeatureWithContextStats,
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
                X_ctx.reset()  # Reset context stats for next audio file.
                continue

            fmt_err = "Audio data must contain {} for final processing."
            if audio_data.taught_probas is None:
                raise ValueError(fmt_err.format("taught probabilities"))
            elif audio_data.feat_vectors is None:
                raise ValueError(fmt_err.format("feature vectors"))

            X_ctx.reset()
            X = np.array(
                [
                    X_ctx.compute(feat_vec=feat_vec)
                    for feat_vec in audio_data.feat_vectors
                ],
                dtype=np.float32,
            )
            X_ctx.reset()
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


# %%
from sklearn.linear_model import SGDClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


class AbstractOfflinePredictor(ABC):
    @abstractmethod
    def predict(self, *, audio_data: AudioData) -> AudioData: ...


class AbstractPredictionModel(ABC):
    @abstractmethod
    def decision_function(self, *, X: np.ndarray) -> np.ndarray: ...


class SGDClassifierModel(AbstractPredictionModel):
    def __init__(self, *, model: SGDClassifier) -> None:
        self._model = model

    def decision_function(self, *, X: np.ndarray) -> np.ndarray:
        return self._model.decision_function(X)


class SGDEnsembleModel(AbstractPredictionModel):
    def __init__(self, *, models: list[SGDClassifier]) -> None:
        self._models = models

    def decision_function(self, *, X: np.ndarray) -> np.ndarray:
        import numpy as np

        all_preds = np.array(
            [model.decision_function(X) for model in self._models],
            dtype=np.float32,
        )
        avg_preds = np.mean(all_preds, axis=0)
        return avg_preds


class BaseOfflineSGDPredictor(AbstractOfflinePredictor):
    def __init__(
        self,
        *,
        model: AbstractPredictionModel,
        scaler: StandardScaler,
        X_ctx: AbstractFeatureWithContextStats,
        print_stats: bool = True,
    ) -> None:
        self._model = model
        self._scaler = scaler
        self._X_ctx = X_ctx
        self._print_stats = print_stats

        self._avg_predict_n = 0
        self._avg_predict_time = 0.0

    def decision_function(self, *, X: np.ndarray) -> np.ndarray:
        return self._model.decision_function(X=X)

    def predict(self, *, audio_data: AudioData) -> AudioData:
        fmt_err = "Audio data must contain {} for prediction."
        if audio_data.feat_vectors is None:
            raise ValueError(fmt_err.format("feature vectors"))

        self._X_ctx.reset()
        X = np.array(
            [
                self._X_ctx.compute(feat_vec=feat_vec)
                for feat_vec in audio_data.feat_vectors
            ],
            dtype=np.float32,
        )
        self._X_ctx.reset()

        s0 = time()
        X = self._scaler.transform(X)
        y_pred = self.decision_function(X=X)
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
            backend="loky",
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


# %%
datasets_meta.show(groups=True)

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
    n_jobs=6,
)

pipeline_manager.run()

# %%
from sklearn.linear_model import SGDClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

scaler = StandardScaler()

models = [
    SGDClassifier(
        loss="log_loss",
        learning_rate="optimal",
        alpha=1e-5,
        penalty="l2",
        warm_start=True,
    )
    for _ in range(5)
]

# %%

from time import time  # noqa: E402

X_ctx = FeatureWithContextStats(context_size=5)  # One instance is safe.
learning_pipeline = DatasetPartialLearningPipeline(print_stats=False)

SKIP = True
NOT_SKIP = False

learners = {
    "mix_ava": (
        dataset_metas["mix_ava"],
        NOT_SKIP,
        learning_pipeline.process(
            dataset_meta=dataset_metas["mix_ava"].shuffled(
                random_state=42,
            ),
            dataset_loader=dataset_loader,
            dataset_teacher=make_mix_dataset_teacher(),
            dataset_feature_extractor=dataset_feature_extractor,
            X_ctx=X_ctx,
            train_split=0.9,
            skip_first=0.0,
        ),
    ),
    "mix_private_telephony": (
        dataset_metas["mix_private_telephony"],
        NOT_SKIP,
        learning_pipeline.process(
            dataset_meta=dataset_metas["mix_private_telephony"].shuffled(
                random_state=42,
            ),
            dataset_loader=dataset_loader,
            dataset_teacher=make_mix_dataset_teacher(),
            dataset_feature_extractor=dataset_feature_extractor,
            X_ctx=X_ctx,
            train_split=0.9,
            skip_first=0.0,
        ),
    ),
    "mix_voxconverse_test": (
        dataset_metas["mix_voxconverse_test"],
        NOT_SKIP,
        learning_pipeline.process(
            dataset_meta=dataset_metas["mix_voxconverse_test"].shuffled(
                random_state=42,
            ),
            dataset_loader=dataset_loader,
            dataset_teacher=make_mix_dataset_teacher(),
            dataset_feature_extractor=dataset_feature_extractor,
            X_ctx=X_ctx,
            train_split=0.9,
            skip_first=0.0,
        ),
    ),
    "nonspeech_esc_50": (
        dataset_metas["nonspeech_esc_50"],
        NOT_SKIP,
        learning_pipeline.process(
            dataset_meta=dataset_metas["nonspeech_esc_50"].shuffled(
                random_state=42,
            ),
            dataset_loader=dataset_loader,
            dataset_teacher=make_nonspeech_dataset_teacher(),
            dataset_feature_extractor=dataset_feature_extractor,
            X_ctx=X_ctx,
            train_split=0.9,
            skip_first=0.0,
        ),
    ),
    "nonspeech_musan_music_rmf": (
        dataset_metas["nonspeech_musan_music_rmf"],
        NOT_SKIP,
        learning_pipeline.process(
            dataset_meta=dataset_metas["nonspeech_musan_music_rmf"].shuffled(
                random_state=42,
            ),
            dataset_loader=dataset_loader,
            dataset_teacher=make_nonspeech_dataset_teacher(),
            dataset_feature_extractor=dataset_feature_extractor,
            X_ctx=X_ctx,
            train_split=0.9,
            skip_first=0.0,
        ),
    ),
    "nonspeech_musan_noise": (
        dataset_metas["nonspeech_musan_noise"],
        NOT_SKIP,
        learning_pipeline.process(
            dataset_meta=dataset_metas["nonspeech_musan_noise"].shuffled(
                random_state=42,
            ),
            dataset_loader=dataset_loader,
            dataset_teacher=make_nonspeech_dataset_teacher(),
            dataset_feature_extractor=dataset_feature_extractor,
            X_ctx=X_ctx,
            train_split=0.9,
            skip_first=0.0,
        ),
    ),
    "speech_callhome_deu": (
        dataset_metas["speech_callhome_deu"],
        NOT_SKIP,
        learning_pipeline.process(
            dataset_meta=dataset_metas["speech_callhome_deu"].shuffled(
                random_state=42,
            ),
            dataset_loader=dataset_loader,
            dataset_teacher=make_mix_dataset_teacher(),
            dataset_feature_extractor=dataset_feature_extractor,
            X_ctx=X_ctx,
            train_split=0.9,
            skip_first=0.0,
        ),
    ),
    "speech_musan_speech": (
        dataset_metas["speech_musan_speech"],
        NOT_SKIP,
        learning_pipeline.process(
            dataset_meta=dataset_metas["speech_musan_speech"].shuffled(
                random_state=42,
            ),
            dataset_loader=dataset_loader,
            dataset_teacher=make_mix_dataset_teacher(),
            dataset_feature_extractor=dataset_feature_extractor,
            X_ctx=X_ctx,
            train_split=0.9,
            skip_first=0.0,
        ),
    ),
}

# %%
# data = []
# for X, y, sample_metadata in learners["mix_private_telephony"]():
#     print(sample_metadata.is_train)
#     data.append((X, y, sample_metadata))

# %%

s0 = time()
for dataset_meta, skip_dataset, learner in learners.values():
    if skip_dataset:
        print(f"\nSkipping dataset '{dataset_meta.dataset_name}'.")
        continue

    print(
        f"\nLearning on dataset '{dataset_meta.dataset_name}' {dataset_meta.dataset_type}:"
    )

    for X, y, sample_metadata in learner:
        # print(
        #     f"Sample: {sample_metadata.samples_slug} "
        #     f"({sample_metadata.samples_index + 1}/"
        #     f"{sample_metadata.samples_total}) "
        #     f"{'TRAIN' if sample_metadata.is_train else 'VAL'} "
        #     f"X.shape={X.shape} y.shape={y.shape}",
        # )

        t0 = time()
        if dataset_meta.dataset_type != DatasetType.NONSPEECH:
            scaler.partial_fit(X)
        X = scaler.transform(X)

        for clf in models:
            clf.partial_fit(X, y, classes=[0.0, 1.0])

        # print(f"Fitted in {time() - t0:.3f}s.")
        print(end="")
        print(
            f"{dataset_meta.dataset_name}: "
            f"Sample {sample_metadata.samples_index + 1}/"
            f"{sample_metadata.samples_total} ",
            end="\r",
        )

        if not sample_metadata.is_next_train:
            break

    print("")

print(f"Total learning time: {time() - s0:.3f}s.")

# %%
import pickle  # noqa: E402
from datetime import datetime  # noqa: E402

with open(f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl", "wb") as f:
    pickle.dump(
        {
            "scaler": scaler,
            "classifier": models,
            "comment": "SGDClassifier trained on 'mix_private_telephony' dataset.",
        },
        f,
    )

# %%
pmeta = DatasetMeta(
    dataset_name="mix_private_telephony",
    datasets_meta=datasets_meta,
    dataset_mask=(
        dataset_metas["mix_private_telephony"].dataset_meta["Slug"]
        == "mix_private_telephony/synthetic/audio_file_634.wav"
    ),
    print_stats=True,
)

DatasetAudioPipeline(print_stats=True).process(
    dataset_meta=pmeta,
    dataset_loader=dataset_loader,
    dataset_teacher=make_mix_dataset_teacher(),
    dataset_feature_extractor=dataset_feature_extractor,
)

# %%
s0 = time()

predictor = BaseOfflineSGDPredictor(
    # model=SGDClassifierModel(model=models[0]),
    model=SGDEnsembleModel(models=models),
    scaler=scaler,
    X_ctx=X_ctx,
    print_stats=False,
)

predicted = []
for dataset_meta, skip_dataset, learner in learners.values():
    if skip_dataset or dataset_meta.dataset_name != "mix_private_telephony":
        print(f"\nSkipping dataset '{dataset_meta.dataset_name}'.")
        continue

    print(
        f"\nPredicting on dataset '{dataset_meta.dataset_name}' "
        f"{dataset_meta.dataset_type}:"
    )

    for X, y, sample_metadata in learner:
        if sample_metadata.is_train:
            print(end="")
            print(
                f"{dataset_meta.dataset_name}: "
                f"Skipped sample {sample_metadata.samples_index + 1}/"
                f"{sample_metadata.samples_total} ",
                end="\r",
            )
            continue

        # print(
        #     f"Sample: {sample_metadata.samples_slug} "
        #     f"({sample_metadata.samples_index + 1}/"
        #     f"{sample_metadata.samples_total}) "
        #     f"{'TRAIN' if sample_metadata.is_train else 'VAL'} "
        #     f"X.shape={X.shape} y.shape={y.shape}",
        # )

        predicted.append(predictor.predict(audio_data=sample_metadata.audio_data))

        print(end="")
        print(
            f"{dataset_meta.dataset_name}: "
            f"Sample {sample_metadata.samples_index + 1}/"
            f"{sample_metadata.samples_total} ",
            end="\r",
        )
    print("")


print(f"Total prediction time: {time() - s0:.3f}s.")

# %%
predicted1 = []
print(len(predicted))
for audio_data in predicted:
    if not audio_data.file_path.startswith("mix_private_telephony"):
        print(f"\nSkipping file '{audio_data.file_path}'.")
        continue

    predicted1.append(predictor.predict(audio_data=audio_data))


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
        ax2.set_ylabel("Speech Probability from Teacher")

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

        plt.title(f"{self._audio_data.file_path}: Waveform + Teacher VAD")
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
            ),
            legend=dict(x=0.01, y=500, bordercolor="Black", borderwidth=1),
        )

        fig.show()


# %%
class InteractiveFeatureVisualizer(AbstractAudioVisualizer):
    def __init__(self, *, audio_data: AudioData) -> None:
        fmt_err = "Audio data must contain {} for visualization."
        if audio_data.audio is None:
            raise ValueError(fmt_err.format("audio samples"))
        elif audio_data.sr is None:
            raise ValueError(fmt_err.format("sampling rate"))
        elif audio_data.feat_vectors is None:
            raise ValueError(fmt_err.format("feature vectors"))
        self._audio_data = audio_data

    def show(self) -> None:
        import numpy as np
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        audio = self._audio_data.audio
        assert audio is not None
        sr = self._audio_data.sr
        assert sr is not None
        feat = self._audio_data.feat_vectors
        assert feat is not None
        chunk_size = self._audio_data.chunk_size

        frames, feat_dim = feat.shape
        frame_duration = chunk_size / sr

        t_audio = np.arange(len(audio)) / sr
        t_frames = np.arange(frames) * frame_duration

        # --- feature indexing ---
        scalar_names = [
            "Log Energy",
            "ZCR",
            "Centroid",
            "Flatness",
            "Tonality",
            "Rolloff",
            "LF/HF Ratio",
            "Peaks",
            "Flux",
        ]
        n_scalar = len(scalar_names)
        mel = feat[:, n_scalar:]

        # --- figure with subplots ---
        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(
                "Waveform",
                "Energy / Temporal",
                "Spectral Shape",
                "Dynamics",
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
            (0, "Log Energy", "red"),
            (1, "ZCR", "blue"),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=t_frames,
                    y=feat[:, idx],
                    mode="lines",
                    name=name,
                    opacity=0.6,
                    line=dict(width=2),
                ),
                row=2,
                col=1,
            )

        # --- spectral shape ---
        for idx, name in [
            (2, "Centroid"),
            (3, "Flatness"),
            (4, "Tonality"),
            (5, "Rolloff"),
            (6, "LF/HF Ratio"),
            (7, "Peaks"),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=t_frames,
                    y=feat[:, idx],
                    mode="lines",
                    name=name,
                    opacity=0.6,
                ),
                row=3,
                col=1,
            )

        # --- dynamics ---
        fig.add_trace(
            go.Scatter(
                x=t_frames,
                y=feat[:, 8],
                mode="lines",
                name="Flux",
                opacity=0.6,
                line=dict(color="purple"),
            ),
            row=4,
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
            title="Audio + Scalar Features",
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
            "Log Energy",
            "ZCR",
            "Centroid",
            "Flatness",
            "Tonality",
            "Rolloff",
            "LF/HF Ratio",
            "Peaks",
            "Flux",
        ]
        n_scalar = len(scalar_names)
        mel = feat[:, n_scalar:]  # everything beyond scalars

        # --- figure with subplots ---
        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(
                "Waveform",
                "Energy / Temporal",
                "Spectral Shape",
                "Dynamics",
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
        for idx, name, color in [(0, "Log Energy", "red"), (1, "ZCR", "blue")]:
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

        # --- spectral shape ---
        for idx, name, color in zip(
            range(2, 8),
            scalar_names[2:],
            ["green", "orange", "purple", "brown", "pink", "gray"],
        ):
            fig.add_trace(
                go.Scatter(
                    x=t_frames,
                    y=feat[:, idx],
                    mode="lines",
                    name=name,
                    opacity=0.7,
                    line=dict(color=color, width=2),
                ),
                row=3,
                col=1,
            )

        # --- dynamics ---
        fig.add_trace(
            go.Scatter(
                x=t_frames,
                y=feat[:, 8],
                mode="lines",
                name="Flux",
                opacity=0.7,
                line=dict(color="magenta", width=2),
            ),
            row=4,
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

        # ===== scaled log-mel heatmap =====
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
                colorbar=dict(title="Scaled Mel Bands"),
                opacity=0.9,
                yaxis="y2",
            )
        )

        fig_mel.update_layout(
            height=600,
            xaxis=dict(title="Time (s)"),
            yaxis=dict(title="Amplitude"),
            yaxis2=dict(
                title="Mel Band Index",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
            title="Waveform + Scaled Mel Spectrogram",
        )

        fig_mel.show()


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


# %%
for i, audio_data in enumerate(predicted):
    print(f"Sample {i}: {audio_data.file_path}")

# %%
audio_data = AudioData(
    file_path="./audio_file_713.wav",
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

print(f"Visualizing sample: {audio_data.file_path}")

visualizer = AudioVisualizer(
    StaticPlotAudioVisualizer(audio_data=audio_data),
    InteractivePlotAudioVisualizer(audio_data=audio_data),
    InteractiveFeatureVisualizer(audio_data=audio_data),
    InteractiveScaledFeatureVisualizer(
        audio_data=audio_data,
        scaled_feat_vectors=Scaler(
            n_features=audio_data.feat_vectors.shape[1]
        ).fit_transform(audio_data.feat_vectors),
    ),
    PlayerWidgetAudioVisualizer(audio_data=audio_data),
)

visualizer.show()

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

print(f"Visualizing sample: {audio_data.file_path}")

visualizer = AudioVisualizer(
    StaticPlotAudioVisualizer(audio_data=audio_data),
    InteractivePlotAudioVisualizer(audio_data=audio_data),
    InteractiveFeatureVisualizer(audio_data=audio_data),
    InteractiveScaledFeatureVisualizer(
        audio_data=audio_data,
        scaled_feat_vectors=Scaler(
            n_features=audio_data.feat_vectors.shape[1]
        ).fit_transform(audio_data.feat_vectors),
    ),
    PlayerWidgetAudioVisualizer(audio_data=audio_data),
)

visualizer.show()

# %% [markdown]
"""
### Datasets

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

### Training

Take all or a portion of each dataset for training.

### Testing

Take a portion from private telephony dataset not used in training.
"""
