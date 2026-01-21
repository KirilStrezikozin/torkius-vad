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
- [ ] Extract and stack features.
- [ ] Save features to `.npz` files.
- [ ] Arrange balanced training and testing splits.
- [ ] Model training pipeline.
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
from abc import ABC, abstractmethod  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from enum import StrEnum  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Literal, cast  # noqa: E402

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
    def dataset_meta(self) -> pd.DataFrame: ...

    @property
    @abstractmethod
    def dataset_path(self) -> Path: ...

    @property
    @abstractmethod
    def dataset_type(self) -> DatasetType: ...


class DatasetMeta(AbstractVisualizer, AbstractDatasetMeta):
    def __init__(
        self,
        *,
        dataset_name: str,
        datasets_meta: AbstractDatasetsMeta,
        use_disk_cache: bool = True,
        print_stats: bool = True,
    ) -> None:
        from time import time

        self._datasets_meta = datasets_meta

        self._check_dataset_name(dataset_name=dataset_name)
        self._dataset_name = dataset_name

        self._dataset_type = self._get_dataset_type(dataset_name=dataset_name)

        dataset_path = self._datasets_meta.datasets_path / dataset_name
        self._check_dataset_path(path=dataset_path)
        self._dataset_path = dataset_path

        if use_disk_cache:
            try:
                s0 = time()
                self._dataset_meta = pd.read_csv(
                    self._dataset_path / "dataset_meta.csv",
                )
                s1 = time()
                if print_stats:
                    print(
                        f"Dataset '{dataset_name}' metadata loaded from disk "
                        f"cache in {s1 - s0:.3f}s.",
                    )
                return
            except (FileNotFoundError, pd.errors.EmptyDataError):
                if print_stats:
                    print(
                        f"Disk cache for dataset '{dataset_name}' not found or "
                        f"invalid. Rebuilding metadata...",
                    )

        s0 = time()
        self._dataset_meta = self._build_meta()
        s1 = time()

        if print_stats:
            print(f"Dataset '{dataset_name}' metadata built in {s1 - s0:.3f}s.")

        if use_disk_cache:
            s0 = time()
            self._dataset_meta.to_csv(
                self._dataset_path / "dataset_meta.csv", index=False
            )
            s1 = time()
            if print_stats:
                print(
                    f"Dataset '{dataset_name}' metadata cached to disk in "
                    f"{s1 - s0:.3f}s.",
                )

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def dataset_meta(self) -> pd.DataFrame:
        return self._dataset_meta

    @property
    def dataset_path(self) -> Path:
        return self._dataset_path

    @property
    def dataset_type(self) -> DatasetType:
        return self._dataset_type

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
        display(self._dataset_meta)

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

        if self._print_stats:
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
    def load(self, *, dataset_meta: DatasetMeta) -> list[AudioData]: ...


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

    def load(self, *, dataset_meta: DatasetMeta) -> list[AudioData]:
        from time import time

        datasets_dir = dataset_meta.dataset_path.parent
        audio_data_list: list[AudioData] = []
        build = True

        if self._use_disk_cache:
            build = False
            s0 = time()

            if not self._cache_dir.exists():
                self._cache_dir.mkdir(parents=True, exist_ok=True)

            for slug in dataset_meta.dataset_meta["Slug"]:
                cached_audio_path = self._cache_dir / Path(slug).with_suffix(".npy")
                if not cached_audio_path.exists():
                    audio_data_list.clear()
                    build = True
                    if self._print_stats:
                        print(
                            f"Cache file '{cached_audio_path}' not found. "
                            f"Rebuilding audio data from source files...",
                        )
                    break

                cached_audio = np.load(cached_audio_path, mmap_mode=self._mmap_mode)
                audio_data = AudioData(
                    file_path=(datasets_dir / slug).as_posix(),
                    target_sr=self._target_sr,
                    chunk_size=self._chunk_size,
                    audio=cached_audio,
                    sr=self._target_sr,
                )
                audio_data_list.append(audio_data)

            s1 = time()
            if not build and self._print_stats:
                print(
                    f"Loaded {len(audio_data_list)} audio files from dataset "
                    f"'{dataset_meta.dataset_name}' cache in {s1 - s0:.3f}s."
                )

        if not build:
            return audio_data_list

        s0 = time()
        for slug in dataset_meta.dataset_meta["Slug"]:
            file_path = datasets_dir / slug
            audio_data = AudioData(
                file_path=file_path.as_posix(),
                target_sr=self._target_sr,
                chunk_size=self._chunk_size,
            )
            loaded_audio_data = self._audio_loader.load(audio_data=audio_data)

            if loaded_audio_data.audio is None:
                raise ValueError(
                    f"Loaded audio data from file '{file_path}' "
                    f"does not contain audio samples.",
                )
            elif loaded_audio_data.sr != self._target_sr:
                # Because when loading from cache, we assume it's at target_sr.
                raise ValueError(
                    f"Loaded audio data from file '{file_path}' "
                    f"has invalid sampling rate "
                    f"({loaded_audio_data.sr} != {self._target_sr}).",
                )

            audio_data_list.append(loaded_audio_data)

            if self._use_disk_cache:
                cached_audio_path = self._cache_dir / Path(slug).with_suffix(".npy")
                if not cached_audio_path.parent.exists():
                    cached_audio_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cached_audio_path, loaded_audio_data.audio, allow_pickle=False)

        s1 = time()
        if self._print_stats:
            print(
                f"Loaded {len(audio_data_list)} audio files from dataset "
                f"'{dataset_meta.dataset_name}' in {s1 - s0:.3f}s."
            )

        return audio_data_list


# %%
class AbstractDatasetAudioTeacher(ABC):
    @abstractmethod
    def teach(
        self,
        *,
        dataset_meta: DatasetMeta,
        audio_data_list: list[AudioData],
    ) -> list[AudioData]: ...


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
        audio_data_list: list[AudioData],
    ) -> list[AudioData]:
        from time import time

        datasets_dir = dataset_meta.dataset_path.parent
        taught_audio_data_list: list[AudioData] = []
        build = True

        if self._use_disk_cache:
            build = False
            s0 = time()

            if not self._cache_dir.exists():
                self._cache_dir.mkdir(parents=True, exist_ok=True)

            for audio_data in audio_data_list:
                slug = Path(audio_data.file_path).relative_to(datasets_dir)
                cached_probas_path = self._cache_dir / slug.with_stem(
                    f"{slug.stem}_probas"
                ).with_suffix(".npy")
                if not cached_probas_path.exists():
                    taught_audio_data_list.clear()
                    build = True
                    if self._print_stats:
                        print(
                            f"Cache file '{cached_probas_path}' not found. "
                            f"Rebuilding taught probabilities from source files...",
                        )
                    break

                cached_probas = np.load(cached_probas_path, mmap_mode=self._mmap_mode)
                taught_audio_data = audio_data.with_taught_probas(
                    taught_probas=cached_probas,
                )
                taught_audio_data_list.append(taught_audio_data)

            s1 = time()
            if not build and self._print_stats:
                print(
                    f"Loaded taught probabilities for "
                    f"{len(taught_audio_data_list)} audio files in dataset "
                    f"'{dataset_meta.dataset_name}' cache in {s1 - s0:.3f}s."
                )

        if not build:
            return taught_audio_data_list

        s0 = time()
        for audio_data in audio_data_list:
            taught_audio_data = self._probability_teacher.teach(audio_data=audio_data)
            if taught_audio_data.taught_probas is None:
                raise ValueError(
                    f"Taught audio data from file '{audio_data.file_path}' "
                    f"does not contain taught probabilities.",
                )

            taught_audio_data_list.append(taught_audio_data)

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

        s1 = time()
        if self._print_stats:
            print(
                f"Taught probabilities for {len(taught_audio_data_list)} audio files "
                f"in dataset '{dataset_meta.dataset_name}' in {s1 - s0:.3f}s."
            )

        return taught_audio_data_list


# %%
dataset_loader = DatasetAudioLoader(
    audio_loader=AudioLoader(print_stats=False),
    target_sr=8000,
    chunk_size=int(0.01 * 8000),  # 10 ms chunks
    use_disk_cache=True,
    print_stats=True,
)

nonspeech_dataset_teacher = DatasetAudioTeacher(
    probability_teacher=NonSpeechProbabilityTeacher(print_stats=False),
    use_disk_cache=True,
    print_stats=True,
)

mix_dataset_teacher = DatasetAudioTeacher(
    probability_teacher=SileroProbabilityTeacher(print_stats=False),
    use_disk_cache=True,
    print_stats=True,
)

dataset_teachers = {
    "mix_ava": None,
    "mix_private_telephony": mix_dataset_teacher,
    "mix_voxconverse_test": mix_dataset_teacher,
    "nonspeech_esc_50": nonspeech_dataset_teacher,
    "nonspeech_musan_music_rmf": nonspeech_dataset_teacher,
    "nonspeech_musan_noise": nonspeech_dataset_teacher,
    "speech_callhome_deu": None,
    "speech_musan_speech": None,
}

# %%
datasets_meta.show(groups=False)

# %%
audio_data_lists = {}
for dataset_name, dataset_meta in dataset_metas.items():
    dataset_teacher = dataset_teachers.get(dataset_name, None)
    if dataset_teacher is None:
        print("Skipping speech dataset for teaching probabilities.")
        continue

    audio_data_list = dataset_loader.load(
        dataset_meta=dataset_meta,
    )

    audio_data_list = dataset_teacher.teach(
        dataset_meta=dataset_meta,
        audio_data_list=audio_data_list,
    )

    audio_data_lists[dataset_name] = audio_data_list


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

        fig.update_layout(
            height=500,
            xaxis=dict(title="Time (s)"),
            yaxis=dict(title="Amplitude"),
            yaxis2=dict(
                title="Speech Probability from Teacher", overlaying="y", side="right"
            ),
            legend=dict(x=0.01, y=500, bordercolor="Black", borderwidth=1),
        )

        fig.show()


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
visualizer = AudioVisualizer(
    StaticPlotAudioVisualizer(audio_data=taught_audio_data),
    InteractivePlotAudioVisualizer(audio_data=taught_audio_data),
    PlayerWidgetAudioVisualizer(audio_data=taught_audio_data),
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
