# %% [md]
"""
# Torkius VAD

WIP.

- [x] Interactive `PlayerWidget` for audio visualization.
- [x] Loading and resampling audio files.
- [x] Probability teacher (pseudo-labelling) with Silero VAD.
- [x] Static and interactive plotting of audio waveforms and VAD probabilities.
- [x] Find and download datasets.
- [ ] Save resampled audio array data loaded from datasets into `.npy` files.
- [ ] Render dataset statistics.
- [ ] Silence/non-silence teacher for speech-only datasets.
- [ ] Save taught probabilities into `.npy` files.
- [ ] Extract and stack features.
- [ ] Save features to `.npz` files.
- [ ] Arrange balanced training and testing splits.
- [ ] Model training pipeline.
- [ ] Model evaluation pipeline.

"""

# %% [md]
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

import numpy as np  # noqa: E402

from torkius_vad.plotting.widgets import play  # noqa: E402

# %%
play("audio_file_550.wav")
play("audio_file_550.wav")


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


# %%
loader = AudioLoader(print_stats=True)
teacher = SileroProbabilityTeacher(print_stats=True)

# %%
audio_data = AudioData(
    file_path="audio_file_713.wav",
    target_sr=8000,
    chunk_size=int(0.01 * 8000),  # 10 ms chunks
)

loaded_audio_data = loader.load(audio_data=audio_data)
taught_audio_data = teacher.teach(audio_data=loaded_audio_data)


# %%
class AbstractAudioVisualizer(ABC):
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

# %% [md]
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
