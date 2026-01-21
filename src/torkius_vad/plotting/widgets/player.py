from importlib.resources import files
import base64
import os
import json
import uuid

from IPython.display import HTML, DisplayHandle, display


class PlayerWidget:
    """
    Display an audio file waveform with media controls.
    Supports local file paths or URLs.
    """

    class PlayerWidgetConfig:
        title: str

        wavesurfer_js: str
        wavesurfer_zoom_js: str
        container_id: str
        time_id: str
        audio_url: str

        height: int
        wave_color: str
        progress_color: str
        cursor_color: str
        normalize: str
        media_controls: str
        max_zoom: int

        def __init__(self, **kwargs) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def __init__(
        self,
        audio_url: str,
        *,
        title: str | None = None,
        height: int = 200,
        wave_color: str = "#595959",
        progress_color: str = "#0b0b0b",
        cursor_color: str = "#000",
        normalize: bool = False,
        media_controls: bool = True,
        max_zoom: int = 10_000,
    ) -> None:
        """
        Initializes and pre-renders an audio player widget.

        Args:
            audio_url (str): Path to local audio file or URL.
            title (str | None): Title of the audio track.
            height (int): Height of the waveform display in pixels.
            wave_color (str): Color of the waveform.
            progress_color (str): Color of the progress bar.
            cursor_color (str): Color of the cursor.
            normalize (bool): Whether to normalize the waveform.
            media_controls (bool): Whether to show media controls.
            max_zoom (int): Maximum zoom level for the waveform.
        """

        if title is None:
            self._title = audio_url
        else:
            self._title = title

        _wf_dist = "https://unpkg.com/wavesurfer.js@7/dist/"
        self._wavesurfer_js = f"{_wf_dist}wavesurfer.esm.js"
        self._wavesurfer_zoom_js = f"{_wf_dist}plugins/zoom.esm.js"
        self._container_id = f"wavesurfer_{uuid.uuid4().hex}"
        self._time_id = f"time_{uuid.uuid4().hex}"
        self._audio_url = self._get_audio_url(audio_url)

        self._height = height
        self._wave_color = wave_color
        self._progress_color = progress_color
        self._cursor_color = cursor_color
        self._normalize = normalize
        self._media_controls = media_controls
        self._max_zoom = max_zoom

        self._template_path = (
            files("torkius_vad.plotting.widgets._templates") / "player_template.html"
        )

        self._rendered_template = self._render()

    def _encode_local_audio(self, audio_url: str) -> str:
        """
        Encode local file into a base64 data URL (for embedding).

        Args:
            audio_url (str): Path to local audio file.

        Returns:
            str: Base64-encoded data URL of the audio file.
        """

        if not os.path.exists(audio_url):
            raise FileNotFoundError(f"File not found: {audio_url}")

        ext = os.path.splitext(audio_url)[1].lower().replace(".", "")
        with open(audio_url, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:audio/{ext};base64,{b64}"

    def _get_audio_url(self, audio_url: str) -> str:
        """
        Return an audio URL or base64 embed if local file.

        Args:
            audio_url (str): Path to local audio file or URL.
        """

        if audio_url.startswith(("http://", "https://")):
            return audio_url
        else:
            return self._encode_local_audio(audio_url)

    def _render(self) -> str:
        """
        Render HTML + JS template that embeds wavesurfer.js and builds the player.

        Returns:
            str: Rendered HTML string.
        """

        template_html = self._template_path.read_text(encoding="utf-8")

        template = PlayerWidget.PlayerWidgetConfig(
            title=json.dumps(self._title).strip('"'),
            wavesurfer_js=json.dumps(self._wavesurfer_js),
            wavesurfer_zoom_js=json.dumps(self._wavesurfer_zoom_js),
            container_id=json.dumps(self._container_id).strip('"'),
            time_id=json.dumps(self._time_id).strip('"'),
            audio_url=json.dumps(self._audio_url),
            height=self._height,
            max_zoom=self._max_zoom,
            wave_color=json.dumps(self._wave_color),
            progress_color=json.dumps(self._progress_color),
            cursor_color=json.dumps(self._cursor_color),
            normalize="true" if self._normalize else "false",
            media_controls="true" if self._media_controls else "false",
        )

        return template_html.format(template=template)

    def show(self) -> DisplayHandle | None:
        """
        Display the player.

        Returns:
            DisplayHandle | None: Handle to the displayed output (if in Jupyter).
        """
        return display(HTML(self._rendered_template))


def play(
    audio_path: str,
    *,
    title: str | None = None,
    height: int = 200,
    wave_color: str = "#595959",
    progress_color: str = "#0b0b0b",
    cursor_color: str = "#000",
    normalize: bool = False,
    media_controls: bool = True,
    max_zoom: int = 10_000,
) -> DisplayHandle | None:
    """
    Initializes and pre-renders an audio player widget.

    Args:
        audio_url (str): Path to local audio file or URL.
        title (str | None): Title of the audio track.
        height (int): Height of the waveform display in pixels.
        wave_color (str): Color of the waveform.
        progress_color (str): Color of the progress bar.
        cursor_color (str): Color of the cursor.
        normalize (bool): Whether to normalize the waveform.
        media_controls (bool): Whether to show media controls.
        max_zoom (int): Maximum zoom level for the waveform.

    Returns:
        DisplayHandle | None: Handle to the displayed output (if in Jupyter).
    """
    return PlayerWidget(
        audio_path,
        title=title,
        height=height,
        wave_color=wave_color,
        progress_color=progress_color,
        cursor_color=cursor_color,
        normalize=normalize,
        media_controls=media_controls,
        max_zoom=max_zoom,
    ).show()
