"""MP4 encoding, probing and assembly for Stage 5.

Encoding backends, chosen explicitly and reported back through ``video.json``:

* ``ffmpeg``  - an external FFmpeg executable, found via ``PATH`` plus a short
  list of well-known locations. Encodes H.264 (``libx264``, ``yuv420p``) by
  piping raw RGB frames.
* ``opencv``  - the bundled OpenCV/FFmpeg writer, used when no executable is
  available. It picks the first fourcc that actually round-trips (H.264 when
  possible, MPEG-4 Part 2 otherwise).

Nothing here pretends a byte blob is a video: every clip and the assembled
file are re-opened and probed before they are reported as playable.

No network access is involved in any path.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

BACKEND_AUTO = "auto"
BACKEND_FFMPEG = "ffmpeg"
BACKEND_OPENCV = "opencv"
BACKENDS = (BACKEND_AUTO, BACKEND_FFMPEG, BACKEND_OPENCV)

FFMPEG_CANDIDATES = (
    r"C:\ffmpeg\bin\ffmpeg.exe",
    r"%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe",
    r"%PROGRAMFILES%\ffmpeg\bin\ffmpeg.exe",
    "/usr/bin/ffmpeg",
    "/usr/local/bin/ffmpeg",
    "/opt/homebrew/bin/ffmpeg",
)


class VideoEncodingError(RuntimeError):
    """Raised when frames cannot be encoded into a real, playable MP4."""


def detect_ffmpeg() -> str | None:
    """Explicit local FFmpeg executable detection. Returns ``None`` if absent."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    for candidate in FFMPEG_CANDIDATES:
        expanded = os.path.expandvars(candidate)
        if Path(expanded).exists():
            return expanded
    return None


@lru_cache(maxsize=1)
def probe_opencv_fourcc() -> str:
    """Return the first OpenCV fourcc that round-trips on this machine."""
    for fourcc in ("avc1", "mp4v"):
        path = Path(tempfile.gettempdir()) / f"stage5_probe_{fourcc}.mp4"
        try:
            path.unlink(missing_ok=True)
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), 30.0, (64, 64))
            if not writer.isOpened():
                continue
            for index in range(3):
                frame = np.full((64, 64, 3), (index * 40, 90, 140), dtype=np.uint8)
                writer.write(frame)
            writer.release()
            capture = cv2.VideoCapture(str(path))
            read = 0
            while True:
                ok, _frame = capture.read()
                if not ok:
                    break
                read += 1
            capture.release()
            if read >= 3:
                return fourcc
        except cv2.error:  # pragma: no cover - driver specific
            continue
        finally:
            path.unlink(missing_ok=True)
    raise VideoEncodingError(
        "OpenCV could not encode a readable MP4 with any supported fourcc "
        "(tried 'avc1' and 'mp4v'); install an FFmpeg executable or check the "
        "OpenCV video I/O build."
    )


class FrameWriter:
    """Minimal frame sink interface used by the scene renderer."""

    backend: str = "abstract"

    def write(self, rgb: np.ndarray) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> dict:  # pragma: no cover - interface
        raise NotImplementedError


class OpenCVFrameWriter(FrameWriter):
    backend = "opencv"

    def __init__(self, path: str | Path, width: int, height: int, fps: int, fourcc: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.fourcc = fourcc
        self.frames = 0
        self._writer = cv2.VideoWriter(
            str(self.path),
            cv2.VideoWriter_fourcc(*fourcc),
            self.fps,
            (self.width, self.height),
        )
        if not self._writer.isOpened():
            raise VideoEncodingError(
                f"OpenCV could not open an MP4 writer for {self.path} "
                f"(fourcc={fourcc!r}, {self.width}x{self.height} @ {self.fps} fps)"
            )

    def write(self, rgb: np.ndarray) -> None:
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        self._writer.write(bgr)
        self.frames += 1

    def close(self) -> dict:
        self._writer.release()
        if not self.path.exists() or self.path.stat().st_size == 0:
            raise VideoEncodingError(f"OpenCV produced no data for {self.path}")
        return {
            "backend": self.backend,
            "codec": self.fourcc,
            "frames_written": self.frames,
            "path": str(self.path),
        }


class FFmpegFrameWriter(FrameWriter):
    backend = "ffmpeg"

    def __init__(self, path: str | Path, width: int, height: int, fps: int, ffmpeg: str) -> None:
        if width % 2 or height % 2:
            raise VideoEncodingError(
                f"FFmpeg yuv420p encoding needs even dimensions, got {width}x{height}"
            )
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.ffmpeg = ffmpeg
        self.frames = 0
        self._log = tempfile.TemporaryFile()
        command = [
            ffmpeg,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{self.width}x{self.height}",
            "-r", f"{self.fps:g}",
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(self.path),
        ]
        try:
            self._proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self._log,
            )
        except OSError as exc:
            raise VideoEncodingError(f"failed to start FFmpeg at {ffmpeg}: {exc}") from exc
        if self._proc.stdin is None:  # pragma: no cover - defensive
            raise VideoEncodingError(f"FFmpeg at {ffmpeg} opened no stdin pipe")

    def write(self, rgb: np.ndarray) -> None:
        if rgb.shape[0] != self.height or rgb.shape[1] != self.width:
            raise VideoEncodingError(
                f"frame size {rgb.shape[1]}x{rgb.shape[0]} does not match "
                f"{self.width}x{self.height}"
            )
        try:
            self._proc.stdin.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
        except BrokenPipeError as exc:
            raise VideoEncodingError(f"FFmpeg pipe broke while writing {self.path}") from exc
        self.frames += 1

    def close(self) -> dict:
        if self._proc.stdin:
            self._proc.stdin.close()
        returncode = self._proc.wait(timeout=120)
        log_tail = b""
        try:
            self._log.seek(0)
            log_tail = self._log.read()[-2000:]
        finally:
            self._log.close()
        if returncode != 0:
            raise VideoEncodingError(
                f"FFmpeg exited with code {returncode} while writing {self.path}: "
                f"{log_tail.decode('utf-8', 'replace')}"
            )
        if not self.path.exists() or self.path.stat().st_size == 0:
            raise VideoEncodingError(f"FFmpeg produced no data for {self.path}")
        return {
            "backend": self.backend,
            "codec": "libx264",
            "frames_written": self.frames,
            "path": str(self.path),
        }


def resolve_backend(requested: str = BACKEND_AUTO) -> dict:
    """Pick an encoding backend and report exactly what will be used."""
    if requested not in BACKENDS:
        raise VideoEncodingError(f"unknown encoder backend {requested!r}; expected one of {BACKENDS}")
    ffmpeg = detect_ffmpeg()
    if requested == BACKEND_FFMPEG:
        if not ffmpeg:
            raise VideoEncodingError(
                "FFmpeg executable not found on PATH (or in the known install "
                "locations). Install FFmpeg or use --encoder opencv."
            )
        selected = BACKEND_FFMPEG
    elif requested == BACKEND_OPENCV:
        selected = BACKEND_OPENCV
    else:
        selected = BACKEND_FFMPEG if ffmpeg else BACKEND_OPENCV
    info = {
        "requested": requested,
        "selected": selected,
        "ffmpeg_executable": ffmpeg,
        "ffmpeg_available": bool(ffmpeg),
    }
    if selected == BACKEND_OPENCV:
        info["fourcc"] = probe_opencv_fourcc()
        info["codec"] = info["fourcc"]
    else:
        info["codec"] = "libx264"
    return info


def open_frame_writer(
    path: str | Path,
    width: int,
    height: int,
    fps: int,
    *,
    backend_info: dict,
) -> FrameWriter:
    if backend_info["selected"] == BACKEND_FFMPEG:
        return FFmpegFrameWriter(path, width, height, fps, backend_info["ffmpeg_executable"])
    return OpenCVFrameWriter(path, width, height, fps, backend_info.get("fourcc", "mp4v"))


def probe_video(path: str | Path) -> dict:
    """Re-open an MP4 and report what is actually inside it."""
    path = Path(path)
    result = {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
        "frames": 0,
        "fps": 0.0,
        "width": 0,
        "height": 0,
        "readable": False,
        "valid": False,
    }
    if not result["exists"] or result["bytes"] == 0:
        return result
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return result
    result["frames"] = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    result["fps"] = float(capture.get(cv2.CAP_PROP_FPS))
    result["width"] = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    result["height"] = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, _frame = capture.read()
    capture.release()
    result["readable"] = bool(ok)
    result["valid"] = bool(ok and result["frames"] > 0)
    return result


def concat_videos(
    paths: list[str | Path],
    out_path: str | Path,
    *,
    width: int,
    height: int,
    fps: int,
    backend_info: dict,
) -> dict:
    """Assemble ordered scene clips into one MP4.

    Uses the FFmpeg concat demuxer (stream copy) when FFmpeg is available and
    falls back to a frame-by-frame OpenCV re-encode otherwise.
    """
    paths = [Path(p) for p in paths]
    out_path = Path(out_path)
    if not paths:
        raise VideoEncodingError("no scene clips to assemble")
    for clip in paths:
        probe = probe_video(clip)
        if not probe["valid"]:
            raise VideoEncodingError(f"scene clip is not a playable MP4: {clip}")

    if backend_info.get("selected") == BACKEND_FFMPEG:
        return _concat_ffmpeg(paths, out_path, backend_info["ffmpeg_executable"])
    return _concat_opencv(paths, out_path, width, height, fps, backend_info)


def _concat_ffmpeg(paths: list[Path], out_path: Path, ffmpeg: str) -> dict:
    list_file = Path(tempfile.gettempdir()) / f"stage5_concat_{os.getpid()}.txt"
    lines = []
    for clip in paths:
        entry = clip.resolve().as_posix().replace("'", "'\\''")
        lines.append(f"file '{entry}'")
    list_file.write_text("\n".join(lines), encoding="utf-8")
    command = [
        ffmpeg,
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(out_path),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise VideoEncodingError(f"FFmpeg concat failed to run: {exc}") from exc
    finally:
        list_file.unlink(missing_ok=True)
    if completed.returncode != 0:
        tail = (completed.stderr or b"")[-2000:].decode("utf-8", "replace")
        raise VideoEncodingError(f"FFmpeg concat exited with {completed.returncode}: {tail}")
    return {"method": "ffmpeg_concat", "inputs": len(paths), "path": str(out_path)}


def _concat_opencv(
    paths: list[Path],
    out_path: Path,
    width: int,
    height: int,
    fps: int,
    backend_info: dict,
) -> dict:
    writer = open_frame_writer(out_path, width, height, fps, backend_info=backend_info)
    written = 0
    try:
        for clip in paths:
            capture = cv2.VideoCapture(str(clip))
            if not capture.isOpened():
                raise VideoEncodingError(f"could not reopen scene clip {clip}")
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                rgb = np.ascontiguousarray(frame[:, :, ::-1])
                writer.write(rgb)
                written += 1
            capture.release()
    except Exception:
        try:
            writer.close()
        except Exception:  # noqa: BLE001 - keep the original failure visible
            pass
        raise
    info = writer.close()
    info.update({"method": "opencv_reencode", "inputs": len(paths), "frames_written": written})
    return info
