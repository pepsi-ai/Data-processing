from __future__ import annotations

import importlib
from pathlib import Path

from mmhe_v1.types import CanonicalImage, CanonicalVideo


def _load_cv2():
    try:
        return importlib.import_module("cv2")
    except ModuleNotFoundError as error:
        raise RuntimeError("opencv-python dependency is required for video processing") from error


def _sample_indices(frame_count: int, *, clip_len: int, frame_sample_rate: int) -> tuple[int, ...]:
    if frame_count <= 0:
        raise ValueError("video must contain at least one frame")
    if clip_len <= 0:
        raise ValueError("clip_len must be > 0")
    if frame_sample_rate <= 0:
        raise ValueError("frame_sample_rate must be > 0")
    return tuple(min(index * frame_sample_rate, frame_count - 1) for index in range(clip_len))


def canonicalize_video(
    path: Path | str,
    *,
    clip_len: int = 16,
    frame_sample_rate: int = 4,
    size: tuple[int, int] = (224, 224),
) -> CanonicalVideo:
    source_path = Path(path)
    if len(size) != 2 or size[0] <= 0 or size[1] <= 0:
        raise ValueError("size must be (width, height) with positive integers")

    cv2 = _load_cv2()
    capture = cv2.VideoCapture(str(source_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"unable to open video: {source_path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        sampled_indices = _sample_indices(
            frame_count,
            clip_len=clip_len,
            frame_sample_rate=frame_sample_rate,
        )
        width, height = size
        frames: list[CanonicalImage] = []
        for frame_index in sampled_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f"unable to read frame {frame_index} from {source_path}")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
            frames.append(
                CanonicalImage(
                    source_path=source_path,
                    width=width,
                    height=height,
                    pixel_bytes=resized.tobytes(),
                )
            )
    finally:
        capture.release()

    duration_seconds = frame_count / fps if fps > 0.0 else 0.0
    return CanonicalVideo(
        source_path=source_path,
        frame_count=frame_count,
        fps=fps,
        duration_seconds=duration_seconds,
        sampled_indices=sampled_indices,
        frames=tuple(frames),
    )


def canonical_video_payload(video: CanonicalVideo) -> bytes:
    header = (
        f"VIDEO|frame_count={video.frame_count}|sampled={','.join(str(i) for i in video.sampled_indices)}"
        f"|frames={len(video.frames)}|"
    ).encode("ascii")
    chunks = [header]
    for index, frame in enumerate(video.frames):
        frame_header = f"frame={index}|RGB:{frame.width}x{frame.height}|length={len(frame.pixel_bytes)}|".encode(
            "ascii"
        )
        chunks.append(frame_header)
        chunks.append(frame.pixel_bytes)
    return b"".join(chunks)
