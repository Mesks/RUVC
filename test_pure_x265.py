#!/usr/bin/env python3
"""
Export pure-x265 reconstructed frames using RUVC's active x265 configuration.

The active RUVC test_modules/codec_module.py configuration is reproduced as:
    input:       raw RGB24
    encoder:     libx265
    pixel format:yuv420p
    x265 params: crf=<CRF>:no-info=1
    metadata:    RUVC_CRF=<CRF>

The commented preset/tune/keyint settings in RUVC are intentionally NOT enabled,
because they are not part of the currently active RUVC x265 baseline.

Example:
    python3 pure_x265_ruvc_reconstruction_export_v1.py \
      --input-root /path/to/draw_paper_img \
      --output-root /path/to/pure_x265_visualization \
      --auto-discover \
      --frames-per-sequence 200 \
      --crf 28 \
      --expected-width 1280 \
      --expected-height 720
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, List, Sequence, Tuple

from PIL import Image


SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True)
class SequenceJob:
    name: str
    source_dir: Path
    frames: Tuple[Path, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Encode image sequences with RUVC's active pure-libx265 settings "
            "and save all decoded reconstruction frames."
        )
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Root containing one subdirectory per test sequence.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Root used to save reconstructed PNG frames and H.265 MKV files.",
    )
    parser.add_argument(
        "--crf",
        type=float,
        default=28.0,
        help="x265 CRF. RUVC passes this value through x265-params.",
    )
    parser.add_argument(
        "--frames-per-sequence",
        type=int,
        default=200,
        help="Number of numerically sorted source frames encoded per sequence.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Input/output frame rate. RUVC bitrate calculations assume 25 fps.",
    )
    parser.add_argument(
        "--expected-width",
        type=int,
        default=1280,
        help="Required input and reconstructed width; use 0 to disable.",
    )
    parser.add_argument(
        "--expected-height",
        type=int,
        default=720,
        help="Required input and reconstructed height; use 0 to disable.",
    )
    parser.add_argument(
        "--expected-sequences",
        type=int,
        default=0,
        help="Required sequence count; use 0 to disable.",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Explicit sequence directory names under --input-root.",
    )
    parser.add_argument(
        "--auto-discover",
        action="store_true",
        help="Use every immediate subdirectory of --input-root, sorted naturally.",
    )
    parser.add_argument(
        "--ffmpeg",
        default=None,
        help="FFmpeg executable. Defaults to the first ffmpeg in PATH.",
    )
    parser.add_argument(
        "--ffprobe",
        default=None,
        help="FFprobe executable. Defaults to the first ffprobe in PATH.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing sequence/CRF directory before processing.",
    )
    parser.add_argument(
        "--delete-mkv",
        action="store_true",
        help="Delete the encoded MKV after all reconstructed PNGs are saved.",
    )
    parser.add_argument(
        "--keep-temporary-raw",
        action="store_true",
        help="Keep decoded RGB24 raw files for debugging. Normally disabled.",
    )
    return parser.parse_args()


def natural_key(value: str) -> Tuple[object, ...]:
    """Natural sort key; numeric fields are ordered by integer value."""
    parts = re.split(r"(\d+)", value)
    key: List[object] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.casefold())
    return tuple(key)


def numeric_key(path: Path) -> Tuple[object, ...]:
    """
    Prefer the integer tokens in the filename stem, then natural-sort the name.

    Examples:
        000.png < 001.png < 2.png < 010.png < frame_11.png
    """
    numbers = tuple(int(item) for item in re.findall(r"\d+", path.stem))
    if numbers:
        return (0, *numbers, natural_key(path.name))
    return (1, natural_key(path.name))


def resolve_executable(explicit: str | None, default_name: str) -> str:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise FileNotFoundError(f"Executable is not usable: {path}")
        return str(path)

    found = shutil.which(default_name)
    if not found:
        raise FileNotFoundError(f"{default_name} was not found in PATH.")
    return str(Path(found).resolve())


def run_text(command: Sequence[str], *, check: bool = True) -> str:
    result = subprocess.run(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result.stdout + result.stderr


def verify_ffmpeg(ffmpeg: str) -> None:
    encoders = run_text([ffmpeg, "-hide_banner", "-encoders"])
    if not re.search(r"\blibx265\b", encoders):
        raise RuntimeError(f"FFmpeg does not expose libx265: {ffmpeg}")

    buildconf = run_text([ffmpeg, "-hide_banner", "-buildconf"], check=False)
    if "--disable-x86asm" in buildconf:
        raise RuntimeError(
            "The selected FFmpeg was built with --disable-x86asm. This exact "
            "configuration previously corrupted RUVC's RGB/YUV420 conversion. "
            f"Selected FFmpeg: {ffmpeg}"
        )


def collect_frame_files(sequence_dir: Path, count: int) -> Tuple[Path, ...]:
    candidates = sorted(
        (
            item
            for item in sequence_dir.iterdir()
            if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=numeric_key,
    )
    if len(candidates) < count:
        raise ValueError(
            f"{sequence_dir.name}: found {len(candidates)} image files, "
            f"but {count} are required."
        )
    return tuple(candidates[:count])


def build_jobs(args: argparse.Namespace, input_root: Path) -> List[SequenceJob]:
    if args.sequences and args.auto_discover:
        raise ValueError("--sequences and --auto-discover cannot be used together.")

    if args.sequences:
        sequence_names = list(args.sequences)
    else:
        # Auto-discovery is also the default when no explicit list is supplied.
        sequence_names = sorted(
            (
                item.name
                for item in input_root.iterdir()
                if item.is_dir() and not item.name.startswith(".")
            ),
            key=natural_key,
        )

    if not sequence_names:
        raise ValueError(f"No sequence directories found under: {input_root}")

    if args.expected_sequences and len(sequence_names) != args.expected_sequences:
        raise ValueError(
            f"Expected {args.expected_sequences} sequences, found "
            f"{len(sequence_names)}: {sequence_names}"
        )

    jobs: List[SequenceJob] = []
    for name in sequence_names:
        source_dir = (input_root / name).resolve()
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Sequence directory does not exist: {source_dir}")
        frames = collect_frame_files(source_dir, args.frames_per_sequence)
        jobs.append(SequenceJob(name=name, source_dir=source_dir, frames=frames))
    return jobs


def read_rgb_frame(path: Path, expected_size: Tuple[int, int] | None) -> Tuple[bytes, int, int]:
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        if expected_size is not None and image.size != expected_size:
            raise ValueError(
                f"Frame size changed inside a sequence: {path} is "
                f"{width}x{height}, expected {expected_size[0]}x{expected_size[1]}."
            )
        return image.tobytes(), width, height


def check_expected_resolution(
    sequence: str,
    width: int,
    height: int,
    expected_width: int,
    expected_height: int,
) -> None:
    if expected_width and width != expected_width:
        raise ValueError(
            f"{sequence}: input width is {width}, expected {expected_width}."
        )
    if expected_height and height != expected_height:
        raise ValueError(
            f"{sequence}: input height is {height}, expected {expected_height}."
        )
    if width % 2 or height % 2:
        raise ValueError(
            f"{sequence}: yuv420p requires even width and height, got "
            f"{width}x{height}."
        )


def log_file_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "<unable to read log>"


def encode_sequence(
    *,
    ffmpeg: str,
    job: SequenceJob,
    mkv_path: Path,
    crf: float,
    fps: float,
    expected_width: int,
    expected_height: int,
) -> Tuple[int, int]:
    first_bytes, width, height = read_rgb_frame(job.frames[0], None)
    check_expected_resolution(
        job.name,
        width,
        height,
        expected_width,
        expected_height,
    )

    x265_params = f"crf={crf:g}:no-info=1"
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        f"{fps:g}",
        "-i",
        "pipe:0",
        "-frames:v",
        str(len(job.frames)),
        "-an",
        "-c:v",
        "libx265",
        "-s:v",
        f"{width}x{height}",
        "-pix_fmt",
        "yuv420p",
        "-x265-params",
        x265_params,
        "-metadata:s:v:0",
        f"RUVC_CRF={crf:g}",
        str(mkv_path),
    ]

    log_path = mkv_path.with_suffix(".encode.log")
    with log_path.open("wb") as log_handle:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=log_handle,
        )
        assert process.stdin is not None
        try:
            process.stdin.write(first_bytes)
            for frame_path in job.frames[1:]:
                frame_bytes, current_width, current_height = read_rgb_frame(
                    frame_path,
                    (width, height),
                )
                if current_width != width or current_height != height:
                    raise AssertionError("Resolution validation did not trigger.")
                process.stdin.write(frame_bytes)
            process.stdin.close()
            return_code = process.wait()
        except BaseException:
            try:
                process.stdin.close()
            except Exception:
                pass
            process.kill()
            process.wait()
            raise

    if return_code != 0:
        raise RuntimeError(
            f"x265 encoding failed for {job.name} with exit code "
            f"{return_code}.\n{log_file_text(log_path)}"
        )
    if not mkv_path.is_file() or mkv_path.stat().st_size == 0:
        raise RuntimeError(f"x265 produced no usable MKV: {mkv_path}")

    log_path.unlink(missing_ok=True)
    return width, height


def probe_video(ffprobe: str, mkv_path: Path) -> dict:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=codec_name,profile,width,height,pix_fmt,nb_read_frames,r_frame_rate",
        "-show_entries",
        "format=format_name,duration,size",
        "-of",
        "json",
        str(mkv_path),
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed for {mkv_path}:\n{result.stderr}"
        )
    return json.loads(result.stdout)


def validate_probe(
    probe: dict,
    *,
    frame_count: int,
    width: int,
    height: int,
) -> None:
    streams = probe.get("streams", [])
    if not streams:
        raise RuntimeError("ffprobe returned no video stream.")
    stream = streams[0]
    expected = {
        "codec_name": "hevc",
        "width": width,
        "height": height,
        "pix_fmt": "yuv420p",
    }
    for key, value in expected.items():
        if stream.get(key) != value:
            raise RuntimeError(
                f"Encoded stream {key}={stream.get(key)!r}, expected {value!r}."
            )
    decoded_count = stream.get("nb_read_frames")
    if decoded_count not in (None, "N/A") and int(decoded_count) != frame_count:
        raise RuntimeError(
            f"Encoded stream contains {decoded_count} readable frames, "
            f"expected {frame_count}."
        )


def read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: List[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def decode_and_save(
    *,
    ffmpeg: str,
    mkv_path: Path,
    output_dir: Path,
    output_names: Sequence[str],
    width: int,
    height: int,
    keep_temporary_raw: bool,
) -> None:
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(mkv_path),
        "-frames:v",
        str(len(output_names)),
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]

    log_path = mkv_path.with_suffix(".decode.log")
    raw_debug_path = mkv_path.with_suffix(".decoded.rgb")
    raw_debug = raw_debug_path.open("wb") if keep_temporary_raw else None
    frame_bytes = width * height * 3

    with log_path.open("wb") as log_handle:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=log_handle,
        )
        assert process.stdout is not None
        try:
            for index, output_name in enumerate(output_names):
                raw = read_exact(process.stdout, frame_bytes)
                if len(raw) != frame_bytes:
                    raise RuntimeError(
                        f"Decoded frame {index} has {len(raw)} bytes, expected "
                        f"{frame_bytes}. The MKV or FFmpeg decode path is incomplete."
                    )
                if raw_debug is not None:
                    raw_debug.write(raw)
                image = Image.frombytes("RGB", (width, height), raw)
                image.save(output_dir / output_name, format="PNG")

            # Consume any remaining data before waiting, preventing image2pipe
            # Broken pipe messages and detecting unexpected extra frames.
            extra = process.stdout.read()
            return_code = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
        finally:
            if raw_debug is not None:
                raw_debug.close()

    if return_code != 0:
        raise RuntimeError(
            f"x265 decode failed for {mkv_path} with exit code {return_code}.\n"
            f"{log_file_text(log_path)}"
        )
    if extra:
        if len(extra) % frame_bytes == 0:
            extra_frames = len(extra) // frame_bytes
            raise RuntimeError(
                f"Decoder returned {extra_frames} unexpected extra frame(s)."
            )
        raise RuntimeError(
            f"Decoder returned {len(extra)} unexpected trailing bytes."
        )

    log_path.unlink(missing_ok=True)


def process_job(
    *,
    args: argparse.Namespace,
    ffmpeg: str,
    ffprobe: str,
    output_root: Path,
    job: SequenceJob,
) -> None:
    crf_label = f"{args.crf:g}".replace(".", "p")
    crf_root = output_root / job.name / f"CRF_{crf_label}"
    sr_dir = crf_root / "SR"
    bitstream_dir = crf_root / "bitstream"

    if crf_root.exists():
        if not args.overwrite:
            existing = list(crf_root.rglob("*"))
            if existing:
                raise FileExistsError(
                    f"Output already exists for {job.name}: {crf_root}\n"
                    "Use a new --output-root or add --overwrite."
                )
        else:
            shutil.rmtree(crf_root)

    sr_dir.mkdir(parents=True, exist_ok=True)
    bitstream_dir.mkdir(parents=True, exist_ok=True)
    mkv_path = bitstream_dir / f"pure_x265_crf{crf_label}_yuv420p.mkv"

    print(
        f"[{job.name}] selected {len(job.frames)} frames in numeric order: "
        f"{job.frames[0].name} -> {job.frames[-1].name}"
    )

    width, height = encode_sequence(
        ffmpeg=ffmpeg,
        job=job,
        mkv_path=mkv_path,
        crf=args.crf,
        fps=args.fps,
        expected_width=args.expected_width,
        expected_height=args.expected_height,
    )

    probe = probe_video(ffprobe, mkv_path)
    validate_probe(
        probe,
        frame_count=len(job.frames),
        width=width,
        height=height,
    )

    # Preserve source basenames. All reconstructions are PNG, so a non-PNG
    # source extension is replaced by .png while its numeric stem is retained.
    output_names = [f"{path.stem}.png" for path in job.frames]
    if len(set(output_names)) != len(output_names):
        raise RuntimeError(
            f"{job.name}: source names would collide after conversion to PNG."
        )

    decode_and_save(
        ffmpeg=ffmpeg,
        mkv_path=mkv_path,
        output_dir=sr_dir,
        output_names=output_names,
        width=width,
        height=height,
        keep_temporary_raw=args.keep_temporary_raw,
    )

    manifest = {
        "sequence": job.name,
        "source_directory": str(job.source_dir),
        "source_first_frame": job.frames[0].name,
        "source_last_frame": job.frames[-1].name,
        "frames": len(job.frames),
        "width": width,
        "height": height,
        "fps": args.fps,
        "encoder": "libx265",
        "codec": "hevc",
        "pixel_format": "yuv420p",
        "x265_params": f"crf={args.crf:g}:no-info=1",
        "RUVC_CRF_metadata": args.crf,
        "ffmpeg": ffmpeg,
        "ffprobe": ffprobe,
        "bitstream_bytes": mkv_path.stat().st_size,
        "ffprobe_result": probe,
    }
    (crf_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if args.delete_mkv:
        mkv_path.unlink()
        try:
            bitstream_dir.rmdir()
        except OSError:
            pass

    saved_count = len(list(sr_dir.glob("*.png")))
    if saved_count != len(job.frames):
        raise RuntimeError(
            f"{job.name}: saved {saved_count} PNG files, expected "
            f"{len(job.frames)}."
        )

    print(
        f"[{job.name}] saved {saved_count} reconstructed frames to {sr_dir}"
    )


def main() -> int:
    args = parse_args()

    if args.frames_per_sequence <= 0:
        raise ValueError("--frames-per-sequence must be positive.")
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")
    if not (0 <= args.crf <= 51):
        raise ValueError("--crf must be between 0 and 51 for this baseline.")

    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root is not a directory: {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    ffmpeg = resolve_executable(args.ffmpeg, "ffmpeg")
    ffprobe = resolve_executable(args.ffprobe, "ffprobe")
    verify_ffmpeg(ffmpeg)

    print(f"FFmpeg: {ffmpeg}")
    print(f"FFprobe: {ffprobe}")
    print(
        "RUVC x265 config: libx265, yuv420p, "
        f"x265-params=crf={args.crf:g}:no-info=1, fps={args.fps:g}"
    )
    print(f"Input root: {input_root}")
    print(f"Output root: {output_root}")

    jobs = build_jobs(args, input_root)
    for index, job in enumerate(jobs, start=1):
        print(f"\n=== Sequence {index}/{len(jobs)}: {job.name} ===")
        process_job(
            args=args,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            output_root=output_root,
            job=job,
        )

    print(
        f"\nCompleted {len(jobs)} sequence(s), "
        f"{len(jobs) * args.frames_per_sequence} reconstructed frames."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
