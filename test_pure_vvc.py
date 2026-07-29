import argparse
import csv
import datetime
import json
import os
import subprocess
import time

import cv2
import torch

from auxiliary_modules import compute_metrics


FPS = 25
QP_MAX = 63


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Evaluate a pure VVC baseline without RUVC resampling or reconstruction.',
    )
    root_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--testdata', type=str, required=True,
                        help='Directory containing PNG or JPG frames.')
    parser.add_argument('--qp', type=int, required=True,
                        help='VVenC base QP in [0, 63].')
    parser.add_argument('--prefix', type=str, default='PureVVC',
                        help='Output prefix under log and intermediate.')
    parser.add_argument('--frame_number', type=int, default=-1,
                        help='Number of frames to evaluate. -1 uses all frames.')
    parser.add_argument('--GPU_index', type=int, default=0,
                        help='GPU index for metric computation. Use -1 for CPU.')
    parser.add_argument('--measure_step_by_step', type=int, default=1,
                        help='Use the same frame-averaged metric protocol as test_RUVC.py.')
    parser.add_argument('--print_each_frame', type=int, default=0,
                        help='Print per-frame metric values.')
    parser.add_argument('--keep_bitstream', type=int, default=0,
                        help='Keep the generated MKV file after metric computation.')
    parser.add_argument('--root_dir', type=str, default=root_dir,
                        help='Project root used for output directories.')
    return parser.parse_args()


def run_command(command, input_data=None):
    result = subprocess.run(
        command,
        input=input_data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.decode(errors='replace').strip()
        raise RuntimeError(f"FFmpeg command failed: {' '.join(command)}\n{message}")
    return result.stdout


def read_frames(data_path, frame_number):
    files = sorted(
        os.path.join(data_path, name)
        for name in os.listdir(data_path)
        if name.lower().endswith(('.jpg', '.jpeg', '.png'))
    )
    if not files:
        raise RuntimeError(f'No PNG or JPG frames were found in: {data_path}')
    if frame_number == -1:
        frame_number = len(files)
    if frame_number <= 0 or frame_number > len(files):
        raise ValueError(f'frame_number must be in [1, {len(files)}] or -1.')

    frames = []
    height = width = None
    for path in files[:frame_number]:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f'Failed to read image: {path}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if height is None:
            height, width = image.shape[:2]
            if height % 2 or width % 2:
                raise ValueError(
                    'Pure VVC uses yuv420p10le and requires even frame dimensions. '
                    f'Got {width}x{height}.'
                )
        elif image.shape[:2] != (height, width):
            raise ValueError(f'Frame size mismatch in {data_path}: {path}')
        frames.append(torch.from_numpy(image).permute(2, 0, 1))

    video = torch.stack(frames).to(torch.float32) / 255.0
    return video, width, height


def read_metadata(video_name):
    output = run_command([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream_tags', '-of', 'json', video_name,
    ])
    streams = json.loads(output.decode()).get('streams', [])
    if not streams:
        raise RuntimeError('The VVC Matroska file does not contain a video stream.')
    return {key.upper(): value for key, value in streams[0].get('tags', {}).items()}


def encode_decode(video, width, height, qp, video_name):
    frame_count = video.shape[0]
    input_bytes = (
        (video.clamp(0.0, 1.0) * 255.0)
        .round()
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .contiguous()
        .cpu()
        .numpy()
        .tobytes()
    )

    encode_command = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-f', 'rawvideo', '-pixel_format', 'rgb24',
        '-video_size', f'{width}x{height}', '-framerate', str(FPS), '-i', '-',
        '-c:v', 'libvvenc', '-qp', str(qp), '-pix_fmt', 'yuv420p10le',
        '-frames:v', str(frame_count),
        '-metadata:s:v:0', 'RUVC_CODEC=vvenc',
        '-metadata:s:v:0', 'RUVC_BASELINE=pure_vvc',
        '-metadata:s:v:0', f'RUVC_QP={qp}',
        '-metadata:s:v:0', f'RUVC_QP_MAX={QP_MAX}',
        '-f', 'matroska', video_name,
    ]
    print(f"{'Encoding...':<21}", end='', flush=True)
    encode_start = time.time()
    run_command(encode_command, input_bytes)
    encode_time = time.time() - encode_start
    print(f'Finished. Consumed {encode_time:.6f} seconds', flush=True)

    file_size = os.path.getsize(video_name)
    metadata = read_metadata(video_name)
    expected_metadata = {
        'RUVC_CODEC': 'vvenc',
        'RUVC_BASELINE': 'pure_vvc',
        'RUVC_QP': str(qp),
        'RUVC_QP_MAX': str(QP_MAX),
    }
    if any(metadata.get(key, '') != value for key, value in expected_metadata.items()):
        raise RuntimeError('VVC Matroska metadata is missing or invalid.')

    print(f"{'Decoding...':<21}", end='', flush=True)
    decode_start = time.time()
    decoded_bytes = run_command([
        'ffmpeg', '-loglevel', 'error', '-c:v', 'libvvdec', '-i', video_name,
        '-map', '0:v:0', '-vf', 'format=rgb24', '-fps_mode', 'passthrough',
        '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-',
    ])
    decode_time = time.time() - decode_start
    print(f'Finished. Consumed {decode_time:.6f} seconds', flush=True)

    expected_size = frame_count * height * width * 3
    if len(decoded_bytes) != expected_size:
        probe = run_command([
            'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,pix_fmt,nb_read_frames',
            '-of', 'default=noprint_wrappers=1', video_name,
        ]).decode().strip().replace('\n', ', ')
        raise RuntimeError(
            f'VVC decoded byte count mismatch: expected {expected_size}, got {len(decoded_bytes)}. '
            f'Stream details: {probe}'
        )

    decoded = torch.from_numpy(
        torch.frombuffer(bytearray(decoded_bytes), dtype=torch.uint8)
        .reshape(frame_count, height, width, 3)
        .numpy()
        .copy()
    ).permute(0, 3, 1, 2).to(torch.float32) / 255.0
    return decoded, file_size, encode_time, decode_time


def resolve_device(gpu_index):
    if gpu_index >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(gpu_index)
        return torch.device(f'cuda:{gpu_index}')
    return torch.device('cpu')


def compute_all_metrics(decoded, reference, device, step_by_step, print_each_frame):
    metric_names = ('PSNR', 'SSIM', 'MSSSIM', 'LPIPS')
    values = {name: 0.0 for name in metric_names}
    if step_by_step:
        for index in range(reference.shape[0]):
            frame_values = {
                name: compute_metrics.compute_metric(
                    decoded[index], reference[index], metric=name,
                    step_by_step=True, device=device,
                )
                for name in metric_names
            }
            if print_each_frame:
                print(
                    f"\tFrame index: {index:3d}, PSNR: {frame_values['PSNR']:.6f}, "
                    f"SSIM: {frame_values['SSIM']:.6f}, MSSSIM: {frame_values['MSSSIM']:.6f}, "
                    f"LPIPS: {frame_values['LPIPS']:.6f}."
                )
            for name, value in frame_values.items():
                values[name] += value
        for name in values:
            values[name] /= reference.shape[0]
    else:
        for name in metric_names:
            values[name] = compute_metrics.compute_metric(
                decoded, reference, metric=name, step_by_step=False, device=device,
            )
    return values


def ensure_result_file(result_path):
    if not os.path.exists(result_path):
        with open(result_path, 'w', newline='') as file:
            csv.writer(file).writerow([
                'Sequence', 'QP', 'PSNR', 'MS-SSIM', 'SSIM', 'LPIPS',
                'bpp', 'bitrate', 'runtime',
            ])


def main():
    args = parse_args()
    if not 0 <= args.qp <= QP_MAX:
        raise ValueError(f'QP must be an integer in [0, {QP_MAX}].')
    if not os.path.isdir(args.testdata):
        raise FileNotFoundError(f'Test data directory does not exist: {args.testdata}')

    log_dir = os.path.join(args.root_dir, 'log', args.prefix)
    intermediate_dir = os.path.join(args.root_dir, 'intermediate', args.prefix, 'pure_vvc')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(intermediate_dir, exist_ok=True)
    result_path = os.path.join(log_dir, 'result_pure_vvenc.csv')
    ensure_result_file(result_path)

    sequence = os.path.basename(os.path.normpath(args.testdata))
    video_name = os.path.join(intermediate_dir, f'{sequence}_qp{args.qp}.mkv')
    device = resolve_device(args.GPU_index)

    print(f"\n====================>>>{'Pure VVC Test'.center(25)}<<<====================")
    print(f"{'Start':<8}{'Time':<7}: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f'Codec: VVC (QP: {args.qp}, default inter coding configuration)')
    print('Pipeline: original frames -> VVenC -> VVdeC -> metrics')

    reference, width, height = read_frames(args.testdata, args.frame_number)
    print(f'Resolution: {width}x{height}')
    print(f'Frame number: {reference.shape[0]}')
    print(f'Metric device: {device}')

    runtime_start = time.time()
    decoded, file_size, _, _ = encode_decode(reference, width, height, args.qp, video_name)
    runtime = time.time() - runtime_start

    print(f"{'Metrics computing...':<21}", end='', flush=True)
    metrics = compute_all_metrics(
        decoded, reference, device,
        step_by_step=args.measure_step_by_step > 0,
        print_each_frame=args.print_each_frame > 0,
    )
    print('Finished.', flush=True)

    frame_count = reference.shape[0]
    bpp = file_size * 8.0 / (frame_count * height * width)
    bitrate = file_size * 8.0 / (frame_count / FPS * 1000.0)

    with open(result_path, 'a', newline='') as file:
        csv.writer(file).writerow([
            sequence, args.qp,
            f"{metrics['PSNR']:.6f}", f"{metrics['MSSSIM']:.6f}",
            f"{metrics['SSIM']:.6f}", f"{metrics['LPIPS']:.6f}",
            f'{bpp:.6f}', f'{bitrate:.6f}', f'{runtime:.6f}',
        ])

    print(f"\n====================>>>{'Pure VVC Test Completed'.center(25)}<<<====================")
    print(f"Total Video: (Codec: VVC, QP: {args.qp})")
    print(f"\t{'PSNR':<8}= {metrics['PSNR']:.6f}")
    print(f"\t{'MSSSIM':<8}= {metrics['MSSSIM']:.6f}")
    print(f"\t{'SSIM':<8}= {metrics['SSIM']:.6f}")
    print(f"\t{'LPIPS':<8}= {metrics['LPIPS']:.6f}")
    print(f"\t{'bpp':<8}= {bpp:.6f}")
    print(f"\t{'bitrate':<8}= {bitrate:.6f}")
    print(f"\t{'runtime':<8}= {runtime:.6f}")
    print(f'Result CSV: {result_path}')

    if args.keep_bitstream <= 0:
        os.remove(video_name)


if __name__ == '__main__':
    main()
