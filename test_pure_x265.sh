#!/usr/bin/env bash
python test_pure_x265.py \
  --input-root ./dataset/plot \
  --output-root ./output/pure_x265 \
  --auto-discover \
  --frames-per-sequence 200 \
  --crf 28 \
  --fps 25 \
  --expected-sequences 1 \
  --expected-width 1280 \
  --expected-height 720 \
  --ffmpeg "$PWD/dependencies/ffmpeg/bin/ffmpeg" \
  --ffprobe "$PWD/dependencies/ffmpeg/bin/ffprobe"