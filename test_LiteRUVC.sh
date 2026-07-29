#!/bin/bash

## GPU_index:           Limits the GPU used for encoding.
## frame_number:        Value in ["", "x"], "" means encode all frames, "x" means encode x frames, x should be a concrete integer number.
## prefix:              The systematic prefix used to control the files generated for this test, which is generated under the 'log' and 'intermediate' folders.
## crf_values:          The CRF values tested, recommended: (49 44 39 34 27 23 20 16).
## test_sequence_model: Value in ["all", "single"], "all" means test all sequences in the dataset, "single" means test one sequence.
## test_sequence:       Only used when test_sequence_model is "single", DeepSeaFish with most frames, FreeDiver2 with least frames.
## data_dir:            The directory of test image sequences. The secondary directory of this should be the video name folder where each video frame is stored.
## model_path:          The path of test model.
## log_path:            The path to save log file.
GPU_index="0"
# GPU_memory_limitation="0.5"
frame_number=""
reference_step="32"
useLiteRUVC=1
save_intremediate=0
prefix="LiteRUVC"
crf_values=(33 28 23 18)
test_sequence_model="all"
test_sequence="class4_HeartShape"
data_dir="./dataset/UVC46k_testset"
model_path="./model/LiteRUVC/RUVC_model_50.pt"
log_dir="./log/$prefix"
log_path="$log_dir/test.txt"
if [ -d $log_dir ]; then
  if [ -f $log_path ]; then
    rm $log_path
  fi
else
  mkdir -p "$log_dir"
fi

if [ "$test_sequence_model" == "single" ]; then
  data_path="$data_dir/$test_sequence"
  if [ -d "$data_path" ]; then
    for crf in ${crf_values[@]}; do
      if [ "$frame_number" == "" ]; then
        python -u test_RUVC.py --testdata $data_path --model $model_path --GPU_index $GPU_index --use_surrogate 0 --x265_CRF $crf --prefix $prefix --useLiteRUVC $useLiteRUVC --rescaling_times 2 --show_each_frame $save_intremediate --reference_step $reference_step | tee -a $log_path
      else
        python -u test_RUVC.py --testdata $data_path --model $model_path --GPU_index $GPU_index --use_surrogate 0 --x265_CRF $crf --prefix $prefix --useLiteRUVC $useLiteRUVC --frame_number $frame_number --rescaling_times 2 --show_each_frame $save_intremediate --reference_step $reference_step | tee -a $log_path
      fi
    done
  else
    echo "'$data_path' is not a video directory."
  fi
elif [ "$test_sequence_model" == "all" ]; then
  for sequence in $(ls $data_dir); do
    data_path="$data_dir/$sequence"
    if [ -d "$data_path" ]; then
      for crf in ${crf_values[@]}; do
        if [ "$frame_number" == "" ]; then
          python -u test_RUVC.py --testdata $data_path --model $model_path --GPU_index $GPU_index --use_surrogate 0 --x265_CRF $crf --prefix $prefix --useLiteRUVC $useLiteRUVC --rescaling_times 2 --show_each_frame $save_intremediate --reference_step $reference_step | tee -a $log_path
        else
          python -u test_RUVC.py --testdata $data_path --model $model_path --GPU_index $GPU_index --use_surrogate 0 --x265_CRF $crf --prefix $prefix --useLiteRUVC $useLiteRUVC --frame_number $frame_number --rescaling_times 2 --show_each_frame $save_intremediate --reference_step $reference_step | tee -a $log_path
        fi
      done
    else
      echo "'$data_path' is not a video directory, have been skipped."
    fi
  done
else
  echo "ERROR::Test_sequence_model must be in ['all', 'single']."
  echo "If you want to test one sequence, please set test_sequence_model to and set test_sequence to the sequence name."
  echo "If you want to test multiple sequences, please set test_sequence_model to 'all', and directory structure is 'data_dir/[each your video name directories]/[each frame files]', jpg file is recommended."
fi
