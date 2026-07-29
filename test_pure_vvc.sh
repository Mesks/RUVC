#!/usr/bin/env bash

# qp_values: VVenC base QP values.
# frame_number: Empty uses all frames; an integer limits each sequence.
# test_sequence_mode: "all" evaluates every subdirectory; "single" evaluates test_sequence only.

GPU_index="1"
frame_number=""
prefix="PureVVC"
qp_values=(37 32 27 22)
# qp_values=(42 17)
test_sequence_mode="all"
test_sequence="class1_ClickerAndTarget"
data_dir="./dataset/UVC46k_testset"
log_dir="./log/$prefix"
log_path="$log_dir/test.txt"

mkdir -p "$log_dir"
: > "$log_path"

run_test() {
    local data_path="$1"
    local qp="$2"
    local args=(
        --testdata "$data_path"
        --qp "$qp"
        --GPU_index "$GPU_index"
        --prefix "$prefix"
        --measure_step_by_step 1
        --print_each_frame 0
        --keep_bitstream 0
    )

    if [ -n "$frame_number" ]; then
        args+=(--frame_number "$frame_number")
    fi

    echo "Running Pure VVC: sequence=$(basename "$data_path"), qp=$qp" | tee -a "$log_path"

    set -o pipefail
    PYTHONUNBUFFERED=1 stdbuf -oL -eL \
    python -u test_pure_vvc.py "${args[@]}" < /dev/null 2>&1 | tee -a "$log_path"

    local status=${PIPESTATUS[0]}
    if [ "$status" -ne 0 ]; then
        echo "Run failed: sequence=$(basename "$data_path"), qp=$qp" | tee -a "$log_path"
        exit "$status"
    fi
}

if [ "$test_sequence_mode" = "single" ]; then
    data_path="$data_dir/$test_sequence"
    if [ ! -d "$data_path" ]; then
        echo "'$data_path' is not a video directory."
        exit 1
    fi
    for qp in "${qp_values[@]}"; do
        run_test "$data_path" "$qp"
    done
elif [ "$test_sequence_mode" = "all" ]; then
    mapfile -d '' sequence_paths < <(
        find "$data_dir" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z
    )

    for data_path in "${sequence_paths[@]}"; do
        for qp in "${qp_values[@]}"; do
            run_test "$data_path" "$qp"
        done
    done
else
    echo "test_sequence_mode must be 'all' or 'single'."
    exit 1
fi