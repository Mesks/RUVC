#!/usr/bin/env bash
set -Eeuo pipefail

trap 'printf "\nERROR: command failed at line %s: %s\n" "$LINENO" "$BASH_COMMAND" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$SCRIPT_DIR"

FFMPEG_SCRIPT="$SCRIPT_DIR/dependencies/FFmpegSetting.sh"
FFMPEG_ENV_POINTER="$SCRIPT_DIR/dependencies/.ruvc_ffmpeg_env_path"
REQUIREMENTS_FILE="$SCRIPT_DIR/dependencies/requirements.txt"
SKVIDEO_FFMPEG_PATCH="$SCRIPT_DIR/dependencies/skvideo_change/ffmpeg.py"
SKVIDEO_ABSTRACT_PATCH="$SCRIPT_DIR/dependencies/skvideo_change/abstract.py"

PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
PYTORCH_INDEX="https://download.pytorch.org/whl/cu117"

PYTORCH_PACKAGES=("torch==1.13.0+cu117" "torchvision==0.14.0+cu117" "torchaudio==0.13.0")
PROJECT_EXTRA_PACKAGES=("einops==0.8.2" "lpips==0.1.4" "timm==1.0.27")
SPATIAL_CORRELATION_PACKAGE="spatial-correlation-sampler==0.4.0"
LEGACY_SETUPTOOLS_VERSION="69.5.1"
CUDA_117_CHANNEL="nvidia/label/cuda-11.7.0"
CUDA_117_TOOLKIT_PACKAGE="cuda-toolkit=11.7.0"

fail() {
    local message="$1"
    local status="${2:-1}"
    printf '\nERROR: %s\n' "$message" >&2
    exit "$status"
}

info() {
    printf '\n==> %s\n' "$*"
}

run_step() {
    local description="$1"
    local status
    shift

    printf '\n==> %s\n' "$description"

    if "$@"; then
        return 0
    else
        status=$?
        fail "$description failed. No later initialization steps were run." "$status"
    fi
}

require_file() {
    local path="$1"
    [ -f "$path" ] || fail "Required file not found: $path"
}

require_command() {
    local command_name="$1"
    command -v "$command_name" >/dev/null 2>&1 || fail "Required command was not found: $command_name"
}

load_ffmpeg_environment() {
    local environment_file="$1"
    require_file "$environment_file"
    . "$environment_file"
}

persist_ffmpeg_environment() {
    local environment_file="$1"
    local profile_file="$HOME/.bashrc"
    local begin_marker="# >>> RUVC FFmpeg environment >>>"
    local end_marker="# <<< RUVC FFmpeg environment <<<"
    local temporary_file

    touch "$profile_file"
    temporary_file="$(mktemp "${profile_file}.ruvc.XXXXXX")"

    awk -v begin_marker="$begin_marker" -v end_marker="$end_marker" '
        $0 == begin_marker { skip = 1; next }
        $0 == end_marker { skip = 0; next }
        !skip { print }
    ' "$profile_file" > "$temporary_file"

    printf '%s\n' "$begin_marker" >> "$temporary_file"
    printf '[ -f %q ] && . %q\n' "$environment_file" "$environment_file" >> "$temporary_file"
    printf '%s\n' "$end_marker" >> "$temporary_file"

    mv "$temporary_file" "$profile_file"
}

validate_vvc_codecs() {
    ffmpeg -hide_banner -h encoder=libvvenc >/dev/null
    ffmpeg -hide_banner -h decoder=libvvdec >/dev/null
}

install_pytorch_dependencies() {
    python -m pip install --no-cache-dir "${PYTORCH_PACKAGES[@]}" -i "$PIP_MIRROR" --extra-index-url "$PYTORCH_INDEX"

    python - <<'PY'
import torch

if torch.version.cuda is None:
    raise RuntimeError("The installed torch package does not include CUDA support.")

print(f"Torch: {torch.__version__}")
print(f"Torch CUDA: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
PY
}

install_remaining_python_dependencies() {
    local filtered_requirements
    local status

    filtered_requirements="$(mktemp "${TMPDIR:-/tmp}/ruvc-requirements.XXXXXX")"

    awk '
        {
            line = $0
            sub(/[[:space:]]*#.*/, "", line)
            gsub(/^[[:space:]]+/, "", line)
            gsub(/[[:space:]]+$/, "", line)

            if (line ~ /^(torch|torchvision|torchaudio|spatial-correlation-sampler)([[:space:]]*[<>=!~].*)?$/) {
                next
            }

            print $0
        }
    ' "$REQUIREMENTS_FILE" > "$filtered_requirements"

    status=0
    python -m pip install --no-cache-dir -r "$filtered_requirements" -i "$PIP_MIRROR" --extra-index-url "$PYTORCH_INDEX" || status=$?
    rm -f "$filtered_requirements"

    return "$status"
}

install_project_extra_dependencies() {
    python -m pip install --no-cache-dir "${PROJECT_EXTRA_PACKAGES[@]}" -i "$PIP_MIRROR" --extra-index-url "$PYTORCH_INDEX"
}

validate_project_extra_dependencies() {
    python - <<'PY'
from importlib.metadata import version

expected_versions = {
    "einops": "0.8.2",
    "lpips": "0.1.4",
    "timm": "1.0.27",
}

for package_name, expected_version in expected_versions.items():
    actual_version = version(package_name)

    if actual_version != expected_version:
        raise RuntimeError(f"{package_name} version mismatch: expected {expected_version}, found {actual_version}")

    print(f"{package_name}: {actual_version}")
PY
}

install_legacy_setuptools_compatibility() {
    python -m pip install --no-cache-dir --force-reinstall "setuptools==$LEGACY_SETUPTOOLS_VERSION" "wheel"

    python - <<'PY'
from pkg_resources import packaging
import setuptools
import torch

print(f"setuptools: {setuptools.__version__}")
print(f"Torch: {torch.__version__}")
print("Torch extension build environment: OK")
PY
}

get_torch_cuda_major_minor() {
    python - <<'PY'
import torch

version = torch.version.cuda

if not version:
    raise SystemExit("Torch does not report a CUDA build version.")

parts = version.split(".")

if len(parts) < 2:
    raise SystemExit(f"Unexpected Torch CUDA version: {version}")

print(f"{parts[0]}.{parts[1]}")
PY
}

get_nvcc_major_minor() {
    local nvcc_binary="$1"

    "$nvcc_binary" --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -n 1 | awk -F. 'NF >= 2 { print $1 "." $2 }'
}

configure_cuda_environment() {
    local cuda_home="$1"
    local cuda_library_directory=""

    [ -x "$cuda_home/bin/nvcc" ] || fail "CUDA toolkit does not contain nvcc: $cuda_home/bin/nvcc"

    export CUDA_HOME="$cuda_home"
    export CUDA_PATH="$cuda_home"
    export CUDACXX="$cuda_home/bin/nvcc"
    export PATH="$cuda_home/bin:$PATH"

    for cuda_library_directory in "$cuda_home/lib64" "$cuda_home/lib" "$cuda_home/targets/x86_64-linux/lib"; do
        if [ -d "$cuda_library_directory" ]; then
            export LIBRARY_PATH="$cuda_library_directory${LIBRARY_PATH:+:$LIBRARY_PATH}"
            break
        fi
    done

    hash -r
}

find_matching_cuda_home() {
    local expected_version="$1"
    local candidate=""
    local candidate_nvcc=""
    local candidate_version=""
    local -a candidates=()

    if [ -n "${RUVC_CUDA_HOME:-}" ]; then
        candidates+=("$RUVC_CUDA_HOME")
    fi

    if [ -n "${CUDA_HOME:-}" ]; then
        candidates+=("$CUDA_HOME")
    fi

    if [ -n "${CONDA_PREFIX:-}" ]; then
        candidates+=("$CONDA_PREFIX")
    fi

    if command -v nvcc >/dev/null 2>&1; then
        candidate_nvcc="$(command -v nvcc)"
        candidates+=("$(cd "$(dirname "$candidate_nvcc")/.." && pwd -P)")
    fi

    for candidate in "${candidates[@]}"; do
        [ -n "$candidate" ] || continue
        [ -x "$candidate/bin/nvcc" ] || continue

        candidate_version="$(get_nvcc_major_minor "$candidate/bin/nvcc")"

        if [ "$candidate_version" = "$expected_version" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    return 1
}

install_matching_conda_cuda_toolkit() {
    local expected_version="$1"

    [ "$expected_version" = "11.7" ] || fail "Automatic CUDA toolkit installation is only configured for Torch CUDA 11.7. Set RUVC_CUDA_HOME to a matching CUDA toolkit for Torch CUDA $expected_version."
    [ -n "${CONDA_PREFIX:-}" ] || fail "Torch CUDA $expected_version requires a matching CUDA toolkit. Activate a Conda environment or set RUVC_CUDA_HOME to an existing CUDA 11.7 toolkit."

    require_command conda

    info "Installing CUDA 11.7 toolkit into the active Conda environment for the Torch C++ extension build"
    conda install --yes --channel "$CUDA_117_CHANNEL" "$CUDA_117_TOOLKIT_PACKAGE"
}

ensure_matching_cuda_toolkit() {
    local expected_version=""
    local cuda_home=""
    local actual_version=""

    expected_version="$(get_torch_cuda_major_minor)"

    if cuda_home="$(find_matching_cuda_home "$expected_version")"; then
        configure_cuda_environment "$cuda_home"
    else
        install_matching_conda_cuda_toolkit "$expected_version"
        hash -r
        cuda_home="$(find_matching_cuda_home "$expected_version")" || fail "CUDA $expected_version was installed but a matching nvcc executable was not found. Set RUVC_CUDA_HOME to the CUDA toolkit root and run the script again."
        configure_cuda_environment "$cuda_home"
    fi

    actual_version="$(get_nvcc_major_minor "$CUDA_HOME/bin/nvcc")"

    [ "$actual_version" = "$expected_version" ] || fail "CUDA compiler mismatch: Torch requires CUDA $expected_version, but $CUDA_HOME/bin/nvcc reports CUDA $actual_version."

    info "Using CUDA toolkit $actual_version from $CUDA_HOME"
    "$CUDA_HOME/bin/nvcc" --version
}

spatial_correlation_sampler_is_installed() {
    python - <<'PY'
from spatial_correlation_sampler import SpatialCorrelationSampler

print("SpatialCorrelationSampler import: OK")
PY
}

install_spatial_correlation_sampler() {
    local extension_jobs="${RUVC_EXTENSION_BUILD_JOBS:-1}"

    [[ "$extension_jobs" =~ ^[1-9][0-9]*$ ]] || fail "RUVC_EXTENSION_BUILD_JOBS must be a positive integer."

    if spatial_correlation_sampler_is_installed >/dev/null 2>&1; then
        info "Skipping spatial-correlation-sampler because it is already installed"
        return 0
    fi

    ensure_matching_cuda_toolkit
    command -v nvcc >/dev/null 2>&1 || fail "nvcc was not found after CUDA toolkit setup."

    MAX_JOBS="$extension_jobs" python -m pip install --no-build-isolation --no-deps --no-cache-dir -i "$PIP_MIRROR" --extra-index-url "$PYTORCH_INDEX" "$SPATIAL_CORRELATION_PACKAGE"
}

validate_spatial_correlation_sampler() {
    python - <<'PY'
import torch
from spatial_correlation_sampler import SpatialCorrelationSampler

print(f"Torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("SpatialCorrelationSampler import: OK")
PY
}

require_file "$FFMPEG_SCRIPT"
require_file "$REQUIREMENTS_FILE"
require_file "$SKVIDEO_FFMPEG_PATCH"
require_file "$SKVIDEO_ABSTRACT_PATCH"

run_step "Building codec dependencies and FFmpeg" bash "$FFMPEG_SCRIPT"

require_file "$FFMPEG_ENV_POINTER"
FFMPEG_ENV_FILE="$(<"$FFMPEG_ENV_POINTER")"
FFMPEG_ENV_FILE="${FFMPEG_ENV_FILE%$'\r'}"

[ -n "$FFMPEG_ENV_FILE" ] || fail "FFmpeg environment path file is empty: $FFMPEG_ENV_POINTER"

run_step "Loading the project FFmpeg environment" load_ffmpeg_environment "$FFMPEG_ENV_FILE"
run_step "Registering the FFmpeg environment for future Bash terminals" persist_ffmpeg_environment "$FFMPEG_ENV_FILE"
run_step "Checking the active FFmpeg VVC codecs" validate_vvc_codecs

run_step "Installing PyTorch dependencies" install_pytorch_dependencies
run_step "Installing remaining Python dependencies" install_remaining_python_dependencies
run_step "Installing project extra Python dependencies" install_project_extra_dependencies
run_step "Checking project extra Python dependencies" validate_project_extra_dependencies
run_step "Installing legacy setuptools compatibility layer" install_legacy_setuptools_compatibility
run_step "Installing spatial-correlation-sampler" install_spatial_correlation_sampler
run_step "Checking spatial-correlation-sampler" validate_spatial_correlation_sampler
run_step "Checking Python package dependencies" python -m pip check

SKVIDEO_PATH="$(python -c 'import skvideo; print(skvideo.__file__)')"
SKVIDEO_DIR="$(dirname "$SKVIDEO_PATH")/io"
SKVIDEO_FFMPEG_TARGET="$SKVIDEO_DIR/ffmpeg.py"
SKVIDEO_ABSTRACT_TARGET="$SKVIDEO_DIR/abstract.py"

require_file "$SKVIDEO_FFMPEG_TARGET"
require_file "$SKVIDEO_ABSTRACT_TARGET"

run_step "Applying skvideo FFmpeg patch" cp -- "$SKVIDEO_FFMPEG_PATCH" "$SKVIDEO_FFMPEG_TARGET"
run_step "Applying skvideo abstract patch" cp -- "$SKVIDEO_ABSTRACT_PATCH" "$SKVIDEO_ABSTRACT_TARGET"

printf '\nInitialization completed successfully.\n'
printf 'The FFmpeg environment was loaded for this initialization and added to %s for future Bash terminals.\n' "$HOME/.bashrc"
printf 'A script launched with bash cannot modify the already-running parent shell environment. Open a new Bash terminal before running project commands in this terminal session.\n'
