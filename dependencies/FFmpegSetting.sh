#!/usr/bin/env bash
set -Eeuo pipefail

trap 'printf "\nERROR: command failed at line %s: %s\n" "$LINENO" "$BASH_COMMAND" >&2' ERR

VVENC_VERSION="1.14.0"
VVDEC_VERSION="3.1.0"
FFMPEG_VERSION="8.1.2"
VVDEC_PATCH_NAME="libvvdec-ffmpeg-8.1.2.patch"
VVDEC_PATCH_URL="https://raw.githubusercontent.com/wiki/fraunhoferhhi/vvdec/data/patch/v8-0001-avcodec-add-external-dec-libvvdec-for-H266-VVC.patch"

X264_ARCHIVE_NAME="x264-stable.tar.gz"
X264_ARCHIVE_URL="https://code.videolan.org/videolan/x264/-/archive/stable/x264-stable.tar.gz"
X265_ARCHIVE_NAME="x265-master.tar.gz"
X265_ARCHIVE_URL="https://github.com/multicorewareinc/x265/archive/refs/heads/master.tar.gz"
VVENC_ARCHIVE_NAME="vvenc-${VVENC_VERSION}.tar.gz"
VVENC_ARCHIVE_URL="https://github.com/fraunhoferhhi/vvenc/archive/refs/tags/v${VVENC_VERSION}.tar.gz"
VVDEC_ARCHIVE_NAME="vvdec-${VVDEC_VERSION}.tar.gz"
VVDEC_ARCHIVE_URL="https://github.com/fraunhoferhhi/vvdec/archive/refs/tags/v${VVDEC_VERSION}.tar.gz"
FFMPEG_ARCHIVE_NAME="ffmpeg-${FFMPEG_VERSION}.tar.xz"
FFMPEG_ARCHIVE_URL="https://ffmpeg.org/releases/${FFMPEG_ARCHIVE_NAME}"

SYSTEM_PACKAGE_MANAGER=""
declare -a SYSTEM_PACKAGES=()
declare -a MISSING_REQUIREMENTS=()
declare -a SUDO=()

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

info() {
    printf '\n==> %s\n' "$*" >&2
}

version_ge() {
    [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n 1)" = "$2" ]
}

is_true() {
    case "${1,,}" in
        true|y|yes|1) return 0 ;;
        *) return 1 ;;
    esac
}

run_make() {
    env MAKEFLAGS= make -j"$JOBS" "$@"
}

archive_is_valid() {
    local archive="$1"
    local format_name="${2:-$1}"

    case "$format_name" in
        *.tar.gz|*.tgz) tar -tzf "$archive" >/dev/null ;;
        *.tar.xz) tar -tJf "$archive" >/dev/null ;;
        *.tar.bz2) tar -tjf "$archive" >/dev/null ;;
        *) fail "Unsupported source archive format: $format_name" ;;
    esac
}

file_is_valid() {
    local file="$1"
    [ -s "$file" ]
}

download_to_cache() {
    local url="$1"
    local destination="$2"
    local validation_kind="$3"
    local format_name="${4:-$destination}"
    local partial_file="${destination}.part"
    local attempt=1
    local -a curl_args=()

    if [ -f "$destination" ]; then
        if [ "$validation_kind" = "archive" ]; then
            if archive_is_valid "$destination" "$format_name"; then
                printf '%s\n' "$destination"
                return 0
            fi
        elif file_is_valid "$destination"; then
            printf '%s\n' "$destination"
            return 0
        fi
        printf 'Cached file is invalid and will be replaced: %s\n' "$destination" >&2
        rm -f "$destination"
    fi

    while [ "$attempt" -le "$NETWORK_RETRIES" ]; do
        curl_args=(--fail --location --retry 0 --connect-timeout 15 --max-time 1800 --output "$partial_file")
        if [ -s "$partial_file" ]; then
            info "Resuming $(basename "$destination") (attempt $attempt/$NETWORK_RETRIES)"
            curl_args+=(--continue-at -)
        else
            info "Downloading $(basename "$destination") (attempt $attempt/$NETWORK_RETRIES)"
        fi

        if curl "${curl_args[@]}" "$url"; then
            if [ "$validation_kind" = "archive" ]; then
                if ! archive_is_valid "$partial_file" "$format_name"; then
                    rm -f "$partial_file"
                    fail "Downloaded archive is invalid: $url"
                fi
            elif ! file_is_valid "$partial_file"; then
                rm -f "$partial_file"
                fail "Downloaded file is empty: $url"
            fi

            mv "$partial_file" "$destination"
            printf '%s\n' "$destination"
            return 0
        fi

        if [ "$attempt" -lt "$NETWORK_RETRIES" ]; then
            printf 'Download failed. Retrying after %s second(s).\n' "$NETWORK_RETRY_DELAY" >&2
            sleep "$NETWORK_RETRY_DELAY"
        fi
        attempt=$((attempt + 1))
    done

    fail "Unable to download $url. The partial file was kept at $partial_file for a future resume."
}

find_cached_file() {
    local filename
    local directory

    for filename in "$@"; do
        for directory in "$package_dir" "$source_dir" "$ffmpeg_root"; do
            if [ -f "$directory/$filename" ]; then
                printf '%s\n' "$directory/$filename"
                return 0
            fi
        done
    done
    return 1
}

adopt_archive_into_cache() {
    local archive="$1"
    local cached_name
    local cached_path

    if [ "$(dirname "$archive")" = "$package_dir" ]; then
        printf '%s\n' "$archive"
        return 0
    fi

    cached_name="$(basename "$archive")"
    cached_path="$package_dir/$cached_name"

    if [ -f "$cached_path" ] && archive_is_valid "$cached_path" "$cached_name"; then
        printf '%s\n' "$cached_path"
        return 0
    fi

    info "Caching local archive as $cached_name"
    cp -- "$archive" "${cached_path}.part"
    archive_is_valid "${cached_path}.part" "$cached_name" || fail "Local archive is invalid: $archive"
    mv "${cached_path}.part" "$cached_path"
    printf '%s\n' "$cached_path"
}

ensure_cached_archive() {
    local canonical_name="$1"
    local url="$2"
    shift 2
    local existing_archive
    local canonical_path="$package_dir/$canonical_name"

    if [ -f "$canonical_path" ] && archive_is_valid "$canonical_path" "$canonical_name"; then
        printf '%s\n' "$canonical_path"
        return 0
    fi

    existing_archive="$(find_cached_file "$canonical_name" "$@" || true)"
    if [ -n "$existing_archive" ]; then
        archive_is_valid "$existing_archive" "$existing_archive" || fail "Local archive is invalid: $existing_archive"
        adopt_archive_into_cache "$existing_archive"
        return 0
    fi

    if is_true "$RUVC_OFFLINE"; then
        fail "Offline mode is enabled and the required archive is missing: $package_dir/$canonical_name"
    fi

    download_to_cache "$url" "$canonical_path" archive "$canonical_name"
}

ensure_cached_file() {
    local canonical_name="$1"
    local url="$2"
    shift 2
    local existing_file
    local canonical_path="$package_dir/$canonical_name"

    if file_is_valid "$canonical_path"; then
        printf '%s\n' "$canonical_path"
        return 0
    fi

    existing_file="$(find_cached_file "$canonical_name" "$@" || true)"
    if [ -n "$existing_file" ]; then
        info "Caching local file as $canonical_name"
        cp -- "$existing_file" "${canonical_path}.part"
        file_is_valid "${canonical_path}.part" || fail "Local file is empty: $existing_file"
        mv "${canonical_path}.part" "$canonical_path"
        printf '%s\n' "$canonical_path"
        return 0
    fi

    if is_true "$RUVC_OFFLINE"; then
        fail "Offline mode is enabled and the required file is missing: $package_dir/$canonical_name"
    fi

    download_to_cache "$url" "$canonical_path" file
}

extract_archive_to_source_dir() {
    local archive="$1"
    local destination="$2"
    local temporary_directory="${destination}.extract.$$"
    local -a extracted_roots=()

    archive_is_valid "$archive" "$archive"
    rm -rf "$temporary_directory"
    mkdir -p "$temporary_directory"

    case "$archive" in
        *.tar.gz|*.tgz) tar -xzf "$archive" -C "$temporary_directory" ;;
        *.tar.xz) tar -xJf "$archive" -C "$temporary_directory" ;;
        *.tar.bz2) tar -xjf "$archive" -C "$temporary_directory" ;;
        *) rm -rf "$temporary_directory"; fail "Unsupported source archive: $archive" ;;
    esac

    mapfile -t extracted_roots < <(find "$temporary_directory" -mindepth 1 -maxdepth 1 -type d -printf '%p\n')
    if [ "${#extracted_roots[@]}" -ne 1 ]; then
        rm -rf "$temporary_directory"
        fail "Archive $archive must contain exactly one top-level source directory."
    fi

    rm -rf "$destination"
    mv "${extracted_roots[0]}" "$destination"
    rm -rf "$temporary_directory"
}

source_tree_is_usable() {
    local directory="$1"
    local required_path="$2"
    [ -e "$directory/$required_path" ]
}

prepare_source_from_cache() {
    local component="$1"
    local destination="$2"
    local required_path="$3"
    local canonical_archive_name="$4"
    local archive_url="$5"
    shift 5
    local archive

    if source_tree_is_usable "$destination" "$required_path"; then
        info "Reusing existing $component source tree"
        return 0
    fi

    archive="$(ensure_cached_archive "$canonical_archive_name" "$archive_url" "$@")"
    info "Extracting cached source archive for $component: $archive"
    extract_archive_to_source_dir "$archive" "$destination"
    source_tree_is_usable "$destination" "$required_path" || fail "Prepared source tree for $component is invalid: $destination"
}

ffmpeg_has_verified_vvdec_integration() {
    local source_directory="$1"

    [ -f "$source_directory/configure" ] || return 1
    [ -f "$source_directory/libavcodec/libvvdec.c" ] || return 1
    grep -Fq -- '--enable-libvvdec' "$source_directory/configure" || return 1
    grep -Fq 'libvvdec' "$source_directory/libavcodec/Makefile" || return 1
    grep -Fq 'libvvdec' "$source_directory/libavcodec/allcodecs.c" || return 1
}

vvdec_patch_is_reverse_applicable() {
    local source_directory="$1"
    local patch_file="$2"

    (
        cd "$source_directory"
        patch --batch --reverse --dry-run -p1 < "$patch_file" >/dev/null 2>&1
    )
}

validate_vvdec_patch_file() {
    local patch_file="$1"

    file_is_valid "$patch_file" || fail "VVdeC patch file is empty: $patch_file"
    grep -Fq -- '--enable-libvvdec' "$patch_file" || fail "VVdeC patch does not add the --enable-libvvdec configure option: $patch_file"
    grep -Fq 'libavcodec/libvvdec.c' "$patch_file" || fail "VVdeC patch does not contain the libvvdec decoder source addition: $patch_file"
}

apply_vvdec_patch() {
    local source_directory="$1"
    local patch_file="$2"

    validate_vvdec_patch_file "$patch_file"

    if ffmpeg_has_verified_vvdec_integration "$source_directory"; then
        info "VVdeC patch is already applied and verified"
        return 0
    fi

    if (
        cd "$source_directory"
        patch --batch --dry-run -p1 < "$patch_file" >/dev/null 2>&1
    ); then
        (
            cd "$source_directory"
            patch --batch -p1 < "$patch_file"
        )
        ffmpeg_has_verified_vvdec_integration "$source_directory" || fail "VVdeC patch was applied but FFmpeg configure does not expose --enable-libvvdec."
        return 0
    fi

    fail "VVdeC patch cannot be applied to $source_directory. Recreate the FFmpeg source tree from the cached FFmpeg $FFMPEG_VERSION archive and retry."
}

pkg_version_is() {
    local package_name="$1"
    local expected_version="$2"
    local actual_version

    actual_version="$(pkg-config --modversion "$package_name" 2>/dev/null || true)"
    [ "$actual_version" = "$expected_version" ]
}

has_x264() {
    compgen -G "$build_dir/lib/libx264.*" >/dev/null
}

has_x265() {
    compgen -G "$build_dir/lib/libx265.*" >/dev/null
}

has_vvenc() {
    pkg_version_is libvvenc "$VVENC_VERSION"
}

has_vvdec() {
    pkg_version_is libvvdec "$VVDEC_VERSION"
}

has_complete_ffmpeg() {
    [ -x "$bin_dir/ffmpeg" ] || return 1
    "$bin_dir/ffmpeg" -hide_banner -h encoder=libvvenc >/dev/null 2>&1 || return 1
    "$bin_dir/ffmpeg" -hide_banner -h decoder=libvvdec >/dev/null 2>&1 || return 1
}

print_component_state() {
    printf '  %-24s %s\n' "$1" "$2" >&2
}

patch_x264_ffmpeg_const_compatibility() {
    local source_file="$1/input/lavf.c"
    [ -f "$source_file" ] || return 0
    sed -i -E 's/\bAVInputFormat \*format;/const AVInputFormat *format;/' "$source_file"
    sed -i -E 's/\bAVCodec \*codec;/const AVCodec *codec;/' "$source_file"
}

detect_system_package_manager() {
    if command -v apt-get >/dev/null 2>&1; then
        SYSTEM_PACKAGE_MANAGER="apt-get"
        SYSTEM_PACKAGES=(build-essential cmake git pkg-config nasm patch curl ca-certificates xz-utils bzip2 zlib1g-dev)
    elif command -v dnf >/dev/null 2>&1; then
        SYSTEM_PACKAGE_MANAGER="dnf"
        SYSTEM_PACKAGES=(gcc gcc-c++ make cmake git pkgconf-pkg-config nasm patch curl ca-certificates xz bzip2 zlib-devel)
    elif command -v yum >/dev/null 2>&1; then
        SYSTEM_PACKAGE_MANAGER="yum"
        SYSTEM_PACKAGES=(gcc gcc-c++ make cmake git pkgconfig nasm patch curl ca-certificates xz bzip2 zlib-devel)
    elif command -v zypper >/dev/null 2>&1; then
        SYSTEM_PACKAGE_MANAGER="zypper"
        SYSTEM_PACKAGES=(gcc gcc-c++ make cmake git pkg-config nasm patch curl ca-certificates xz bzip2 zlib-devel)
    elif command -v pacman >/dev/null 2>&1; then
        SYSTEM_PACKAGE_MANAGER="pacman"
        SYSTEM_PACKAGES=(base-devel cmake git pkgconf nasm patch curl ca-certificates xz bzip2 zlib)
    elif command -v apk >/dev/null 2>&1; then
        SYSTEM_PACKAGE_MANAGER="apk"
        SYSTEM_PACKAGES=(build-base cmake git pkgconf nasm patch curl ca-certificates xz bzip2 zlib-dev)
    fi
}

install_system_dependencies() {
    [ -n "$SYSTEM_PACKAGE_MANAGER" ] || fail "No supported package manager was found. Install the required build dependencies manually."
    case "$SYSTEM_PACKAGE_MANAGER" in
        apt-get)
            "${SUDO[@]}" apt-get update
            "${SUDO[@]}" apt-get install -y "${SYSTEM_PACKAGES[@]}"
            ;;
        dnf)
            "${SUDO[@]}" dnf install -y "${SYSTEM_PACKAGES[@]}"
            ;;
        yum)
            "${SUDO[@]}" yum install -y "${SYSTEM_PACKAGES[@]}"
            ;;
        zypper)
            "${SUDO[@]}" zypper --non-interactive install --no-recommends "${SYSTEM_PACKAGES[@]}"
            ;;
        pacman)
            "${SUDO[@]}" pacman --needed --noconfirm -S "${SYSTEM_PACKAGES[@]}"
            ;;
        apk)
            "${SUDO[@]}" apk add --no-cache "${SYSTEM_PACKAGES[@]}"
            ;;
        *) fail "Unsupported package manager: $SYSTEM_PACKAGE_MANAGER" ;;
    esac
}

collect_missing_requirements() {
    local tool
    local cmake_version

    MISSING_REQUIREMENTS=()
    for tool in bash gcc g++ make cmake git pkg-config nasm patch curl tar xz bzip2; do
        command -v "$tool" >/dev/null 2>&1 || MISSING_REQUIREMENTS+=("command: $tool")
    done

    if command -v cmake >/dev/null 2>&1; then
        cmake_version="$(cmake --version | awk 'NR == 1 { print $3 }')"
        version_ge "$cmake_version" "3.13.0" || MISSING_REQUIREMENTS+=("CMake >= 3.13.0 (found $cmake_version)")
    fi

    if command -v pkg-config >/dev/null 2>&1 && ! pkg-config --exists zlib; then
        MISSING_REQUIREMENTS+=("zlib development files")
    fi
}

show_missing_requirements() {
    local requirement

    printf '\nThe following system build requirements are missing or insufficient:\n' >&2
    for requirement in "${MISSING_REQUIREMENTS[@]}"; do
        printf '  - %s\n' "$requirement" >&2
    done

    if [ -n "$SYSTEM_PACKAGE_MANAGER" ]; then
        printf '\nAsk an administrator to install the following packages with %s:\n  %s\n' "$SYSTEM_PACKAGE_MANAGER" "${SYSTEM_PACKAGES[*]}" >&2
    else
        printf '\nNo supported package manager was detected. Ask an administrator to install a C/C++ toolchain, CMake, Git, pkg-config, NASM, Patch, Curl, XZ, Bzip2, and zlib development files.\n' >&2
    fi
}

prepare_system_dependencies() {
    collect_missing_requirements
    if [ "${#MISSING_REQUIREMENTS[@]}" -eq 0 ]; then
        info "Required system build dependencies are already available; no administrator access is needed"
        return
    fi

    if [ -z "$SYSTEM_PACKAGE_MANAGER" ]; then
        show_missing_requirements
        fail "Cannot install missing system build dependencies because no supported package manager was found."
    fi

    if [ "${EUID}" -eq 0 ]; then
        printf '\nSystem build dependencies are missing. They will be installed through the system package manager.\n'
        printf 'VVenC, VVdeC, and FFmpeg will be installed only under:\n%s\n' "$INSTALL_ROOT"
        info "Installing system build dependencies"
        install_system_dependencies
    elif command -v sudo >/dev/null 2>&1; then
        printf '\nSystem build dependencies are missing. Administrator access is needed only to install these OS-level packages through %s.\n' "$SYSTEM_PACKAGE_MANAGER"
        printf 'VVenC, VVdeC, and FFmpeg will be built and installed only under:\n%s\n' "$INSTALL_ROOT"
        printf 'If your account is authorized for sudo, enter your password at the next prompt.\n'
        if sudo -v; then
            SUDO=(sudo)
            info "Installing system build dependencies"
            install_system_dependencies
        else
            printf '\nWARNING: sudo authentication or authorization is unavailable. The script will not install system packages.\n' >&2
        fi
    else
        printf '\nWARNING: sudo is unavailable. The script will not install system packages.\n' >&2
    fi

    collect_missing_requirements
    if [ "${#MISSING_REQUIREMENTS[@]}" -ne 0 ]; then
        show_missing_requirements
        fail "Cannot continue until the required system build dependencies are available."
    fi
}

resolve_cuda() {
    local candidate

    command -v nvcc >/dev/null 2>&1 || fail "nvcc was not found. Install the CUDA toolkit or disable CUDA/NVENC support."
    command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi was not found. Install a working NVIDIA driver or disable CUDA/NVENC support."

    CUDA_HOME="${CUDA_HOME:-${CUDA_PATH:-}}"
    if [ -z "$CUDA_HOME" ]; then
        CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd -P)"
    fi

    [ -f "$CUDA_HOME/include/cuda.h" ] || fail "CUDA headers were not found under $CUDA_HOME/include. Set CUDA_HOME to a valid CUDA toolkit path."

    CUDA_LIB_DIR=""
    for candidate in "$CUDA_HOME/lib64" "$CUDA_HOME/targets/x86_64-linux/lib" "$CUDA_HOME/targets/aarch64-linux/lib"; do
        if [ -f "$candidate/libnppc.so" ]; then
            CUDA_LIB_DIR="$candidate"
            break
        fi
    done
    [ -n "$CUDA_LIB_DIR" ] || fail "CUDA NPP libraries were not found. Install the CUDA toolkit with NPP support or disable CUDA/NVENC support."

    NVIDIA_DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | tr -d '[:space:]')"
    [ -n "$NVIDIA_DRIVER_VERSION" ] || fail "Unable to determine the NVIDIA driver version."

    if version_ge "$NVIDIA_DRIVER_VERSION" "570.0"; then
        NV_CODEC_HEADERS_TAG="n13.0.19.0"
    elif version_ge "$NVIDIA_DRIVER_VERSION" "550.54.14"; then
        NV_CODEC_HEADERS_TAG="n12.2.72.0"
    elif version_ge "$NVIDIA_DRIVER_VERSION" "530.41.03"; then
        NV_CODEC_HEADERS_TAG="n12.1.14.0"
    elif version_ge "$NVIDIA_DRIVER_VERSION" "470.57.02"; then
        NV_CODEC_HEADERS_TAG="n11.1.5.3"
    else
        fail "NVIDIA driver $NVIDIA_DRIVER_VERSION is too old for the supported NVENC header versions. Upgrade the driver or disable CUDA/NVENC support."
    fi
}

write_environment_file() {
    local environment_file="$1"
    cat > "$environment_file" <<ENVFILE
export PATH="$bin_dir:\${PATH}"
export LD_LIBRARY_PATH="$build_dir/lib:\${LD_LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="$build_dir/lib/pkgconfig:\${PKG_CONFIG_PATH:-}"
ENVFILE
    chmod 0644 "$environment_file"
}

if [ "$(uname -s)" != "Linux" ]; then
    fail "This script supports Linux only."
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_ROOT="$SCRIPT_DIR"
INSTALL_ROOT="${RUVC_FFMPEG_ROOT:-}"

if [ -z "$INSTALL_ROOT" ] && [ -t 0 ]; then
    read -r -p "FFmpeg installation root (default: $DEFAULT_ROOT): " INSTALL_ROOT
fi
INSTALL_ROOT="${INSTALL_ROOT:-$DEFAULT_ROOT}"
INSTALL_ROOT="${INSTALL_ROOT/#\~/$HOME}"
mkdir -p "$INSTALL_ROOT"
INSTALL_ROOT="$(cd "$INSTALL_ROOT" && pwd -P)"

USE_CUDA="${RUVC_USE_CUDA:-}"
if [ -z "$USE_CUDA" ] && [ -t 0 ]; then
    printf 'Optional CUDA/NVENC support accelerates FFmpeg CUDA filters and NVIDIA hardware codecs only. libx264, libx265, libvvenc, and libvvdec remain CPU codecs.\n'
    read -r -p "Enable optional CUDA/NVENC support? [y/N] (default: N, CPU-only FFmpeg): " CUDA_REPLY
    case "${CUDA_REPLY:-N}" in
        y|Y|yes|YES|Yes|1) USE_CUDA="true" ;;
        n|N|no|NO|No|0) USE_CUDA="false" ;;
        *) fail "Invalid CUDA/NVENC selection. Enter y or N." ;;
    esac
fi
case "${USE_CUDA,,}" in
    true|y|yes|1) USE_CUDA="true" ;;
    false|n|no|0|"") USE_CUDA="false" ;;
    *) fail "RUVC_USE_CUDA must be true/false, yes/no, or 1/0." ;;
esac

RUVC_OFFLINE="${RUVC_OFFLINE:-false}"
case "${RUVC_OFFLINE,,}" in
    true|y|yes|1) RUVC_OFFLINE="true" ;;
    false|n|no|0|"") RUVC_OFFLINE="false" ;;
    *) fail "RUVC_OFFLINE must be true/false, yes/no, or 1/0." ;;
esac

NETWORK_RETRIES="${RUVC_NETWORK_RETRIES:-3}"
NETWORK_RETRY_DELAY="${RUVC_NETWORK_RETRY_DELAY:-10}"
JOBS="${RUVC_BUILD_JOBS:-1}"
[[ "$NETWORK_RETRIES" =~ ^[1-9][0-9]*$ ]] || fail "RUVC_NETWORK_RETRIES must be a positive integer."
[[ "$NETWORK_RETRY_DELAY" =~ ^[0-9]+$ ]] || fail "RUVC_NETWORK_RETRY_DELAY must be a non-negative integer."
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || fail "RUVC_BUILD_JOBS must be a positive integer."

detect_system_package_manager
prepare_system_dependencies

ffmpeg_root="$INSTALL_ROOT/ffmpeg"
source_dir="$ffmpeg_root/ffmpeg_sources"
build_dir="$ffmpeg_root/ffmpeg_build"
bin_dir="$ffmpeg_root/bin"
package_dir="$ffmpeg_root/packages"
environment_file="$ffmpeg_root/ffmpeg_env.sh"
environment_pointer_file="$SCRIPT_DIR/.ruvc_ffmpeg_env_path"

mkdir -p "$source_dir" "$build_dir" "$bin_dir" "$package_dir"
export PKG_CONFIG_PATH="$build_dir/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export LD_LIBRARY_PATH="$build_dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

info "Build parallelism: $JOBS job(s)"
info "Download cache directory: $package_dir"
if is_true "$RUVC_OFFLINE"; then info "Offline mode: enabled"; else info "Offline mode: disabled"; fi

info "Detected installation state"
if has_x264; then print_component_state "x264" "complete"; else print_component_state "x264" "incomplete"; fi
if has_x265; then print_component_state "x265" "complete"; else print_component_state "x265" "incomplete"; fi
if has_vvenc; then print_component_state "VVenC $VVENC_VERSION" "complete"; else print_component_state "VVenC $VVENC_VERSION" "incomplete"; fi
if has_vvdec; then print_component_state "VVdeC $VVDEC_VERSION" "complete"; else print_component_state "VVdeC $VVDEC_VERSION" "incomplete"; fi
if has_complete_ffmpeg; then print_component_state "FFmpeg $FFMPEG_VERSION" "complete"; else print_component_state "FFmpeg $FFMPEG_VERSION" "incomplete"; fi

x264_dir="$source_dir/x264"
if has_x264; then
    info "Skipping x264 because it is already installed"
else
    prepare_source_from_cache "x264" "$x264_dir" "configure" "$X264_ARCHIVE_NAME" "$X264_ARCHIVE_URL" "x264-stable.tar.gz" "x264.tar.gz"
    patch_x264_ffmpeg_const_compatibility "$x264_dir"
    (
        cd "$x264_dir"
        ./configure --prefix="$build_dir" --enable-static
        run_make
        run_make install
    )
fi

x265_dir="$source_dir/x265_git"
if has_x265; then
    info "Skipping x265 because it is already installed"
else
    prepare_source_from_cache "x265" "$x265_dir" "build/linux" "$X265_ARCHIVE_NAME" "$X265_ARCHIVE_URL" "x265_git.tar.gz" "x265-master.tar.gz" "x265.tar.gz"
    (
        cd "$x265_dir/build/linux"
        cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="$build_dir" -DENABLE_LIBNUMA=OFF ../../source
        run_make
        run_make install
    )
fi

vvenc_dir="$source_dir/vvenc-$VVENC_VERSION"
if has_vvenc; then
    info "Skipping VVenC $VVENC_VERSION because it is already installed"
else
    prepare_source_from_cache "VVenC $VVENC_VERSION" "$vvenc_dir" "CMakeLists.txt" "$VVENC_ARCHIVE_NAME" "$VVENC_ARCHIVE_URL" "v$VVENC_VERSION.tar.gz" "vvenc-$VVENC_VERSION.tar.xz"
    vvenc_cmake_args=(-S "$vvenc_dir" -B "$vvenc_dir/build/release-shared" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DVVENC_LIBRARY_ONLY=ON -DVVENC_ENABLE_WERROR=OFF -DCMAKE_INSTALL_PREFIX="$build_dir" -DCMAKE_INSTALL_LIBDIR=lib)
    cmake "${vvenc_cmake_args[@]}"
    cmake --build "$vvenc_dir/build/release-shared" --target install -- -j"$JOBS"
fi

vvdec_dir="$source_dir/vvdec-$VVDEC_VERSION"
if has_vvdec; then
    info "Skipping VVdeC $VVDEC_VERSION because it is already installed"
else
    prepare_source_from_cache "VVdeC $VVDEC_VERSION" "$vvdec_dir" "CMakeLists.txt" "$VVDEC_ARCHIVE_NAME" "$VVDEC_ARCHIVE_URL" "v$VVDEC_VERSION.tar.gz" "vvdec-$VVDEC_VERSION.tar.xz"
    vvdec_cmake_args=(-S "$vvdec_dir" -B "$vvdec_dir/build/release-shared" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DVVDEC_LIBRARY_ONLY=ON -DVVDEC_ENABLE_WERROR=OFF -DCMAKE_INSTALL_PREFIX="$build_dir" -DCMAKE_INSTALL_LIBDIR=lib)
    cmake "${vvdec_cmake_args[@]}"
    cmake --build "$vvdec_dir/build/release-shared" --target install -- -j"$JOBS"
fi

pkg-config --exists libvvenc || fail "libvvenc was not found by pkg-config after installation."
pkg-config --exists libvvdec || fail "libvvdec was not found by pkg-config after installation."

CUDA_CFLAGS=""
CUDA_LDFLAGS=""
if [ "$USE_CUDA" = "true" ]; then
    info "Configuring optional CUDA/NVENC support"
    resolve_cuda
    NV_CODEC_HEADERS_ARCHIVE_NAME="nv-codec-headers-${NV_CODEC_HEADERS_TAG}.tar.gz"
    NV_CODEC_HEADERS_ARCHIVE_URL="https://github.com/FFmpeg/nv-codec-headers/archive/refs/tags/${NV_CODEC_HEADERS_TAG}.tar.gz"
    nv_codec_headers_dir="$source_dir/nv-codec-headers-$NV_CODEC_HEADERS_TAG"
    prepare_source_from_cache "nv-codec-headers $NV_CODEC_HEADERS_TAG" "$nv_codec_headers_dir" "Makefile" "$NV_CODEC_HEADERS_ARCHIVE_NAME" "$NV_CODEC_HEADERS_ARCHIVE_URL" "$NV_CODEC_HEADERS_TAG.tar.gz"
    run_make -C "$nv_codec_headers_dir" PREFIX="$build_dir" install
    pkg-config --exists ffnvcodec || fail "ffnvcodec was not found by pkg-config after installation."
    CUDA_CFLAGS=" -I$CUDA_HOME/include"
    CUDA_LDFLAGS=" -L$CUDA_LIB_DIR -Wl,-rpath,$CUDA_LIB_DIR"
fi

ffmpeg_dir="$source_dir/ffmpeg-$FFMPEG_VERSION"
if has_complete_ffmpeg; then
    info "Skipping FFmpeg $FFMPEG_VERSION because it is already installed"
else
    prepare_source_from_cache "FFmpeg $FFMPEG_VERSION" "$ffmpeg_dir" "configure" "$FFMPEG_ARCHIVE_NAME" "$FFMPEG_ARCHIVE_URL" "ffmpeg-$FFMPEG_VERSION.tar.gz"
    vvdec_patch="$(ensure_cached_file "$VVDEC_PATCH_NAME" "$VVDEC_PATCH_URL" "v8-0001-avcodec-add-external-dec-libvvdec-for-H266-VVC.patch")"
    validate_vvdec_patch_file "$vvdec_patch"

    if ! ffmpeg_has_verified_vvdec_integration "$ffmpeg_dir"; then
        if vvdec_patch_is_reverse_applicable "$ffmpeg_dir" "$vvdec_patch"; then
            info "Existing FFmpeg source tree has an incomplete VVdeC patch state; recreating it from the cached archive"
            rm -rf "$ffmpeg_dir"
            prepare_source_from_cache "FFmpeg $FFMPEG_VERSION" "$ffmpeg_dir" "configure" "$FFMPEG_ARCHIVE_NAME" "$FFMPEG_ARCHIVE_URL" "ffmpeg-$FFMPEG_VERSION.tar.gz"
        fi
        apply_vvdec_patch "$ffmpeg_dir" "$vvdec_patch"
    else
        info "VVdeC patch is already applied and verified"
    fi

    ffmpeg_has_verified_vvdec_integration "$ffmpeg_dir" || fail "FFmpeg source verification failed: --enable-libvvdec is unavailable."

    info "Building FFmpeg $FFMPEG_VERSION"
    ffmpeg_configure_args=(--prefix="$build_dir" --pkg-config-flags="--static" --extra-cflags="-I$build_dir/include$CUDA_CFLAGS" --extra-ldflags="-L$build_dir/lib -Wl,-rpath,$build_dir/lib$CUDA_LDFLAGS" --bindir="$bin_dir" --enable-gpl --enable-rpath --enable-libx264 --enable-libx265 --enable-libvvenc --enable-libvvdec)
    if [ "$USE_CUDA" = "true" ]; then
        ffmpeg_configure_args+=(--enable-nonfree --enable-cuda-nvcc --enable-libnpp --enable-cuda --enable-nvenc)
    fi
    (
        cd "$ffmpeg_dir"
        PKG_CONFIG_PATH="$PKG_CONFIG_PATH" ./configure "${ffmpeg_configure_args[@]}"
        run_make
        run_make install
    )
fi

write_environment_file "$environment_file"
printf '%s\n' "$environment_file" > "$environment_pointer_file"

info "Validating the FFmpeg installation"
"$bin_dir/ffmpeg" -hide_banner -h encoder=libvvenc >/dev/null
"$bin_dir/ffmpeg" -hide_banner -h decoder=libvvdec >/dev/null

printf '\nFFmpeg installation completed successfully.\n'
printf 'Download cache directory: %s\n' "$package_dir"
printf 'Environment file: %s\n' "$environment_file"
printf 'Environment path file: %s\n' "$environment_pointer_file"
printf 'VVC validation: %s\n' "$("$bin_dir/ffmpeg" -hide_banner -codecs | grep ' H.266 / VVC ' | tr -s ' ')"
