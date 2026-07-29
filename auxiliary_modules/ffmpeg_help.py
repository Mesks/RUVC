import subprocess

def get_ffmpeg_x265_version():
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
        version_info = result.stdout

        for line in version_info.splitlines():
            if "libx265" in line:
                return line.strip()
        return "x265 not found in FFmpeg configuration."
    except FileNotFoundError:
        return "FFmpeg is not installed or not in PATH."
    
def get_x265_version():
    try:
        result = subprocess.run(["x265", "--version"], capture_output=True, text=True, check=True)
        version_info = result.stdout.splitlines()[0]
        return version_info
    except FileNotFoundError:
        return "x265 is not installed or not in PATH."
    except subprocess.CalledProcessError as e:
        return f"Error running x265: {e}"
