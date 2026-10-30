import os
import shutil
import subprocess


def to_h264(path):
    ff = shutil.which("ffmpeg")
    if not ff or not os.path.exists(path):
        return False
    tmp = path + ".h264.tmp.mp4"
    r = subprocess.run([ff, "-v", "error", "-y", "-i", path, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                        "-preset", "veryfast", "-movflags", "+faststart", "-threads", "2", tmp])
    if r.returncode != 0:
        if os.path.exists(tmp):
            os.remove(tmp)
        return False
    os.replace(tmp, path)
    return True
