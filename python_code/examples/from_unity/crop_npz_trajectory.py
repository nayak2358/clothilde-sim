from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

input_file = DATA_DIR / "traj_mantel_new2_smooth.npz"
output_file = DATA_DIR / "traj_mantel_new2_smooth_cropped.npz"

FIRST_FRAME = 200
LAST_FRAME = 380   # inclusive

with np.load(input_file) as f:
    data = {key: f[key].copy() for key in f.files}

# Arrays with one entry per recorded frame.
frame_keys = [
    "cloth_frames",
    "gripper_positions",
    "gripper_quaternions",
    "jaw_status",
    "times",
]

start = FIRST_FRAME
stop = LAST_FRAME + 1  # slicing excludes stop

for key in frame_keys:
    if key in data:
        data[key] = data[key][start:stop]

# Reset cropped trajectory time so it starts at 0.
if "times" in data and len(data["times"]) > 0:
    data["times"] -= data["times"][0]

# Mesh geometry, faces, simulator parameters, etc. stay unchanged.
np.savez_compressed(output_file, **data)

print(f"Saved frames {FIRST_FRAME} to {LAST_FRAME} -> {output_file}")
