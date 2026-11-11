from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

input = DATA_DIR / "traj_mantel_new2.npz"
output = DATA_DIR / "traj_mantel_new2_smooth.npz"
sigma = 3.0

def smooth_quaternions(q, sigma):
    q = q.copy()

    # q and -q represent the same rotation. Remove sign flips first,
    # otherwise averaging can create artificial orientation jumps.
    for g in range(q.shape[1]):
        for i in range(1, q.shape[0]):
            if np.dot(q[i - 1, g], q[i, g]) < 0:
                q[i, g] *= -1

    q = gaussian_filter1d(q, sigma=sigma, axis=0, mode="nearest")

    # Smoothed quaternion components must be projected back to unit length.
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return q


def smooth_npz(input_file, output_file, sigma):
    with np.load(input_file) as f:
        data = {k: f[k].copy() for k in f.files}

    # Smooth only the robot-command trajectories; preserve cloth states,
    # timestamps, jaw states, mesh data, and metadata exactly as stored.
    data["gripper_positions"] = gaussian_filter1d(
        data["gripper_positions"], sigma=sigma, axis=0, mode="nearest"
    )
    data["gripper_quaternions"] = smooth_quaternions(
        data["gripper_quaternions"], sigma
    )

    np.savez_compressed(output_file, **data)


if __name__ == "__main__":
    with np.load(input) as d:
        frames = d["cloth_frames"]
        print(f'number of entries: {frames.shape[0]}')
    smooth_npz(input, output, sigma)
