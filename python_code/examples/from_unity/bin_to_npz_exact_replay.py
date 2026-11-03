"""
Convert replay_mantel.bin (Unity export, two grippers) to a self-contained NPZ
for EXACT recorded-motion replay in Clothilde.

This file does NOT:
  - reconstruct a rest mesh,
  - run physics,
  - add a ground offset,
  - rotate/re-frame the trajectory again.

Why no coordinate transform here?
---------------------------------
The current Unity exporter already writes the gripper in Clothilde/Python
coordinates:

    position:
        [x, y, z]_Unity -> [x, z, y - 1]_Clothilde

    quaternion:
        [qw, qx, qy, qz]_Unity -> [qw, -qx, -qz, -qy]_Clothilde

The cloth frames are also exported from the Python/Clothilde simulation state.
Therefore applying another Unity->Python transform would be wrong.

NPZ contents
------------
    mesh_vertices           (V, 3)
    mesh_faces              (F, 4)
    cloth_frames            (N, V, 3)

    gripper_positions       (N, 2, 3)
    gripper_quaternions     (N, 2, 4), [qw,qx,qy,qz]
    jaw_status              (N, 2), bool

    times                   (N,)
    sim_params              (14,)      # metadata only
    sim_param_names

Gripper index:
    0 = left
    1 = right
"""

from pathlib import Path
import numpy as np


# =============================================================================
# USER SETTINGS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

INPUT_BIN = DATA_DIR / "replay_mantel_short.bin"
OUTPUT_NPZ = DATA_DIR / "traj_mantel_short.npz"

# Optional crop of the recorded motion.
# Both indices are inclusive.
START_INDEX = 0
END_INDEX = None

# If True, only the time vector is rebased so the selected segment starts at 0.
# Geometry is never changed.
REBASE_TIME = True


# =============================================================================
# BINARY READER
# =============================================================================

PARAM_NAMES = np.array([
    "dt", "tol", "sub_steps", "rho", "delta", "alpha",
    "kappa", "kappa_bnd", "str", "shr", "slf",
    "mu_f", "mu_s", "thck",
])


def read_exact(f, dtype, count, name):
    values = np.fromfile(f, dtype=dtype, count=count)

    if values.size != count:
        raise ValueError(
            f"Unexpected end of file while reading {name}: "
            f"expected {count}, got {values.size}."
        )

    return values


def read_replay_bin(path):
    path = Path(path)

    with path.open("rb") as f:
        magic = f.read(8).decode("ascii")

        if magic != "CLTHSIM1":
            raise ValueError(
                f"Wrong binary signature: expected 'CLTHSIM1', got {magic!r}."
            )

        version = int(read_exact(f, "<i4", 1, "version")[0])

        if version != 2:
            raise ValueError(
                f"Expected the two-gripper version-2 format, got version {version}."
            )

        header = read_exact(f, "<i4", 5, "header")
        n_frames, n_vertices, n_faces, n_params, n_grippers = map(int, header)

        if n_grippers not in (1, 2):
            raise ValueError(f"Expected 1 or 2 grippers, got {n_grippers}.")

        mesh_vertices = read_exact(
            f,
            "<f8",
            n_vertices * 3,
            "mesh vertices",
        ).reshape(n_vertices, 3)

        mesh_faces = read_exact(
            f,
            "<i4",
            n_faces * 4,
            "mesh faces",
        ).reshape(n_faces, 4)

        cloth_frames = read_exact(
            f,
            "<f8",
            n_frames * n_vertices * 3,
            "cloth frames",
        ).reshape(n_frames, n_vertices, 3)

        gripper_poses = read_exact(
            f,
            "<f4",
            n_frames * n_grippers * 7,
            "gripper poses",
        ).reshape(n_frames, n_grippers, 7)

        jaw_status = read_exact(
            f,
            "<i4",
            n_frames * n_grippers,
            "jaw status",
        ).reshape(n_frames, n_grippers)

        times = read_exact(
            f,
            "<f4",
            n_frames,
            "timestamps",
        )

        sim_params = read_exact(
            f,
            "<f4",
            n_params,
            "simulator parameters",
        )

        remaining = f.read()

        if remaining:
            raise ValueError(
                f"Binary layout mismatch: {len(remaining)} unread bytes remain."
            )

    return {
        "version": version,
        "n_frames": n_frames,
        "n_vertices": n_vertices,
        "n_faces": n_faces,
        "n_params": n_params,
        "n_grippers": n_grippers,

        "mesh_vertices": mesh_vertices,
        "mesh_faces": mesh_faces,
        "cloth_frames": cloth_frames,
        "gripper_poses": gripper_poses,
        "jaw_status": jaw_status,
        "times": times,
        "sim_params": sim_params,
    }


# =============================================================================
# CONVERSION
# =============================================================================

def main():
    data = read_replay_bin(INPUT_BIN)

    start = int(START_INDEX)
    end = (
        data["n_frames"] - 1
        if END_INDEX is None
        else int(END_INDEX)
    )

    if start < 0:
        raise IndexError("START_INDEX must be >= 0.")

    if end < start:
        raise IndexError(
            f"END_INDEX={end} must be >= START_INDEX={start}."
        )

    if end >= data["n_frames"]:
        raise IndexError(
            f"END_INDEX={end} is outside the recording. "
            f"Valid indices are 0..{data['n_frames'] - 1}."
        )

    sl = slice(start, end + 1)

    cloth_frames = data["cloth_frames"][sl].copy()

    poses = data["gripper_poses"][sl]
    gripper_positions = poses[..., 0:3].copy()
    gripper_quaternions = poses[..., 3:7].copy()

    jaw_status = data["jaw_status"][sl].astype(bool, copy=True)

    times = data["times"][sl].astype(np.float64, copy=True)

    if REBASE_TIME and len(times):
        times -= times[0]

    # No spatial transformation and no ground shift.
    np.savez_compressed(
        OUTPUT_NPZ,

        format_version=np.array(1, dtype=np.int32),
        coordinate_frame=np.array("clothilde_python"),
        source_bin=np.array(INPUT_BIN.name),

        start_index=np.array(start, dtype=np.int32),
        end_index=np.array(end, dtype=np.int32),

        mesh_vertices=data["mesh_vertices"].astype(np.float64),
        mesh_faces=data["mesh_faces"].astype(np.int32),
        cloth_frames=cloth_frames.astype(np.float64),

        gripper_positions=gripper_positions.astype(np.float64),
        gripper_quaternions=gripper_quaternions.astype(np.float64),
        jaw_status=jaw_status,

        times=times,

        sim_params=data["sim_params"].astype(np.float64),
        sim_param_names=PARAM_NAMES[:data["n_params"]],

        n_grippers=np.array(data["n_grippers"], dtype=np.int32),

        gripper_order=np.array(
            ["mouse"] if data["n_grippers"] == 1
            else ["left", "right"]),
        quaternion_order=np.array("qw_qx_qy_qz"),
    )

    print("Saved:", OUTPUT_NPZ)
    print()
    print(f"Source frames:          {start} ... {end}")
    print("cloth_frames:          ", cloth_frames.shape)
    print("gripper_positions:     ", gripper_positions.shape)
    print("gripper_quaternions:   ", gripper_quaternions.shape)
    print("jaw_status:            ", jaw_status.shape)
    print("times:                 ", times.shape)

    print()
    print("No spatial transformation was applied.")
    print("No dz / ground shift was applied.")

    print()
    print("Recorded coordinate ranges:")
    print(
        f"  cloth z:   "
        f"{float(np.min(cloth_frames[..., 2])):.6f} .. "
        f"{float(np.max(cloth_frames[..., 2])):.6f}"
    )
    # print(
    #     f"  left z:    "
    #     f"{float(np.min(gripper_positions[:, 0, 2])):.6f} .. "
    #     f"{float(np.max(gripper_positions[:, 0, 2])):.6f}"
    # )
    # print(
    #     f"  right z:   "
    #     f"{float(np.min(gripper_positions[:, 1, 2])):.6f} .. "
    #     f"{float(np.max(gripper_positions[:, 1, 2])):.6f}"
    # )
    
    for i in range(data["n_grippers"]):
        print(
            f"  gripper {i} z: "
            f"{float(np.min(gripper_positions[:, i, 2])):.6f} .. "
            f"{float(np.max(gripper_positions[:, i, 2])):.6f}"
        )


if __name__ == "__main__":
    main()
