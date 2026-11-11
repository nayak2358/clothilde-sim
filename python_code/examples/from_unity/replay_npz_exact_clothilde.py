"""
Replay EXACT recorded Unity motion from traj_mantel.npz in Clothilde.

This is a recorded-motion player, NOT a new physics simulation.

It does NOT:
  - construct a rest mesh,
  - call setSimulatorParameters(),
  - call SimulateGripper.step(),
  - change cloth positions through Clothilde dynamics,
  - add any dz/ground offset.

At frame k:
    cloth.positions = cloth_frames[k]

and both gripper boxes are placed directly from:
    gripper_positions[k, 0/1]
    gripper_quaternions[k, 0/1]

All orientation handling uses Clothilde's quaternion utilities.
No rotation matrices are used.
"""

from pathlib import Path
import sys
import time

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

import trimesh


#region Read .npz file
# =============================================================================
# USER SETTINGS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

NPZ_PATH = DATA_DIR / "traj_mantel_new2_smooth_cropped.npz"
# NPZ_PATH = DATA_DIR / "traj_cloth2.npz"

SMOOTH = 2

SHOW_GRASP_BOXES = True
SHOW_NODES = False

SHOW_TRACE = True

# Must match the Unity grasp-box dimensions if you want the box itself drawn.
# This affects VISUALIZATION ONLY; recorded gripper poses are unchanged.
GRASP_BOX = np.array([0.04, 0.04, 0.04], dtype=np.float64)

# Because the exported transform is graspFrame itself.
# TIP_CENTER_LOCAL = np.zeros(3, dtype=np.float64)
TIP_CENTER_LOCAL = 0.001 * np.array([0.0, 0.0, - (52 - GRASP_BOX[2] * 1000 / 2)], dtype=float)

USE_RECORDED_TIMING = True
FIXED_FPS = 60.0

REPEAT = True

GROUND_Z = 0.0

# The Unity-coupled Python frame uses:
#     z_python = y_unity - 1
#
# Standalone Clothilde uses its existing ground at z = 0.
# Therefore replayed positions need the inverse origin shift:
#     z_clothilde = z_recorded + 1
#
# Apply the SAME translation to cloth and both grippers.
# Quaternions are not changed.
DISPLAY_OFFSET = np.array([0.0, 0.0, 1.0], dtype=np.float64)


# =============================================================================
# CLOTHILDE IMPORTS
# =============================================================================

def add_clothilde_root():
    here = Path(__file__).resolve().parent
    cwd = Path.cwd()

    candidates = [
        here,
        here / "python_code",
        here.parent,
        here.parent / "python_code",
        cwd,
        cwd / "python_code",
        cwd.parent,
        cwd.parent / "python_code",
    ]

    for root in candidates:
        if (root / "implementation" / "Gripper.py").exists():
            sys.path.insert(0, str(root))
            return root

    raise FileNotFoundError(
        "Could not find the Clothilde python_code directory. "
        "Edit add_clothilde_root() if needed."
    )


CLOTHILDE_ROOT = add_clothilde_root()

from implementation.Cloth_speed import Cloth
from implementation.Gripper import (
    quat_transform_points,
    quat_normalize,
)


# =============================================================================
# LOAD
# =============================================================================

d = np.load(NPZ_PATH, allow_pickle=False)

X0 = d["mesh_vertices"].astype(np.float64)
T0 = d["mesh_faces"].astype(np.int32)

frames = d["cloth_frames"].astype(np.float64)

p = d["gripper_positions"].astype(np.float64)
q = d["gripper_quaternions"].astype(np.float64)
jaw = d["jaw_status"].astype(bool)

times = d["times"].astype(np.float64)

# Convert the recorded Unity-coupled Python origin to the standalone
# Clothilde display origin. This is a pure translation.
X0 = X0 + DISPLAY_OFFSET
frames = frames + DISPLAY_OFFSET.reshape(1, 1, 3)
p = p + DISPLAY_OFFSET.reshape(1, 1, 3)

n_frames = frames.shape[0]

if p.shape != (n_frames, 2, 3):
    raise ValueError(
        f"Expected gripper_positions shape ({n_frames},2,3), got {p.shape}."
    )

if q.shape != (n_frames, 2, 4):
    raise ValueError(
        f"Expected gripper_quaternions shape ({n_frames},2,4), got {q.shape}."
    )

if jaw.shape != (n_frames, 2):
    raise ValueError(
        f"Expected jaw_status shape ({n_frames},2), got {jaw.shape}."
    )

print("Loaded:", NPZ_PATH)
print("mesh:", X0.shape)
print("faces:", T0.shape)
print("cloth frames:", frames.shape)
print("gripper positions:", p.shape)
print("gripper quaternions:", q.shape)
print("jaw status:", jaw.shape)
print("standalone Clothilde display offset:", DISPLAY_OFFSET)
print("minimum replay cloth z after offset:", float(np.min(frames[..., 2])))

print()
print("Exact recorded z ranges:")
print(
    f"  cloth: {float(np.min(frames[...,2])):.6f} .. "
    f"{float(np.max(frames[...,2])):.6f}"
)
print(
    f"  left:  {float(np.min(p[:,0,2])):.6f} .. "
    f"{float(np.max(p[:,0,2])):.6f}"
)
print(
    f"  right: {float(np.min(p[:,1,2])):.6f} .. "
    f"{float(np.max(p[:,1,2])):.6f}"
)

if np.min(frames[..., 2]) < GROUND_Z:
    print(
        "\nNOTE: the recorded cloth contains points below z=0. "
    )


# =============================================================================
# CLOTHILDE VISUALIZATION OBJECT
# =============================================================================

# Cloth() is used only for Clothilde's topology/smoothing/Polyscope support.
# We intentionally do NOT call setSimulatorParameters().
#
# Since no physics is run, the fact that the recording-start mesh is deformed
# is irrelevant here.
cloth = Cloth(X0.copy(), T0.copy())
cloth.positions = frames[0].copy()

if cloth.polyscoped is False:
    cloth.preparePolyscope()

cloth.label = getattr(cloth, "label", "cloth")

if SHOW_TRACE:
    ps.register_point_cloud("trace_left", p[:,0,:], radius=0.005, color=(0.0, 1.0, 1.0))
    ps.register_point_cloud("trace_right", p[:,1,:], radius=0.005, color=(1.0, 0.4, 0.7))

#region Gripper
# =============================================================================
# GRASP-BOX GEOMETRY — QUATERNIONS ONLY
# =============================================================================

box_faces = np.array([
    [0,1,2], [0,2,3],
    [4,5,6], [4,6,7],
    [0,1,5], [0,5,4],
    [1,2,6], [1,6,5],
    [2,3,7], [2,7,6],
    [3,0,4], [3,4,7],
], dtype=np.int32)


def grasp_box_vertices(position, quaternion):
    hx, hy, hz = 0.5 * GRASP_BOX
    c = TIP_CENTER_LOCAL.reshape(3)

    local = np.array([
        [-hx,-hy,-hz],
        [ hx,-hy,-hz],
        [ hx, hy,-hz],
        [-hx, hy,-hz],
        [-hx,-hy, hz],
        [ hx,-hy, hz],
        [ hx, hy, hz],
        [-hx, hy, hz],
    ], dtype=np.float64)

    local += c

    return quat_transform_points(
        np.asarray(position, dtype=np.float64),
        quat_normalize(quaternion),
        local,
    )

if SHOW_GRASP_BOXES:
    ps.register_surface_mesh(
        "left_grasp_box",
        grasp_box_vertices(p[0, 0], q[0, 0]),
        box_faces,
        transparency=0.55,
        material="wax",
    )

    ps.register_surface_mesh(
        "right_grasp_box",
        grasp_box_vertices(p[0, 1], q[0, 1]),
        box_faces,
        transparency=0.55,
        material="wax",
    )
    
##### Gripper
YELLOW = [186/256, 142/256, 35/256]
jaw_gap_open = 0.0
jaw_gap_closed = -0.001 * (30 - 6)

def load_mesh(path):
    m = trimesh.load_mesh(path)
    return np.asarray(m.vertices), np.asarray(m.faces)

def transform_mesh(V_local, q, p, local_offset=np.zeros(3)):
    V = np.asarray(V_local, dtype=float) + np.asarray(local_offset, dtype=float).reshape(1, 3)
    return quat_transform_points(np.asarray(p, dtype=float), quat_normalize(q), V)

V_base0, F_base = load_mesh(PROJECT_ROOT/"gripper_cad_files/base.stl")
V_left0, F_left = load_mesh(PROJECT_ROOT/"gripper_cad_files/jaw_left.stl")
V_right0, F_right = load_mesh(PROJECT_ROOT/"gripper_cad_files/jaw_right.stl")

mesh_scale = 0.001   # change to 0.001 if CAD comes in mm
V_base0 *= mesh_scale
V_left0 *= mesh_scale
V_right0 *= mesh_scale

ps.register_surface_mesh(
    "gripper_base1",
    transform_mesh(V_base0, q[0, 0], p[0, 0]),
    F_base, color=YELLOW
)
ps.register_surface_mesh(
    "gripper_left1",
    transform_mesh(V_left0, q[0, 0], p[0, 0], local_offset=np.array([-jaw_gap_open/2, 0, 0])),
    F_left, color=YELLOW
)
ps.register_surface_mesh(
    "gripper_right1",
    transform_mesh(V_right0, q[0, 0], p[0, 0], local_offset=np.array([-jaw_gap_open/2, 0, 0])),
    F_right, color=YELLOW
)

ps.register_surface_mesh(
    "gripper_base2",
    transform_mesh(V_base0, q[0, 1], p[0, 1]),
    F_base, color=YELLOW
)
ps.register_surface_mesh(
    "gripper_left2",
    transform_mesh(V_left0, q[0, 1], p[0, 1], local_offset=np.array([-jaw_gap_open/2, 0, 0])),
    F_left, color=YELLOW
)
ps.register_surface_mesh(
    "gripper_right2",
    transform_mesh(V_right0, q[0, 1], p[0, 1], local_offset=np.array([-jaw_gap_open/2, 0, 0])),
    F_right, color=YELLOW
)



# =============================================================================
# TIMING
# =============================================================================

if len(times) > 1:
    dts = np.diff(times)
    dts = dts[np.isfinite(dts) & (dts > 0)]

    default_dt = (
        float(np.median(dts))
        if len(dts)
        else 1.0 / FIXED_FPS
    )
else:
    default_dt = 1.0 / FIXED_FPS

state = {
    "k": 0,
    "last_time": time.time(),
    "paused": False,
}

#region Update scene
# =============================================================================
# EXACT FRAME UPDATE
# =============================================================================

def update_frame(k):
    # Exact recorded vertex state.
    cloth.positions = frames[k].copy()

    # Use Clothilde's existing smoothing representation only for rendering.
    phi_display = cloth.Am @ cloth.positions

    for _ in range(SMOOTH):
        phi_display = cloth.S @ phi_display

    ps.get_surface_mesh(cloth.label).update_vertex_positions(phi_display)

    try:
        ps.get_point_cloud(cloth.label).update_point_positions(cloth.positions)
    except Exception:
        pass

    if SHOW_GRASP_BOXES:
        left_box = ps.get_surface_mesh("left_grasp_box")
        right_box = ps.get_surface_mesh("right_grasp_box")

        left_box.update_vertex_positions(
            grasp_box_vertices(p[k, 0], q[k, 0])
        )

        right_box.update_vertex_positions(
            grasp_box_vertices(p[k, 1], q[k, 1])
        )

        # Jaw state is visualization only.
        left_box.set_color(
            [0.0, 1.0, 0.0]
            if jaw[k, 0]
            else [1.0, 0.0, 0.0]
        )

        right_box.set_color(
            [0.0, 0.6, 1.0]
            if jaw[k, 1]
            else [1.0, 0.0, 0.0]
        )
        
    current_gap1 = jaw_gap_open if jaw[k, 0] else jaw_gap_closed
    current_gap2 = jaw_gap_open if jaw[k, 1] else jaw_gap_closed

    ps.get_surface_mesh("gripper_base1").update_vertex_positions(
        transform_mesh(V_base0, q[k, 0], p[k, 0])
    )
    ps.get_surface_mesh("gripper_left1").update_vertex_positions(
        transform_mesh(V_left0, q[k, 0], p[k, 0], local_offset=np.array([-current_gap1/2, 0, 0]))
    )
    ps.get_surface_mesh("gripper_right1").update_vertex_positions(
        transform_mesh(V_right0, q[k, 0], p[k, 0], local_offset=np.array([ current_gap1/2, 0, 0]))
    )
    
    ps.get_surface_mesh("gripper_base2").update_vertex_positions(
        transform_mesh(V_base0, q[k, 1], p[k, 1])
    )
    ps.get_surface_mesh("gripper_left2").update_vertex_positions(
        transform_mesh(V_left0, q[k, 1], p[k, 1], local_offset=np.array([-current_gap2/2, 0, 0]))
    )
    ps.get_surface_mesh("gripper_right2").update_vertex_positions(
        transform_mesh(V_right0, q[k, 1], p[k, 1], local_offset=np.array([ current_gap2/2, 0, 0]))
    )
    
    # psim.TextUnformatted(f"Unity frame number = {k}")


# =============================================================================
# CALLBACK
# =============================================================================

def callback():
    if state["paused"]:
        return

    k = state["k"]
    
    psim.TextUnformatted(f"Unity frame number = {k}")

    if USE_RECORDED_TIMING and k > 0:
        frame_dt = float(times[k] - times[k - 1])

        if not np.isfinite(frame_dt) or frame_dt <= 0:
            frame_dt = default_dt
    else:
        frame_dt = (
            default_dt
            if USE_RECORDED_TIMING
            else 1.0 / FIXED_FPS
        )

    now = time.time()

    if now - state["last_time"] < frame_dt:
        return

    state["last_time"] = now

    update_frame(k)

    if (
        k == 0
        or jaw[k, 0] != jaw[k - 1, 0]
        or jaw[k, 1] != jaw[k - 1, 1]
    ):
        print(
            f"frame={k:4d} | "
            f"left={'OPEN' if jaw[k,0] else 'CLOSED'} | "
            f"right={'OPEN' if jaw[k,1] else 'CLOSED'}"
        )

    state["k"] += 1

    if state["k"] >= n_frames:
        if REPEAT:
            state["k"] = 0
            state["last_time"] = time.time()
        else:
            state["k"] = n_frames - 1
            state["paused"] = True
            print("Reached final recorded frame.")


# =============================================================================
# RUN
# =============================================================================

update_frame(0)

ps.set_user_callback(callback)
ps.show()
ps.clear_user_callback()
