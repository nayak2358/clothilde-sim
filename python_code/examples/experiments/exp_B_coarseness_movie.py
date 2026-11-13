#!/usr/bin/env python3
"""
Polyscope movie for the mesh-coarseness shearing experiment.

This follows the simple format of exp_B_shearing_movie.py:
- choose the model and resolution at the top;
- create the mesh with createRectangularMesh();
- simulate the loading;
- call cloth.makeMovie().

Run this file four times to obtain screenshots for:
    four_point, N=7
    diagonal,   N=7
    four_point, N=21
    diagonal,   N=21

Recommended location:
    python_code/examples/experiments/exp_B_coarseness_movie.py
"""

import inspect
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
PYTHON_CODE = HERE.parents[2]

if str(PYTHON_CODE) not in sys.path:
    sys.path.insert(0, str(PYTHON_CODE))

from implementation.Cloth import Cloth
from implementation.utils import createRectangularMesh


# -----------------------------------------------------------------------------
# Change only these two lines for each movie
# -----------------------------------------------------------------------------

# SHEAR_MODEL = "four_point"
SHEAR_MODEL = "diagonal"

NODES_PER_SIDE = 21
# NODES_PER_SIDE = 21


# -----------------------------------------------------------------------------
# Same settings as H2_coarseness_comparison.py
# -----------------------------------------------------------------------------

CLOTH_SIZE = 1.0
INITIAL_CURVATURE = 0.001
INITIAL_HEIGHT = 1.0

TOP_DISPLACEMENT = 0.25
TOP_PROFILE_HALF_WIDTH = 0.25

FRAME_DT = 1.0 / 60.0
SUB_STEPS = 8

LOADING_FRAMES = 80
HOLD_FRAMES = 100

REFERENCE_N = 21
REFERENCE_EDGE_LENGTH = (
    CLOTH_SIZE / (REFERENCE_N - 1)
)

REFERENCE_STRETCH_COMPLIANCE = 1.0e-6
REFERENCE_FOUR_POINT_COMPLIANCE = 1.0e-3

DIAGONAL_COMPLIANCE_FACTOR = 0.5

REFERENCE_ETA_SHEAR = 0.005
SHEAR_JACOBIAN_REFRESH = 5

MOVIE_SPEED = 1
REPEAT_MOVIE = True
SMOOTHING_PASSES = 0


def smoothstep(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def compact_top_profile(x):
    normalized = x / TOP_PROFILE_HALF_WIDTH

    profile = np.zeros_like(
        normalized
    )

    active = (
        np.abs(normalized) < 1.0
    )

    profile[active] = (
        1.0
        - normalized[active] ** 2
    ) ** 2

    return TOP_DISPLACEMENT * profile


# -----------------------------------------------------------------------------
# Mesh and model
# -----------------------------------------------------------------------------

n = NODES_PER_SIDE

X, T = createRectangularMesh(
    a=CLOTH_SIZE,
    b=CLOTH_SIZE,
    na=n,
    nb=n,
    h=INITIAL_CURVATURE,
)

X[:, 2] += INITIAL_HEIGHT

cloth = Cloth(
    X.copy(),
    T.copy(),
    shear_model=SHEAR_MODEL,
)


# -----------------------------------------------------------------------------
# Resolution-consistent compliances
# -----------------------------------------------------------------------------

edge_length = CLOTH_SIZE / (n - 1)

compliance_scale = (
    edge_length
    / REFERENCE_EDGE_LENGTH
) ** 2

stretch_compliance = (
    REFERENCE_STRETCH_COMPLIANCE
    * compliance_scale
)

four_point_compliance = (
    REFERENCE_FOUR_POINT_COMPLIANCE
    * compliance_scale
)

if SHEAR_MODEL == "four_point":
    shear_compliance = (
        four_point_compliance
    )

elif SHEAR_MODEL == "diagonal":
    shear_compliance = (
        DIAGONAL_COMPLIANCE_FACTOR
        * four_point_compliance
    )

else:
    raise ValueError(
        "SHEAR_MODEL must be "
        "'four_point' or 'diagonal'."
    )


parameters = {
    "dt": FRAME_DT,
    "sub_steps": SUB_STEPS,
    "tol": 0.0075,
    "rho": 0.1,

    # No gravity or aerodynamic loading.
    "delta": 0.0,

    # Damping is used to approach equilibrium.
    "alpha": 1.0,
    "beta": 0.0,

    # Isolate stretching and shearing.
    "kappa": 0.0,
    "kappa_bnd": 0.0,
    "str": stretch_compliance,
    "shr": shear_compliance,

    # Contact and friction should remain inactive.
    "slf": 1.0e-4,
    "mu_f": 0.0,
    "mu_s": 0.0,
    "thck": 0.95,
    "max_mov": 0.25,
}

signature = inspect.signature(
    cloth.setSimulatorParameters
)

if "eta_shr" in signature.parameters:
    parameters["eta_shr"] = (
        REFERENCE_ETA_SHEAR
        * compliance_scale
        if SHEAR_MODEL == "four_point"
        else 0.0
    )

if "shear_refresh" in signature.parameters:
    parameters["shear_refresh"] = (
        SHEAR_JACOBIAN_REFRESH
    )

cloth.setSimulatorParameters(
    **parameters
)


# -----------------------------------------------------------------------------
# Same continuous non-uniform boundary condition for every resolution
# -----------------------------------------------------------------------------

bottom_edge = np.arange(
    0,
    n,
    dtype=int,
)

top_edge = np.arange(
    (n - 1) * n,
    n * n,
    dtype=int,
)

controlled_nodes = np.concatenate(
    (
        bottom_edge,
        top_edge,
    )
)

control = controlled_nodes.tolist()

initial_targets = X[
    controlled_nodes
].copy()

top_displacement = compact_top_profile(
    X[top_edge, 0]
)

final_targets = initial_targets.copy()

final_targets[n:, 0] += (
    top_displacement
)


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

print(
    f"Running model={SHEAR_MODEL}, "
    f"N={NODES_PER_SIDE}"
)

print(
    f"Shear block: "
    f"{type(cloth.shear[0]).__name__}; "
    f"constraints="
    f"{cloth.shear[0].n_conds}"
)

for frame in range(LOADING_FRAMES):
    fraction = smoothstep(
        (frame + 1)
        / LOADING_FRAMES
    )

    targets = initial_targets.copy()

    targets[n:, 0] += (
        fraction
        * top_displacement
    )

    cloth.simulate(
        targets.copy(),
        control,
    )

    if (
        frame == 0
        or (frame + 1) % 10 == 0
        or frame + 1 == LOADING_FRAMES
    ):
        print(
            f"Loading frame "
            f"{frame + 1}/"
            f"{LOADING_FRAMES}"
        )


# -----------------------------------------------------------------------------
# Fixed hold at the final boundary configuration
# -----------------------------------------------------------------------------

# for frame in range(HOLD_FRAMES):
#     cloth.simulate(
#         final_targets.copy(),
#         control,
#     )

#     if (
#         (frame + 1) % 20 == 0
#         or frame + 1 == HOLD_FRAMES
#     ):
#         print(
#             f"Hold frame "
#             f"{frame + 1}/"
#             f"{HOLD_FRAMES}"
#         )


# -----------------------------------------------------------------------------
# Polyscope movie
# -----------------------------------------------------------------------------

print(
    f"Stored frames: "
    f"{len(cloth.history_pos)}"
)

cloth.makeMovie(
    speed=MOVIE_SPEED,
    repeat=REPEAT_MOVIE,
    smooth=SMOOTHING_PASSES,
)
