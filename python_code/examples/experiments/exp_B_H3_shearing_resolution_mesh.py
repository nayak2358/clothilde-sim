#!/usr/bin/env python3
"""
Non-uniform deformation and shear-only timing comparison at N=21.

This script produces two separate publication figures:

1. nonuniform_deformation_N21.pdf
   Two final deformation panels placed side by side.

2. shear_projection_time_N21.pdf
   Median shear-projection time with interquartile-range error bars.
   The actual number of shear equations is annotated above each bar.

Assumption
----------
Cloth.simulate() updates:

    cloth.last_shear_time

with the time spent only in the shear-projection block for that simulation
frame.

Recommended location
--------------------
python_code/examples/experiments/nonuniform_deformation_and_time_N21.py
"""

import csv
import inspect
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# Repository imports
# -----------------------------------------------------------------------------

HERE = Path(__file__).resolve()
PYTHON_CODE = HERE.parents[2]

if str(PYTHON_CODE) not in sys.path:
    sys.path.insert(0, str(PYTHON_CODE))

from implementation.Cloth import Cloth
from implementation.utils import createRectangularMesh


# -----------------------------------------------------------------------------
# Experiment settings
# -----------------------------------------------------------------------------

NODES_PER_SIDE = 21
REPEATS = 5

CLOTH_SIZE = 1.0
INITIAL_CURVATURE = 0.001
INITIAL_HEIGHT = 1.0

# Stable non-uniform displacement profile on the complete upper boundary.
TOP_DISPLACEMENT = 0.15
TOP_PROFILE_HALF_WIDTH = 0.30

FRAME_DT = 1.0 / 60.0
SUB_STEPS = 8
SOLVER_TOLERANCE = 0.0075

LOADING_FRAMES = 120
HOLD_FRAMES = 40
WARMUP_FRAMES = 10

STRETCH_COMPLIANCE = 1.0e-6
FOUR_POINT_SHEAR_COMPLIANCE = 1.0e-3
DIAGONAL_COMPLIANCE_FACTOR = 0.5

ETA_SHEAR = 0.005
SHEAR_JACOBIAN_REFRESH = 5

OUTPUT_DIRECTORY = (
    HERE.parent / "H3_deformation_and_time_results"
)


# -----------------------------------------------------------------------------
# LaTeX plotting
# -----------------------------------------------------------------------------

def configure_latex_plots():
    if shutil.which("latex") is None:
        raise RuntimeError(
            "A LaTeX installation is required. "
            "Ensure that `latex --version` works in this environment."
        )

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{amsmath}",
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def save_figure(figure, filename):
    figure.savefig(
        OUTPUT_DIRECTORY / f"{filename}.pdf",
        bbox_inches="tight",
    )

    figure.savefig(
        OUTPUT_DIRECTORY / f"{filename}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# -----------------------------------------------------------------------------
# Geometry and diagnostics
# -----------------------------------------------------------------------------

def smoothstep(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def compact_top_profile(x):
    """
    Compact C1 profile:
        u(x) = A [1 - (x/w)^2]^2,  |x| < w,
             = 0,                  otherwise.
    """

    normalized = x / TOP_PROFILE_HALF_WIDTH
    displacement = np.zeros_like(normalized)

    active = np.abs(normalized) < 1.0

    displacement[active] = TOP_DISPLACEMENT * (
        1.0 - normalized[active] ** 2
    ) ** 2

    return displacement


def material_edges(n):
    horizontal = [
        (row * n + column, row * n + column + 1)
        for row in range(n)
        for column in range(n - 1)
    ]

    vertical = [
        (row * n + column, (row + 1) * n + column)
        for row in range(n - 1)
        for column in range(n)
    ]

    return np.asarray(
        horizontal + vertical,
        dtype=int,
    )


def rms_edge_strain(positions, reference, n):
    edges = material_edges(n)

    current_lengths = np.linalg.norm(
        positions[edges[:, 1]]
        - positions[edges[:, 0]],
        axis=1,
    )

    reference_lengths = np.linalg.norm(
        reference[edges[:, 1]]
        - reference[edges[:, 0]],
        axis=1,
    )

    strain = (
        current_lengths - reference_lengths
    ) / reference_lengths

    return float(
        np.sqrt(np.mean(strain * strain))
    )


# -----------------------------------------------------------------------------
# Clothilde setup
# -----------------------------------------------------------------------------

def configure_cloth(cloth, model):
    if model == "four_point":
        shear_compliance = FOUR_POINT_SHEAR_COMPLIANCE

    elif model == "diagonal":
        shear_compliance = (
            DIAGONAL_COMPLIANCE_FACTOR
            * FOUR_POINT_SHEAR_COMPLIANCE
        )

    else:
        raise ValueError(model)

    parameters = {
        "dt": FRAME_DT,
        "sub_steps": SUB_STEPS,
        "tol": SOLVER_TOLERANCE,
        "rho": 0.1,

        # Isolate in-plane stretching and shearing.
        "delta": 0.0,
        "alpha": 1.0,
        "beta": 0.0,
        "kappa": 0.0,
        "kappa_bnd": 0.0,

        "str": STRETCH_COMPLIANCE,
        "shr": shear_compliance,

        # Contacts and friction should remain inactive.
        "slf": 1.0e-4,
        "mu_f": 0.0,
        "mu_s": 0.0,
        "thck": 0.95,
        "max_mov": 0.10,
    }

    signature = inspect.signature(
        cloth.setSimulatorParameters
    )

    if "eta_shr" in signature.parameters:
        # Keep positive because the four-point reference scalar product
        # can be zero on an initially orthogonal grid.
        parameters["eta_shr"] = ETA_SHEAR

    if "shear_refresh" in signature.parameters:
        parameters["shear_refresh"] = (
            SHEAR_JACOBIAN_REFRESH
        )

    cloth.setSimulatorParameters(
        **parameters
    )


def run_one_simulation(model):
    X0, T = createRectangularMesh(
        a=CLOTH_SIZE,
        b=CLOTH_SIZE,
        na=NODES_PER_SIDE,
        nb=NODES_PER_SIDE,
        h=INITIAL_CURVATURE,
    )

    X0[:, 2] += INITIAL_HEIGHT

    cloth = Cloth(
        X0.copy(),
        T.copy(),
        name=f"nonuniform_{model}",
        shear_model=model,
    )

    configure_cloth(
        cloth,
        model,
    )

    bottom_edge = np.arange(
        0,
        NODES_PER_SIDE,
        dtype=int,
    )

    top_edge = np.arange(
        (NODES_PER_SIDE - 1) * NODES_PER_SIDE,
        NODES_PER_SIDE * NODES_PER_SIDE,
        dtype=int,
    )

    controlled_nodes = np.concatenate(
        (
            bottom_edge,
            top_edge,
        )
    )

    control = controlled_nodes.tolist()

    initial_targets = X0[
        controlled_nodes
    ].copy()

    top_displacement = compact_top_profile(
        X0[top_edge, 0]
    )

    final_targets = initial_targets.copy()

    final_targets[
        NODES_PER_SIDE:,
        0,
    ] += top_displacement

    measured_shear_times = []

    # Smooth non-uniform loading.
    for frame in range(LOADING_FRAMES):
        loading_fraction = smoothstep(
            (frame + 1) / LOADING_FRAMES
        )

        targets = initial_targets.copy()

        targets[
            NODES_PER_SIDE:,
            0,
        ] += (
            loading_fraction
            * top_displacement
        )

        cloth.simulate(
            targets.copy(),
            control,
        )

        if not hasattr(
            cloth,
            "last_shear_time",
        ):
            raise AttributeError(
                "Cloth.simulate() must set cloth.last_shear_time."
            )

        if frame >= WARMUP_FRAMES:
            measured_shear_times.append(
                float(
                    cloth.last_shear_time
                )
            )

    # Untimed hold for a clean final configuration.
    for _ in range(HOLD_FRAMES):
        cloth.simulate(
            final_targets.copy(),
            control,
        )

    return {
        "model": model,
        "positions": cloth.positions.copy(),
        "reference": X0,
        "faces": T,
        "shear_times": np.asarray(
            measured_shear_times,
            dtype=float,
        ),
        "shear_constraints": int(
            sum(
                block.n_conds
                for block in cloth.shear
            )
        ),
        "jacobian_entries": int(
            sum(
                len(block.I)
                for block in cloth.shear
            )
        ),
        "rms_edge_strain": (
            rms_edge_strain(
                cloth.positions,
                X0,
                NODES_PER_SIDE,
            )
        ),
        "block_class": type(
            cloth.shear[0]
        ).__name__,
    }


# -----------------------------------------------------------------------------
# Run independent repetitions
# -----------------------------------------------------------------------------

configure_latex_plots()

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

all_runs = {
    "four_point": [],
    "diagonal": [],
}

for model in (
    "four_point",
    "diagonal",
):
    for repeat in range(REPEATS):
        print(
            f"Running {model}: "
            f"repeat {repeat + 1}/{REPEATS}"
        )

        all_runs[model].append(
            run_one_simulation(
                model
            )
        )


# -----------------------------------------------------------------------------
# Aggregate timing statistics
# -----------------------------------------------------------------------------

summary = {}

for model in (
    "four_point",
    "diagonal",
):
    runs = all_runs[model]

    # Use one median shear time per independent simulation.
    run_medians = np.asarray(
        [
            np.median(
                run["shear_times"]
            )
            for run in runs
        ],
        dtype=float,
    )

    summary[model] = {
        "median_time_ms": float(
            1000.0
            * np.median(
                run_medians
            )
        ),
        "q25_time_ms": float(
            1000.0
            * np.percentile(
                run_medians,
                25.0,
            )
        ),
        "q75_time_ms": float(
            1000.0
            * np.percentile(
                run_medians,
                75.0,
            )
        ),
        "constraints": runs[0][
            "shear_constraints"
        ],
        "jacobian_entries": runs[0][
            "jacobian_entries"
        ],
        "block_class": runs[0][
            "block_class"
        ],
        "edge_strain": float(
            np.median(
                [
                    run[
                        "rms_edge_strain"
                    ]
                    for run in runs
                ]
            )
        ),
    }


# Deterministic simulation: one run is sufficient for the deformation figure.
four_point = all_runs[
    "four_point"
][0]

diagonal = all_runs[
    "diagonal"
][0]

nodal_difference = np.linalg.norm(
    four_point["positions"]
    - diagonal["positions"],
    axis=1,
)

rms_position_difference = float(
    np.sqrt(
        np.mean(
            nodal_difference
            * nodal_difference
        )
    )
)

maximum_position_difference = float(
    np.max(
        nodal_difference
    )
)


# -----------------------------------------------------------------------------
# Figure 1: two deformation panels
# -----------------------------------------------------------------------------

figure, axes = plt.subplots(
    1,
    2,
    figsize=(7.2, 3.35),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)

all_positions = np.vstack(
    (
        four_point["positions"],
        diagonal["positions"],
    )
)

x_min = float(
    np.min(
        all_positions[:, 0]
    )
)

x_max = float(
    np.max(
        all_positions[:, 0]
    )
)

y_min = float(
    np.min(
        all_positions[:, 1]
    )
)

y_max = float(
    np.max(
        all_positions[:, 1]
    )
)

padding_x = 0.05 * max(
    x_max - x_min,
    CLOTH_SIZE,
)

padding_y = 0.05 * max(
    y_max - y_min,
    CLOTH_SIZE,
)


def draw_grid(
    axis,
    result,
    color,
    title,
):
    grid = result[
        "positions"
    ].reshape(
        NODES_PER_SIDE,
        NODES_PER_SIDE,
        3,
    )

    for row in range(
        NODES_PER_SIDE
    ):
        axis.plot(
            grid[row, :, 0],
            grid[row, :, 1],
            color=color,
            linewidth=0.75,
        )

    for column in range(
        NODES_PER_SIDE
    ):
        axis.plot(
            grid[:, column, 0],
            grid[:, column, 1],
            color=color,
            linewidth=0.75,
        )

    axis.set_xlim(
        x_min - padding_x,
        x_max + padding_x,
    )

    axis.set_ylim(
        y_min - padding_y,
        y_max + padding_y,
    )

    axis.set_aspect(
        "equal",
        adjustable="box",
    )

    axis.set_xlabel(
        r"$x$ [m]"
    )

    axis.set_ylabel(
        r"$y$ [m]"
    )

    axis.set_title(
        title
    )

    axis.grid(
        True,
        alpha=0.25,
    )


draw_grid(
    axes[0],
    four_point,
    "tab:blue",
    (
        r"(a) Four-point"
        "\n"
        rf"$\varepsilon_{{\mathrm{{edge}}}}"
        rf"={four_point['rms_edge_strain']:.3e}$"
    ),
)

draw_grid(
    axes[1],
    diagonal,
    "tab:orange",
    (
        r"(b) Diagonal"
        "\n"
        rf"$\varepsilon_{{\mathrm{{edge}}}}"
        rf"={diagonal['rms_edge_strain']:.3e}$"
    ),
)

figure.suptitle(
    (
        r"Non-uniform shear deformation at $N=21$"
        "\n"
        rf"RMS final configuration difference: "
        rf"${1000.0 * rms_position_difference:.2f}\,\mathrm{{mm}}$"
    ),
    fontsize=11,
)

save_figure(
    figure,
    "nonuniform_deformation_N21",
)


# -----------------------------------------------------------------------------
# Figure 2: separate shear-time graph
# -----------------------------------------------------------------------------

models = [
    "four_point",
    "diagonal",
]

labels = [
    r"Four-point",
    r"Diagonal",
]

colors = [
    "tab:blue",
    "tab:orange",
]

median_times = np.asarray(
    [
        summary[model][
            "median_time_ms"
        ]
        for model in models
    ],
    dtype=float,
)

lower_errors = median_times - np.asarray(
    [
        summary[model][
            "q25_time_ms"
        ]
        for model in models
    ],
    dtype=float,
)

upper_errors = np.asarray(
    [
        summary[model][
            "q75_time_ms"
        ]
        for model in models
    ],
    dtype=float,
) - median_times

figure, axis = plt.subplots(
    figsize=(3.5, 3.25),
    constrained_layout=True,
)

bars = axis.bar(
    np.arange(2),
    median_times,
    yerr=np.vstack(
        (
            lower_errors,
            upper_errors,
        )
    ),
    capsize=4,
    color=colors,
    width=0.62,
)

axis.set_xticks(
    np.arange(2),
    labels,
)

axis.set_ylabel(
    r"Median shear-projection time [ms/frame]"
)

axis.set_title(
    r"Shear-projection cost at $N=21$"
)

axis.grid(
    True,
    axis="y",
    alpha=0.3,
)

# Annotate the actual intrinsic equation counts above the bars.
vertical_offset = (
    0.04
    * max(
        np.max(
            median_times
            + upper_errors
        ),
        1.0e-9,
    )
)

for index, bar in enumerate(
    bars
):
    model = models[index]

    axis.text(
        bar.get_x()
        + bar.get_width() / 2.0,
        median_times[index]
        + upper_errors[index]
        + vertical_offset,
        (
            rf"$m="
            rf"{summary[model]['constraints']}"
            rf"$"
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )

axis.set_ylim(
    0.0,
    max(
        median_times
        + upper_errors
    )
    + 4.0
    * vertical_offset,
)

time_ratio = (
    summary["diagonal"][
        "median_time_ms"
    ]
    / summary["four_point"][
        "median_time_ms"
    ]
)

axis.text(
    0.5,
    0.96,
    (
        rf"$t_{{\mathrm{{diag}}}}/"
        rf"t_{{4p}}="
        rf"{time_ratio:.2f}$"
    ),
    transform=axis.transAxes,
    ha="center",
    va="top",
    fontsize=9,
)

save_figure(
    figure,
    "shear_projection_time_N21",
)


# -----------------------------------------------------------------------------
# Save compact numerical results
# -----------------------------------------------------------------------------

rows = []

for model in models:
    rows.append(
        {
            "model": model,
            "block_class": summary[
                model
            ][
                "block_class"
            ],
            "nodes_per_side": (
                NODES_PER_SIDE
            ),
            "shear_constraints": (
                summary[
                    model
                ][
                    "constraints"
                ]
            ),
            "jacobian_entries": (
                summary[
                    model
                ][
                    "jacobian_entries"
                ]
            ),
            "median_shear_time_ms": (
                summary[
                    model
                ][
                    "median_time_ms"
                ]
            ),
            "q25_shear_time_ms": (
                summary[
                    model
                ][
                    "q25_time_ms"
                ]
            ),
            "q75_shear_time_ms": (
                summary[
                    model
                ][
                    "q75_time_ms"
                ]
            ),
            "rms_edge_strain": (
                summary[
                    model
                ][
                    "edge_strain"
                ]
            ),
            "rms_model_difference_m": (
                rms_position_difference
            ),
            "maximum_model_difference_m": (
                maximum_position_difference
            ),
        }
    )

summary_path = (
    OUTPUT_DIRECTORY
    / "nonuniform_deformation_and_time_N21_summary.csv"
)

with summary_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=list(
            rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        rows
    )

np.savez_compressed(
    OUTPUT_DIRECTORY
    / "nonuniform_deformation_N21_states.npz",
    reference_positions=(
        four_point[
            "reference"
        ]
    ),
    faces=four_point[
        "faces"
    ],
    four_point_positions=(
        four_point[
            "positions"
        ]
    ),
    diagonal_positions=(
        diagonal[
            "positions"
        ]
    ),
)

print()
print(
    "RMS model difference: "
    f"{1000.0 * rms_position_difference:.3f} mm"
)

print(
    "Four-point shear time: "
    f"{summary['four_point']['median_time_ms']:.3f} ms/frame"
)

print(
    "Diagonal shear time: "
    f"{summary['diagonal']['median_time_ms']:.3f} ms/frame"
)

print(
    "Time ratio t_diag/t_4p: "
    f"{time_ratio:.3f}"
)

print(
    f"Results saved in: "
    f"{OUTPUT_DIRECTORY}"
)
