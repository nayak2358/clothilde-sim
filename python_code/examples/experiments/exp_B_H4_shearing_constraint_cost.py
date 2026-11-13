#!/usr/bin/env python3
"""
H4: shear-block size and measured computational cost.

Place in:
    python_code/examples/experiments/H4_constraint_cost.py

Run from the repository root:
    python python_code/examples/experiments/H4_constraint_cost.py

Set QUICK_TEST = True below for a short installation test.
"""

import csv
import shutil
import sys
import time
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
# Edit these settings
# -----------------------------------------------------------------------------

QUICK_TEST = False

CLOTH_SIZE = 1.0
INITIAL_CURVATURE = 0.001
INITIAL_HEIGHT = 1.0

FRAME_DT = 1.0 / 60.0
SUB_STEPS = 8
SOLVER_TOLERANCE = 0.0075

REFERENCE_RESOLUTION = 21
REFERENCE_EDGE_LENGTH = (
    CLOTH_SIZE / (REFERENCE_RESOLUTION - 1)
)

REFERENCE_STRETCH_COMPLIANCE = 1.0e-6
REFERENCE_FOUR_POINT_COMPLIANCE = 1.0e-3
DIAGONAL_COMPLIANCE_FACTOR = 0.5

CONTROL_AMPLITUDE = 0.01

OUTPUT_DIRECTORY = HERE.parent / "H4_results"

if QUICK_TEST:
    RESOLUTIONS = (7, 9, 13)
    REPEATS = 1
    WARMUP_FRAMES = 3
    MEASURED_FRAMES = 20
else:
    RESOLUTIONS = (9, 13, 21, 29)
    REPEATS = 5
    WARMUP_FRAMES = 5
    MEASURED_FRAMES = 120


# -----------------------------------------------------------------------------
# LaTeX plot style
# -----------------------------------------------------------------------------

def configure_latex_plots():
    if shutil.which("latex") is None:
        raise RuntimeError(
            "A LaTeX installation is required to generate the figures."
        )

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{amsmath}",
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def save_figure(figure, name):
    figure.savefig(
        OUTPUT_DIRECTORY / f"{name}.pdf",
        bbox_inches="tight",
    )

    figure.savefig(
        OUTPUT_DIRECTORY / f"{name}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# -----------------------------------------------------------------------------
# One timing trial
# -----------------------------------------------------------------------------

def cholmod_factor_nnz(block):
    factor = getattr(
        block,
        "factor",
        None,
    )

    if factor is None:
        return np.nan

    try:
        return float(
            factor.L().nnz
        )
    except Exception:
        return np.nan


def run_trial(n, model, repeat_index):
    X0, T = createRectangularMesh(
        a=CLOTH_SIZE,
        b=CLOTH_SIZE,
        na=n,
        nb=n,
        h=INITIAL_CURVATURE,
    )

    X0[:, 2] += INITIAL_HEIGHT

    cloth = Cloth(
        X0.copy(),
        T.copy(),
        name=f"H4_{model}_{n}_{repeat_index}",
        shear_model=model,
    )

    edge_length = CLOTH_SIZE / (n - 1)

    compliance_scale = (
        edge_length / REFERENCE_EDGE_LENGTH
    ) ** 2

    stretch_compliance = (
        REFERENCE_STRETCH_COMPLIANCE
        * compliance_scale
    )

    shear_compliance = (
        REFERENCE_FOUR_POINT_COMPLIANCE
        * compliance_scale
    )

    if model == "diagonal":
        shear_compliance *= (
            DIAGONAL_COMPLIANCE_FACTOR
        )

    cloth.setSimulatorParameters(
        dt=FRAME_DT,
        sub_steps=SUB_STEPS,
        tol=SOLVER_TOLERANCE,
        rho=0.1,
        delta=0.0,
        alpha=0.5,
        beta=0.0,
        kappa=0.0,
        kappa_bnd=0.0,
        str=stretch_compliance,
        shr=shear_compliance,
        slf=1.0e-4,
        mu_f=0.0,
        mu_s=0.0,
        thck=0.95,
        max_mov=0.25,
    )

    bottom_edge = np.arange(0, n, dtype=int)

    top_edge = np.arange(
        (n - 1) * n,
        n * n,
        dtype=int,
    )

    controlled_nodes = np.concatenate(
        (bottom_edge, top_edge)
    )

    control = controlled_nodes.tolist()

    base_targets = X0[
        controlled_nodes
    ].copy()

    def target_for_frame(frame_number):
        targets = base_targets.copy()

        phase = (
            2.0
            * np.pi
            * frame_number
            / MEASURED_FRAMES
        )

        targets[n:, 0] += (
            CONTROL_AMPLITUDE
            * np.sin(phase)
        )

        return targets

    # The first call includes creation/factorization of the control-dependent
    # constraint matrices.
    first_target = target_for_frame(0)

    iteration_count_before = (
        cloth.total_iters
    )

    start = time.perf_counter()

    cloth.simulate(
        first_target.copy(),
        control,
    )

    cold_frame_time = (
        time.perf_counter() - start
    )

    cold_solver_sweeps = (
        cloth.total_iters
        - iteration_count_before
    )

    # Warm-up frames are excluded from timing.
    for frame in range(1, WARMUP_FRAMES + 1):
        cloth.simulate(
            target_for_frame(frame).copy(),
            control,
        )

    frame_times = []
    solver_sweeps = []

    for frame in range(MEASURED_FRAMES):
        target = target_for_frame(
            frame + WARMUP_FRAMES + 1
        )

        iteration_count_before = (
            cloth.total_iters
        )

        start = time.perf_counter()

        cloth.simulate(
            target.copy(),
            control,
        )

        frame_times.append(
            time.perf_counter() - start
        )

        solver_sweeps.append(
            cloth.total_iters
            - iteration_count_before
        )

    shear_equations = int(
        sum(
            block.n_conds
            for block in cloth.shear
        )
    )

    jacobian_entries = int(
        sum(
            len(block.I)
            for block in cloth.shear
        )
    )

    factor_nonzeros = [
        cholmod_factor_nnz(block)
        for block in cloth.shear
    ]

    if np.all(np.isnan(factor_nonzeros)):
        factor_nnz_sum = np.nan
    else:
        factor_nnz_sum = float(
            np.nansum(factor_nonzeros)
        )

    return {
        "model": model,
        "n": n,
        "repeat": repeat_index,
        "vertices": n * n,
        "quadrilaterals": (n - 1) ** 2,
        "shear_equations": shear_equations,
        "intrinsic_jacobian_entries": (
            jacobian_entries
        ),
        "factor_L_nonzeros": factor_nnz_sum,
        "cold_frame_time_s": (
            cold_frame_time
        ),
        "cold_solver_sweeps": (
            cold_solver_sweeps
        ),
        "median_frame_time_s": float(
            np.median(frame_times)
        ),
        "p95_frame_time_s": float(
            np.percentile(
                frame_times,
                95.0,
            )
        ),
        "mean_solver_sweeps": float(
            np.mean(solver_sweeps)
        ),
        "maximum_solver_sweeps": float(
            np.max(solver_sweeps)
        ),
    }


def write_csv(path, rows):
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def median_and_iqr(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    if np.all(np.isnan(values)):
        return np.nan, np.nan

    median = float(
        np.nanmedian(values)
    )

    iqr = float(
        np.nanpercentile(values, 75.0)
        - np.nanpercentile(values, 25.0)
    )

    return median, iqr


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------

configure_latex_plots()
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

trials = []

for n in RESOLUTIONS:
    for model in ("four_point", "diagonal"):
        for repeat_index in range(REPEATS):
            print(
                f"Running H4: "
                f"model={model}, n={n}, "
                f"repeat={repeat_index + 1}/{REPEATS}"
            )

            trials.append(
                run_trial(
                    n,
                    model,
                    repeat_index,
                )
            )

write_csv(
    OUTPUT_DIRECTORY / "H4_all_trials.csv",
    trials,
)

summary = []

metrics = (
    "shear_equations",
    "intrinsic_jacobian_entries",
    "factor_L_nonzeros",
    "cold_frame_time_s",
    "median_frame_time_s",
    "p95_frame_time_s",
    "mean_solver_sweeps",
    "maximum_solver_sweeps",
)

for n in RESOLUTIONS:
    for model in ("four_point", "diagonal"):
        selected_trials = [
            trial
            for trial in trials
            if (
                trial["n"] == n
                and trial["model"] == model
            )
        ]

        row = {
            "model": model,
            "n": n,
        }

        for metric in metrics:
            median, iqr = median_and_iqr(
                [
                    trial[metric]
                    for trial in selected_trials
                ]
            )

            row[metric] = median
            row[f"{metric}_IQR"] = iqr

        summary.append(row)

write_csv(
    OUTPUT_DIRECTORY / "H4_summary.csv",
    summary,
)

plot_definitions = (
    (
        "shear_equations",
        r"Number of intrinsic shear equations",
        "H4_constraint_count",
    ),
    (
        "intrinsic_jacobian_entries",
        r"Intrinsic Jacobian entries",
        "H4_jacobian_entries",
    ),
    (
        "median_frame_time_s",
        r"Median frame time [s]",
        "H4_median_runtime",
    ),
    (
        "p95_frame_time_s",
        r"95th-percentile frame time [s]",
        "H4_p95_runtime",
    ),
    (
        "mean_solver_sweeps",
        r"Mean solver sweeps per frame",
        "H4_solver_sweeps",
    ),
)

for metric, vertical_label, filename in plot_definitions:
    figure, axis = plt.subplots(
        figsize=(6.4, 4.5)
    )

    for model, label in (
        ("four_point", r"Four-point"),
        ("diagonal", r"Diagonal"),
    ):
        rows = sorted(
            [
                row
                for row in summary
                if row["model"] == model
            ],
            key=lambda row: row["n"],
        )

        axis.errorbar(
            [row["n"] for row in rows],
            [row[metric] for row in rows],
            yerr=[
                row[f"{metric}_IQR"]
                for row in rows
            ],
            marker="o",
            capsize=3,
            label=label,
        )

    axis.set_xlabel(
        r"Vertices per side $N$"
    )

    axis.set_ylabel(vertical_label)

    axis.set_title(vertical_label)

    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    save_figure(
        figure,
        filename,
    )

print(f"H4 results saved in: {OUTPUT_DIRECTORY}")
