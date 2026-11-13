#!/usr/bin/env python3
"""
H2: mode selectivity of four-point and diagonal shear constraints.

Place in:
    python_code/examples/experiments/H2_mode_selectivity.py

Run from the repository root:
    python python_code/examples/experiments/H2_mode_selectivity.py

Assumption:
    Cloth(X, T, shear_model="four_point")
    Cloth(X, T, shear_model="diagonal")
"""

import csv
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
# Edit these settings
# -----------------------------------------------------------------------------

NODES_PER_SIDE = 9
CLOTH_SIZE = 1.0
INITIAL_CURVATURE = 0.001
OUTPUT_DIRECTORY = HERE.parent / "H2_results"


# -----------------------------------------------------------------------------
# LaTeX plot style
# -----------------------------------------------------------------------------

def configure_latex_plots():
    if shutil.which("latex") is None:
        raise RuntimeError(
            "A LaTeX installation is required to generate the figures. "
            "Install LaTeX and make sure the `latex` command is on PATH."
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
    figure.savefig(OUTPUT_DIRECTORY / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(
        OUTPUT_DIRECTORY / f"{name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


# -----------------------------------------------------------------------------
# Constraint evaluation
# -----------------------------------------------------------------------------

def apply_affine_deformation(reference_positions, matrix):
    positions = reference_positions.copy()
    positions[:, :2] = reference_positions[:, :2] @ matrix.T
    return positions


def evaluate_intrinsic_shear_constraints(cloth, positions):
    block = cloth.shear[0]

    phi = positions.reshape(
        3 * cloth.n_verts,
        order="F",
    )

    values = block.evaluate(
        phi,
        np.zeros(0),
        [],
        grad=False,
    )

    return np.asarray(
        values[: block.n_conds],
        dtype=float,
    )


def normalized_four_point_residual(cloth, reference_positions, residual):
    block = cloth.shear[0]

    node_0, node_1, node_2, node_3 = block.neighs.T

    tangent_1 = (
        reference_positions[node_1]
        - reference_positions[node_0]
    )

    tangent_2 = (
        reference_positions[node_3]
        - reference_positions[node_2]
    )

    scale = (
        np.linalg.norm(tangent_1, axis=1)
        * np.linalg.norm(tangent_2, axis=1)
    )

    return residual / np.maximum(scale, 1.0e-15)


def diagonal_channels(
    diagonal_cloth,
    reference_positions,
    faces,
    residual,
):
    number_of_faces = len(faces)

    if residual.size != 2 * number_of_faces:
        raise RuntimeError(
            "The diagonal shear block was expected to contain exactly "
            "two constraints per quadrilateral."
        )

    # The patched model builds all (0,2) diagonals first, followed by
    # all (1,3) diagonals.
    c_plus = residual[:number_of_faces]
    c_minus = residual[number_of_faces:]

    p0, p1, _, p3 = faces.T

    material_x = (
        reference_positions[p1]
        - reference_positions[p0]
    )

    material_y = (
        reference_positions[p3]
        - reference_positions[p0]
    )

    shear_scale = (
        np.linalg.norm(material_x, axis=1)
        * np.linalg.norm(material_y, axis=1)
    )

    axial_scale = (
        np.sum(material_x * material_x, axis=1)
        + np.sum(material_y * material_y, axis=1)
    )

    # Difference of the two diagonal equations extracts the mixed metric.
    shear_channel = (
        (c_plus - c_minus)
        / 4.0
        / np.maximum(shear_scale, 1.0e-15)
    )

    # Their sum contains a change in the two axial metric coefficients.
    axial_channel = (
        (c_plus + c_minus)
        / 2.0
        / np.maximum(axial_scale, 1.0e-15)
    )

    return shear_channel, axial_channel


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------

configure_latex_plots()
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

X0, T = createRectangularMesh(
    a=CLOTH_SIZE,
    b=CLOTH_SIZE,
    na=NODES_PER_SIDE,
    nb=NODES_PER_SIDE,
    h=INITIAL_CURVATURE,
)

X0[:, 2] += 1.0

four_point_cloth = Cloth(
    X0.copy(),
    T.copy(),
    shear_model="four_point",
)

diagonal_cloth = Cloth(
    X0.copy(),
    T.copy(),
    shear_model="diagonal",
)

rotation_angle = np.deg2rad(20.0)

modes = [
    (
        "Rigid rotation",
        r"Rigid rotation",
        np.array(
            [
                [np.cos(rotation_angle), -np.sin(rotation_angle)],
                [np.sin(rotation_angle), np.cos(rotation_angle)],
            ]
        ),
    ),
    (
        "10% stretch in x",
        r"$10\%$ stretch in $x$",
        np.array([[1.10, 0.0], [0.0, 1.0]]),
    ),
    (
        "10% stretch in y",
        r"$10\%$ stretch in $y$",
        np.array([[1.0, 0.0], [0.0, 1.10]]),
    ),
    (
        "Simple shear 0.10",
        r"Simple shear $\gamma=0.10$",
        np.array([[1.0, 0.10], [0.0, 1.0]]),
    ),
    (
        "Combined stretch and shear",
        r"Stretch and shear",
        np.array([[1.10, 0.10], [0.0, 1.0]]),
    ),
]

summary_rows = []

for plain_label, _, deformation_matrix in modes:
    positions = apply_affine_deformation(
        X0,
        deformation_matrix,
    )

    four_point_raw = evaluate_intrinsic_shear_constraints(
        four_point_cloth,
        positions,
    )

    diagonal_raw = evaluate_intrinsic_shear_constraints(
        diagonal_cloth,
        positions,
    )

    four_point_normalized = normalized_four_point_residual(
        four_point_cloth,
        X0,
        four_point_raw,
    )

    diagonal_shear, diagonal_axial = diagonal_channels(
        diagonal_cloth,
        X0,
        T,
        diagonal_raw,
    )

    summary_rows.append(
        {
            "mode": plain_label,
            "four_point_mixed_metric_rms": float(
                np.sqrt(
                    np.mean(
                        four_point_normalized
                        * four_point_normalized
                    )
                )
            ),
            "diagonal_shear_channel_rms": float(
                np.sqrt(
                    np.mean(
                        diagonal_shear
                        * diagonal_shear
                    )
                )
            ),
            "diagonal_axial_channel_rms": float(
                np.sqrt(
                    np.mean(
                        diagonal_axial
                        * diagonal_axial
                    )
                )
            ),
        }
    )

write_csv(
    OUTPUT_DIRECTORY / "H2_mode_selectivity.csv",
    summary_rows,
)

plot_labels = [mode[1] for mode in modes]
positions_on_axis = np.arange(len(modes))

bar_width = 0.24

four_point_values = [
    row["four_point_mixed_metric_rms"]
    for row in summary_rows
]

diagonal_shear_values = [
    row["diagonal_shear_channel_rms"]
    for row in summary_rows
]

diagonal_axial_values = [
    row["diagonal_axial_channel_rms"]
    for row in summary_rows
]


figure, axis = plt.subplots(
    figsize=(8.4, 4.8)
)


# Proposed four-point constraint
axis.bar(
    positions_on_axis - bar_width,
    four_point_values,
    bar_width,
    color="tab:blue",
    edgecolor="black",
    linewidth=0.6,
    label=r"Four-point mixed-metric response",
)


# Shear component reconstructed from the diagonal equations
axis.bar(
    positions_on_axis,
    diagonal_shear_values,
    bar_width,
    color="tab:orange",
    edgecolor="black",
    linewidth=0.6,
    label=r"Diagonal shear channel",
)


# Additional axial component contained in the diagonal equations
axis.bar(
    positions_on_axis + bar_width,
    diagonal_axial_values,
    bar_width,
    color="tab:red",
    edgecolor="black",
    linewidth=0.6,
    hatch="//",
    label=r"Diagonal axial channel",
)


axis.set_xticks(
    positions_on_axis,
    plot_labels,
    rotation=18,
    ha="right",
)

axis.set_ylabel(
    r"RMS normalized constraint response"
)

axis.set_title(
    r"Shear response and axial coupling under affine deformation"
)

axis.grid(
    True,
    axis="y",
    alpha=0.3,
)

axis.legend(
    loc="upper left",
)

figure.tight_layout()


save_figure(
    figure,
    "H2_shear_and_axial_response",
)

print(f"H2 results saved in: {OUTPUT_DIRECTORY}")
for row in summary_rows:
    print(row)
