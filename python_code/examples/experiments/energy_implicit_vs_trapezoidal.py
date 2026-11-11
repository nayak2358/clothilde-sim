"""
Unconstrained energy evolution E^n / E^0 for four unconstrained integrations:
    1. Trapezoidal rule, D = 0
    2. Implicit Euler, D = 0
    3. Trapezoidal rule, D != 0
    4. Implicit Euler, D != 0

Only the existing unconstrained integrators in Cloth.py are used. No
stretching, shearing, collision, contact, or post-projection velocity update
is applied. This is a numerical test of Eqs. (2)--(7), not a complete cloth
simulation.

Expected behavior:
    - Trapezoidal, D = 0: E^n / E^0 remains equal to 1 up to roundoff.
    - Trapezoidal, D != 0: energy decreases monotonically.
    - Implicit Euler, D = 0: energy decreases because of numerical damping.
    - Implicit Euler, D != 0: energy decreases because of physical and
      numerical damping.

Recommended Cloth.py assembly, consistent with the paper:
    M = self.rho * self.M
    K = self.kappa * self.K + self.kappa_bnd * self.Kb
    D = self.alpha * M + self.beta * K

Output:
    python_code/examples/energy_results/
        unconstrained_energy_evolution_four_cases.pdf
        unconstrained_energy_evolution_four_cases.png
"""

from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

HERE = Path(__file__).resolve()
PYTHON_CODE = HERE.parents[2]  # stored in python_code/examples/energy_results

if str(PYTHON_CODE) not in sys.path:
    sys.path.insert(0, str(PYTHON_CODE))

from implementation.Cloth import Cloth
from implementation.utils import createRectangularMesh


# -----------------------------------------------------------------------------
# Experiment settings
# -----------------------------------------------------------------------------

FRAME_DT = 1.0 / 60.0
SUB_STEPS = 8
DURATION = 5.0
NODES_PER_SIDE = 20
INITIAL_SPEED = 0.10  # m/s

# Plot one sample per simulation frame. All substeps are still evaluated and
# used in the numerical checks below.
PLOT_STRIDE = SUB_STEPS

# Damped cases. With D = alpha M + beta K, this gives mass-proportional
# Rayleigh damping. Increase DAMPED_ALPHA for a stronger visible decay.
DAMPED_ALPHA = 0.20
DAMPED_BETA = 0.0

BASE_PARAMETERS = dict(
    dt=FRAME_DT,
    sub_steps=SUB_STEPS,
    tol=0.0075,
    rho=0.1,
    delta=0.0,       # no gravity
    kappa=0.5e-4,
    kappa_bnd=0.05e-4,
    str=0.01e-4,     # unused: no projection step is called
    shr=10e-4,       # unused
    slf=1e-4,        # unused
    mu_f=0.0,        # unused
    mu_s=0.0,        # unused
    thck=0.95,       # unused
    max_mov=0.1,     # unused
)


# -----------------------------------------------------------------------------
# Data container
# -----------------------------------------------------------------------------
# Generates a class without writing the __init__() method
@dataclass
class CaseResult:
    label: str
    time: np.ndarray
    normalized_energy: np.ndarray
    normalized_identity_residual: np.ndarray


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def create_case_cloth(
    initial_positions: np.ndarray,
    faces: np.ndarray,
    *,
    implicit_euler: bool,
    alpha: float,
    beta: float,
):
    """Create one Cloth object and return its physical matrices."""

    cloth = Cloth(initial_positions.copy(), faces.copy())

    # Cache the unscaled operators before setSimulatorParameters().
    M0 = cloth.M.copy()
    K0 = cloth.K.copy()
    Kb0 = cloth.Kb.copy()
    Fg0 = cloth.Fg.copy()

    cloth.setSimulatorParameters(
        **BASE_PARAMETERS,
        alpha=alpha,
        beta=beta,
    )
    cloth.implicitEuler = implicit_euler

    # Operators appearing in Eqs. (2) and (6).
    M = BASE_PARAMETERS["rho"] * M0
    K = (
        BASE_PARAMETERS["kappa"] * K0
        + BASE_PARAMETERS["kappa_bnd"] * Kb0
    )
    f_g = BASE_PARAMETERS["delta"] * cloth.g * Fg0.toarray()

    # Use the damping matrix actually assembled by Cloth.py so that the
    # diagnostic identity matches the implemented predictor exactly.
    if not hasattr(cloth, "D"):
        raise AttributeError(
            "Cloth.py must retain self.D for this experiment. "
            "Do not delete the diagnostic matrices."
        )
    D = cloth.D.copy()

    return cloth, M, K, f_g, D


def mechanical_energy(
    positions: np.ndarray,
    velocities: np.ndarray,
    M,
    K,
    f_g: np.ndarray,
) -> float:
    """Mechanical energy E from Eq. (6)."""

    kinetic = 0.5 * np.sum(velocities * (M @ velocities))
    bending = 0.5 * np.sum(positions * (K @ positions))
    gravity = -np.sum(f_g * positions)

    return float(kinetic + bending + gravity)


def apply_initial_velocity(cloth: Cloth, M) -> None:
    """Apply the same smooth out-of-plane velocity to every case."""

    x = cloth.positions[:, 0]
    y = cloth.positions[:, 1]

    mode = np.sin(2.0 * np.pi * x) * np.cos(np.pi * y)

    # Remove any residual mass-weighted rigid translation.
    masses = np.asarray(M.diagonal()).ravel()
    mode -= np.sum(masses * mode) / np.sum(masses)

    velocity = np.zeros_like(cloth.positions)
    velocity[:, 2] = INITIAL_SPEED * mode

    cloth.velocities = np.asfortranarray(velocity)


def run_case(
    initial_positions: np.ndarray,
    faces: np.ndarray,
    *,
    label: str,
    implicit_euler: bool,
    alpha: float,
    beta: float,
) -> CaseResult:
    """Chain only the unconstrained state for one integration method."""

    cloth, M, K, f_g, D = create_case_cloth(
        initial_positions,
        faces,
        implicit_euler=implicit_euler,
        alpha=alpha,
        beta=beta,
    )

    apply_initial_velocity(cloth, M)

    E0 = mechanical_energy(
        cloth.positions,
        cloth.velocities,
        M,
        K,
        f_g,
    )

    if not np.isfinite(E0) or E0 <= 0.0:
        raise RuntimeError(
            f"The initial mechanical energy must be positive; got {E0}."
        )

    h = cloth.dt
    number_of_steps = int(round(DURATION / h))

    times = np.arange(number_of_steps + 1, dtype=float) * h
    energies = np.empty(number_of_steps + 1, dtype=float)
    residuals = np.empty(number_of_steps, dtype=float)

    energies[0] = E0

    for step in range(number_of_steps):
        x_n = cloth.positions.copy(order="F")
        v_n = cloth.velocities.copy(order="F")
        E_n = energies[step]

        # Existing Cloth.py implementation.
        x_next_flat = cloth.unconstrainedStep(implicit_euler)
        x_next = np.asarray(x_next_flat).reshape(
            (cloth.n_verts, 3),
            order="F",
        )

        if implicit_euler:
            # x^{n+1} - x^n = h v^{n+1}
            v_next = (x_next - x_n) / h
        else:
            # Eq. (4): x^{n+1} - x^n = h/2 (v^{n+1} + v^n)
            v_next = 2.0 * (x_next - x_n) / h - v_n

        E_next = mechanical_energy(x_next, v_next, M, K, f_g)
        delta_E = E_next - E_n
        energies[step + 1] = E_next

        # Check the discrete energy identity. These residuals should remain
        # near numerical precision for all four cases.
        if implicit_euler:
            dx = x_next - x_n
            dv = v_next - v_n

            physical_damping = h * np.sum(v_next * (D @ v_next))
            numerical_dissipation = 0.5 * (
                np.sum(dv * (M @ dv))
                + np.sum(dx * (K @ dx))
            )

            residual = (
                delta_E
                + physical_damping
                + numerical_dissipation
            )
        else:
            v_mid = 0.5 * (v_next + v_n)
            physical_damping = h * np.sum(v_mid * (D @ v_mid))

            # Eq. (7): delta_E + h v_bar^T D v_bar = 0.
            residual = delta_E + physical_damping

        residuals[step] = residual / E0

        # The unconstrained candidate becomes the next state.
        cloth.positions = np.asfortranarray(x_next)
        cloth.velocities = np.asfortranarray(v_next)

    normalized_energy = energies / E0

    return CaseResult(
        label=label,
        time=times,
        normalized_energy=normalized_energy,
        normalized_identity_residual=residuals,
    )


def plot_indices(number_of_samples: int, stride: int) -> np.ndarray:
    """Return sparse plotting indices while always retaining the last sample."""

    indices = np.arange(0, number_of_samples, stride, dtype=int)
    if indices[-1] != number_of_samples - 1:
        indices = np.append(indices, number_of_samples - 1)
    return indices


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    X, T = createRectangularMesh(
        a=1.0,
        b=1.0,
        na=NODES_PER_SIDE,
        nb=NODES_PER_SIDE,
        h=0.01,
    )

    # Translation does not change energy because gravity is disabled.
    X[:, 2] += 1.0

    cases = [
        run_case(
            X,
            T,
            label=r"Trapezoidal, $D=0$",
            implicit_euler=False,
            alpha=0.0,
            beta=0.0,
        ),
        run_case(
            X,
            T,
            label=r"Implicit Euler, $D=0$",
            implicit_euler=True,
            alpha=0.0,
            beta=0.0,
        ),
        run_case(
            X,
            T,
            label=r"Trapezoidal, $D\neq0$",
            implicit_euler=False,
            alpha=DAMPED_ALPHA,
            beta=DAMPED_BETA,
        ),
        run_case(
            X,
            T,
            label=r"Implicit Euler, $D\neq0$",
            implicit_euler=True,
            alpha=DAMPED_ALPHA,
            beta=DAMPED_BETA,
        ),
    ]

    output_dir = HERE.parent / "energy_results"
    output_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for index, case in enumerate(cases):
        indices = plot_indices(len(case.time), PLOT_STRIDE)

        # Solid lines: D=0. Dashed lines: D!=0.
        linestyle = "-" if index < 2 else "--"

        ax.plot(
            case.time[indices],
            case.normalized_energy[indices],
            label=case.label,
            linestyle=linestyle,
            linewidth=1.25,
        )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Normalized mechanical energy $E^n/E^0$")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Keep zero outside the plotted range. This also prevents a zero tick from
    # appearing on the vertical axis.
    minimum_energy = min(
        float(np.min(case.normalized_energy))
        for case in cases
    )
    energy_range = max(1.0 - minimum_energy, 1e-3)
    lower_limit = max(minimum_energy - 0.05 * energy_range, 1e-6)
    ax.set_ylim(lower_limit, 1.01)

    fig.tight_layout()

    png_path = output_dir / "unconstrained_energy_evolution_four_cases.png"
    pdf_path = output_dir / "unconstrained_energy_evolution_four_cases.pdf"

    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)

    print()
    print("Unconstrained energy-evolution experiment")
    print("-----------------------------------------")
    print(f"Frame step Delta_t_f = {FRAME_DT:.8f} s")
    print(f"Substeps S            = {SUB_STEPS}")
    print(f"Internal step h       = {FRAME_DT / SUB_STEPS:.8f} s")
    print(
        "Damped case parameters: "
        f"alpha={DAMPED_ALPHA}, beta={DAMPED_BETA}"
    )
    print()

    for case in cases:
        increments = np.diff(case.normalized_energy)
        maximum_increase = float(np.max(increments))
        maximum_identity_residual = float(
            np.max(np.abs(case.normalized_identity_residual))
        )

        print(case.label.replace("$", ""))
        print(f"  final E/E0                    = {case.normalized_energy[-1]:.6f}")
        print(f"  max positive energy increment = {maximum_increase:.3e}")
        print(f"  max identity residual / E0    = {maximum_identity_residual:.3e}")

    print()
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

    plt.show()


if __name__ == "__main__":
    main()
