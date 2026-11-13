import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
PYTHON_CODE = HERE.parents[2]

if str(PYTHON_CODE) not in sys.path:
    sys.path.insert(0, str(PYTHON_CODE))

from implementation.Cloth import Cloth
from implementation.utils import createRectangularMesh


# SHEAR_MODEL = "four_point"
SHEAR_MODEL = "diagonal"

FOUR_POINT_SHEAR_COMPLIANCE = 1.0e-3
TOP_EDGE_DISPLACEMENT = 0.2
FRAMES = 120

n = 13

X, T = createRectangularMesh(a=1, b=1, na=n, nb=n, h=0.001)
X[:, 2] += 1
# X += 0.0001 * np.random.randn(X.shape[0], 3)

cloth = Cloth(X, T, shear_model=SHEAR_MODEL)

if SHEAR_MODEL == "four_point":
    shear_compliance = FOUR_POINT_SHEAR_COMPLIANCE
else:
    shear_compliance = 0.5 * FOUR_POINT_SHEAR_COMPLIANCE
    
cloth.setSimulatorParameters(
    dt=1/60,
    sub_steps=8,
    
    # No gravity or aerodynamic loading
    delta=0.0, 
    
    # Damping is used only to reach equilibrium.
    alpha=1.0,     
    beta=0.0,
    
    # Isolate stretching and shearing
    kappa=0.0, 
    kappa_bnd=0.0,
    str=1e-6,
    shr=shear_compliance,
    
    # Contact and friction should remain inactive.
    mu_f=0.0,
    mu_s=0.0,
    
    eta_shr=0.0
    )

bottom_edge = np.arange(0, n, dtype=int)
top_edge = np.arange((n - 1) * n, n * n, dtype=int)

controlled_nodes = np.concatenate((bottom_edge, top_edge))
control = controlled_nodes.tolist()

initial_targets = X[controlled_nodes].copy()
# final_targets = initial_targets.copy()
# final_targets[n:, 0] += TOP_EDGE_DISPLACEMENT

def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return 3.0 * t**2 - 2.0 * t**3

for frame in range(FRAMES):
    tau = (frame + 1) / FRAMES
    fraction = smoothstep(tau)

    targets = initial_targets.copy()
    targets[n:, 0] += (fraction * TOP_EDGE_DISPLACEMENT)

    cloth.simulate(targets.copy(), control)

    if (
        frame == 0
        or (frame + 1) % 10 == 0
        or frame + 1 == FRAMES
    ):
        print(f"Loading frame {frame + 1}/{FRAMES}")

cloth.makeMovie(speed=1, repeat=True, smooth=0)


## Comparison of final configuration
# cloth_four = Cloth(X, T, shear_model="four_point")
# cloth_diag = Cloth(X, T, shear_model="diagonal")


 
