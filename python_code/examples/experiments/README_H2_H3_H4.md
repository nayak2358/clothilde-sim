# Clothilde Experiment B — H2, H3 and H4

These three scripts assume that your local `python_code/implementation/Cloth.py`
already implements the model selector:

```python
Cloth(X, T, shear_model="four_point")
Cloth(X, T, shear_model="diagonal")
```

They use the actual constraint blocks created by `Cloth.py`. H2 evaluates the
constraint equations directly; H3 and H4 call `Cloth.simulate()`.

## Repository placement

Copy the files into:

```text
python_code/examples/experiments/
    H2_diagonal_mode_selectivity.py
    H3_resolution_convergence.py
    H4_constraint_cost.py
```

Run each script from the repository root, using the same Conda environment in
which Clothilde works.

## H2 — Diagonal constraints: shear/axial coupling

Run:

```bash
python python_code/examples/experiments/H2_diagonal_mode_selectivity.py
```

This is an affine, constraint-level test. It applies rigid rotation, pure
x-stretch, pure y-stretch, simple shear, and combined shear/stretch to the
same initially flat quadrilateral grid. It evaluates the actual shear blocks
created by the two Clothilde models.

For each quadrilateral, define the two diagonal residuals:

```text
Cplus  = ||p2-p0||^2 - ||p2_0-p0_0||^2
Cminus = ||p3-p1||^2 - ||p3_0-p1_0||^2
```

For a parallelogram with material edge vectors `a` and `b`:

```text
(Cplus - Cminus)/4 = change in dot(a,b)
(Cplus + Cminus)/2 = change in (||a||^2 + ||b||^2)
```

The difference is the shear channel; the sum is the axial-coupling channel.
The output is diagnostic of the equations themselves, not of the final
equilibrium after stretching and shearing constraints have been solved
together.

Outputs are written to `H2_results/` beside the script:

- `H2_mode_selectivity.csv`
- `H2_diagonal_channels.csv`
- `H2_mode_selectivity.pdf/.png`
- `H2_diagonal_channels.pdf/.png`

Interpretation: pure axial stretch should leave the four-point mixed-metric
residual near zero on this initially orthogonal grid, while changing both
diagonal lengths. The diagonal residual pair therefore responds to axial
deformation as well as shear. This does not by itself prove that the complete
solver will exhibit a larger final axial strain: the common stretching block
also acts on the solution.

## H3 — Coarse-mesh consistency

Run:

```bash
python python_code/examples/experiments/H3_resolution_convergence.py
```

For a short installation/syntax smoke test:

```bash
python python_code/examples/experiments/H3_resolution_convergence.py --quick
```

The script uses the actual Clothilde solver. It fixes the lower boundary and
prescribes a smooth sinusoidal horizontal displacement on the upper boundary.
Gravity, bending, aerodynamics and friction are disabled. It runs both models
on several resolutions.

The material compliance is scaled with the squared reference edge spacing:

```text
compliance(h) = compliance(h_ref) * (h/h_ref)^2
```

The diagonal shear compliance is initialized at half the four-point value,
which matches the small-strain energy for a regular square/parallelogram
element. This is an initial calibration, not a universal equivalence; a
compliance sensitivity study is recommended before making a strong claim.

Each model is compared against its own finest tested mesh. The script reports:

- relative L2 position error after interpolation to a common material grid;
- RMS and maximum normalized mixed-metric change;
- RMS and maximum material-edge strain;
- final velocity;
- solver iterations and timing diagnostics.

Outputs are written to `H3_results/`.

Important: the finest tested resolution is a numerical reference, not physical
ground truth. H3 is supported only if the four-point model converges more
consistently at coarse resolutions without obtaining that result by allowing
more edge stretch. If it does not, report that outcome rather than claiming
superiority.

## H4 — Constraint count and computational cost

Run:

```bash
python python_code/examples/experiments/H4_constraint_cost.py
```

Short smoke test:

```bash
python python_code/examples/experiments/H4_constraint_cost.py --quick
```

By default, each resolution/model pair is run five times (change with
`--repeats N`). Each trial creates a fresh cloth, performs a cold first frame,
then measures a periodic prescribed top-edge motion for a sequence of frames.

The script reports:

- number of intrinsic shear equations (`n_conds`);
- intrinsic Jacobian structural entries (`len(block.I)`);
- CHOLMOD factor nonzeros if exposed by the installed scikit-sparse version;
- cold first-frame time;
- median and 95th-percentile steady frame time;
- mean and maximum solver sweeps.

Outputs are written to `H4_results/`.

Fewer equations do not automatically mean lower runtime. A four-point row has
more Jacobian entries than a diagonal distance row, and sparse factorization
fill-in can dominate. Use measured time and solver iterations, not equation
count alone, for computational conclusions. For publication, run on an idle
machine and report median and interquartile range across repeated trials.

## General controls and reproducibility

All scripts use:

- the same regular quadrilateral mesh convention `[bottom-left, bottom-right,
  top-right, top-left]`;
- the same physical dimensions;
- the same fixed lower edge and prescribed upper edge where dynamics are used;
- the same frame time, substeps, solver tolerance and material density;
- no gravity, bending, table, floor contact or friction in H3/H4.

The diagonal model adds constraint pairs only. Do not add diagonal pairs to
`self.edges_matrix`, because that would alter mesh topology, collision
adjacency, boundary detection and other parts of the simulator.

Record the exact Git commit, Python/NumPy/SciPy/scikit-sparse versions, CPU,
operating system and all simulation parameters with any results used in the
paper.

## Scientific limits

- H2 demonstrates the algebraic shear/axial decomposition of diagonal
  residuals. It is not a full dynamic comparison.
- H3 tests numerical discretization consistency against a finest-mesh
  reference. It does not establish physical accuracy.
- H4 measures implementation cost on the tested machine and settings. It does
  not establish a universal complexity advantage.
- H1 is not repeated here: its affine mixed-metric identity is derived in the
  paper and explained separately.
