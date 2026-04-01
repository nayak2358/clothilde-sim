import sys, os
notebook_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(notebook_dir, '..'))
sys.path.append(parent_dir + "/python_code")

from implementation.Cloth import Cloth
from implementation.utils import createRectangularMesh
from implementation.Gripper import (
    SimulateGripper,
    quat_from_axis_angle,
    quat_rotate_vector,
)

import numpy as np
from tqdm import tqdm

na = 20
nb = 30
np.random.seed(1)

X, T = createRectangularMesh(a=0.5, b=0.8, na=na, nb=nb, h=0.2)
X[:, 2] += 0.7
X += 0.0001 * np.random.randn(X.shape[0], 3)

cloth = Cloth(X, T)
dt = cloth.estimateTimeStep(L=0.8)
cloth.setSimulatorParameters(dt=dt, mu_s=0.45, kappa=0.1 * 0.0001)
cloth.plotMesh()

grip = SimulateGripper(cloth)

grasp_nodes = [0, 1, 20]
# depending on the dimension, pose and opening/closing of the gripper

t_fall = 2.0
t_lift = 1.0
t_spin = 2.5
t_release = 2.0
tf = int((t_fall + t_lift + t_spin + t_release) / dt)

lift_height = 0.25
spin_angle = 0.9 * np.pi

q_id = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
p_pick = None
p_lift = None
grasp_started = False

rotation_started = False
released = False

for i in tqdm(range(tf), desc="Simulating", unit="step"):
    t = i * dt

    # Free fall
    if t < t_fall:
        grip.free_step()
        continue

    # Grasp once, after free fall
    if not grasp_started:
        p_pick = cloth.positions[grasp_nodes].mean(axis=0).copy()

        grip.set_pose(q_id, p_pick)
        grip.select_nodes(grasp_nodes)

        p_lift = p_pick + np.array([0.0, 0.0, lift_height], dtype=float)
        grasp_started = True

    # Lift
    if t < t_fall + t_lift:
        alpha = np.clip((t - t_fall) / t_lift, 0.0, 1.0) # lift for t_lift * 1.0 seconds
        p = p_pick + alpha * (p_lift - p_pick)

        grip.set_pose(q_id, p)
        grip.step()
        continue

    # Rotate 
    if t < t_fall + t_lift + t_spin:

        beta = np.clip((t - (t_fall + t_lift)) / t_spin, 0.0, 1.0)

        axis_arbitrary = np.array([0.0, 0.0, 1.0], dtype=float)
        axis_arbitrary /= np.linalg.norm(axis_arbitrary)

        q_spin = quat_from_axis_angle(axis_arbitrary, spin_angle * beta)

        # rotating about a different point than the gripper origin
        c = p_lift + np.array([0.05, 0.1, 0.0])
        R_c = quat_rotate_vector(q_spin, p_lift - c)
        p = c + R_c

        grip.set_pose(q_spin, p)
        grip.step()
        continue

    # Release
    if not released:
        grip.clear_nodes()
    
    grip.free_step()

# traj_mean = np.array([phi[grasp_nodes].mean(axis=0) for phi in cloth.history_pos])

trace_grip = np.array(grip.origin_history, dtype=float)

first_vis = next(i for i, v in enumerate(grip.visible_history) if v)
trace_grip[:first_vis] = trace_grip[first_vis]

cloth.makeMovie(
    speed=6,
    repeat=True,
    smooth=2,
    trace_points=[trace_grip],
    axis_points=None,
    world_frame=True,
    gripper_origin_histories=[grip.origin_history],
    gripper_R_histories=[grip.R_history],
    gripper_visible_histories=[grip.visible_history]
)

# cloth.saveFrames(speed=8)