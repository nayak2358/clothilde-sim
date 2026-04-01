# Using two grippers: Vertical lift

import sys, os
notebook_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(notebook_dir, '..'))
sys.path.append(parent_dir + "/python_code")

from implementation.Cloth import Cloth
from implementation.utils import createRectangularMesh
from implementation.Gripper import SimulateGripper

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
cloth.setSimulatorParameters(dt=dt)
cloth.plotMesh()

grip1 = SimulateGripper(cloth)
grip2 = SimulateGripper(cloth)

grasp_nodes_set1 = [0]
grasp_nodes_set2 = [na - 1]

t_fall = 2.0
t_lift = 1.0
tf = int((t_fall + t_lift) / dt)

lift_height = 0.25
q_id = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

p_pick1 = None
p_lift1 = None
p_pick2 = None
p_lift2 = None

grasp_started = False

for i in tqdm(range(tf), desc="Simulating", unit="step"):
    t = i * dt

    # free fall
    if t < t_fall:
        cloth.simulate(u=np.zeros((0, 3)), control=[])
        grip1.record_history()
        grip2.record_history()
        continue

    # grasp once
    if not grasp_started:
        p_pick1 = cloth.positions[grasp_nodes_set1].mean(axis=0).copy()
        p_lift1 = p_pick1 + np.array([0.0, 0.0, lift_height], dtype=float)
        grip1.set_pose(q_id, p_pick1)
        grip1.select_nodes(grasp_nodes_set1)

        p_pick2 = cloth.positions[grasp_nodes_set2].mean(axis=0).copy()
        p_lift2 = p_pick2 + np.array([0.0, 0.0, lift_height], dtype=float)
        grip2.set_pose(q_id, p_pick2)
        grip2.select_nodes(grasp_nodes_set2)

        grasp_started = True

    # lift both independently
    if t < t_fall + t_lift:
        alpha = np.clip((t - t_fall) / t_lift, 0.0, 1.0)

        p1 = p_pick1 + alpha * (p_lift1 - p_pick1)
        p2 = p_pick2 + alpha * (p_lift2 - p_pick2)

        grip1.set_pose(q_id, p1)
        grip2.set_pose(q_id, p2)

        u1 = grip1.current_targets()
        c1 = grip1.controlled

        u2 = grip2.current_targets()
        c2 = grip2.controlled

        u = np.vstack([u1, u2])
        control = c1 + c2

        cloth.simulate(u=u, control=control)
        grip1.record_history()
        grip2.record_history()
        continue

traj_mean1 = np.array([phi[grasp_nodes_set1].mean(axis=0) for phi in cloth.history_pos])
traj_mean2 = np.array([phi[grasp_nodes_set2].mean(axis=0) for phi in cloth.history_pos])

cloth.makeMovie(
    speed=6,
    repeat=True,
    smooth=2,
    trace_points=[traj_mean1, traj_mean2],
    axis_points=None,
    world_frame=True,
    gripper_origin_histories=[grip1.origin_history, grip2.origin_history],
    gripper_R_histories=[grip1.R_history, grip2.R_history],
    gripper_visible_histories=[grip1.visible_history, grip2.visible_history]
)