import sys, os
notebook_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(notebook_dir, '..'))
sys.path.append(parent_dir + "/python_code")

from implementation.Cloth import Cloth
from implementation.utils import createRectangularMesh
from implementation.Gripper_dq import (
    SimulateGripper,
    quat_from_axis_angle,
    dq_from_rt,
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
cloth.setSimulatorParameters(dt=dt)
cloth.plotMesh()

grip = SimulateGripper(cloth)

tf = int(6 / dt)
grasp_nodes = [0, 1]

# Initial pose at selected nodes
q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
p0 = cloth.positions[grasp_nodes].mean(axis=0)

dq0 = dq_from_rt(q0, p0)
grip.set_pose(dq=dq0)
grip.select_nodes(grasp_nodes)

t_start = 2.0
t_rot = 4.0
rot_angle = np.pi

axis_dir = np.array([0.0, 0.0, 1.0], dtype=float)
axis_dir /= np.linalg.norm(axis_dir)

# Fixed rotation center in world frame
axis_center = np.array([0.0, 0.0, 0.0], dtype=float)

rotation_started = False

for i in tqdm(range(tf), desc="Simulating", unit="step"):
    t = i * dt

    if t < t_start:  # let the cloth fall first
        grip.step()
        continue

    if not rotation_started:
        # Re-anchor the selected cloth nodes in the frame whose origin is axis_center
        q_start = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        dq_start = dq_from_rt(q_start, axis_center)

        grip.set_pose(dq=dq_start) # frame is the world frame moved to axis center
        grip.select_nodes(grasp_nodes) # nodes in the previous axis-centered frame

        rotation_started = True
    
    alpha = np.clip((t - t_start) / t_rot, 0.0, 2.0) # rotate for 2.0 seconds
    ### Smooth rotation around a fixed center
    q = quat_from_axis_angle(axis_dir, rot_angle * alpha) # pure rotation about the axis center
    dq = dq_from_rt(q, axis_center)

    ### Translation along an axis
    # q = [1, 0, 0, 0]
    # t = np.array([0, 0, alpha/4])
    # dq = dq_from_rt(q, t)

    # ### Screw motion
    # theta = rot_angle * alpha
    # h = 0.2 # pitch
    # q = quat_from_axis_angle(axis_dir, theta)
    # p = axis_center + h * theta * axis_dir
    # dq = dq_from_rt(q, p)

    grip.set_pose(dq=dq)
    grip.step() # bringing the points back in the world frame

axis_len = 2.0
axis_points = np.array([
    axis_center - 0.5 * axis_len * axis_dir,
    axis_center + 0.5 * axis_len * axis_dir
])

traj_mean = np.array([phi[grasp_nodes].mean(axis=0) for phi in cloth.history_pos])

grasp_history = np.array([phi[grasp_nodes] for phi in cloth.history_pos])
# shape: (n_frames, n_grasp_nodes, 3)

cloth.makeMovie(
    speed=6,
    repeat=True,
    smooth=2,
    trace_points=[traj_mean],
    axis_points=axis_points,
    world_frame=True,
    gripper_frame=True,
    grasp_history=grasp_history
)

# cloth.saveFrames(speed=8)