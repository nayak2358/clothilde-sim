# Gripper chooses the nodes

import sys, os
notebook_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(notebook_dir, '..'))
sys.path.append(parent_dir + "/python_code")

from implementation.Cloth import Cloth
from implementation.utils import createRectangularMesh
from implementation.Gripper_live import (
    SimulateGripper,
    quat_from_axis_angle,
    quat_to_rotmat,
    quat_transform_points
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
# mu_s=0.45, kappa=0.1 * 0.0001 
cloth.plotMesh()

# grip = SimulateGripper(cloth, grasp_radius=0.05, max_grasped_nodes=6)
grip = SimulateGripper(cloth, box_size=np.array([0.06, 0.08, 0.03], dtype=float), max_grasped_nodes=6)

import polyscope as ps
import polyscope.imgui as psim

smooth=2

# initial pose controls 
gripper_pos = cloth.positions.mean(axis=0).copy()
yaw = 0.0
jaw_open = True

# # optional simple random / manual motion parameters
# move_speed = 0.001
# rot_speed = 0.005

# Gripper parallelopiped
box_faces = np.array([
    [0,1,2], [0,2,3],
    [4,5,6], [4,6,7],
    [0,1,5], [0,5,4],
    [1,2,6], [1,6,5],
    [2,3,7], [2,7,6],
    [3,0,4], [3,4,7],
], dtype=int)

def get_box_vertices_world(p, q, box_size):
    hx, hy, hz = 0.5 * np.asarray(box_size, dtype=float)
    V_local = np.array([
        [-hx, -hy, -hz],
        [ hx, -hy, -hz],
        [ hx,  hy, -hz],
        [-hx,  hy, -hz],
        [-hx, -hy,  hz],
        [ hx, -hy,  hz],
        [ hx,  hy,  hz],
        [-hx,  hy,  hz],
    ], dtype=float)

    #V_local is (8, 3)
    return quat_transform_points(p, q, V_local)

green_open = [0.0, 1.0, 0.0]
red_closed = [1.0, 0.0, 0.0]

if cloth.polyscoped is False:
    cloth.preparePolyscope()

# register box once
V_box = get_box_vertices_world(grip.p, grip.q, grip.box_size)
ps_box = ps.register_surface_mesh("grasp_box", V_box, box_faces, color=green_open,
                                  transparency=0.35, material="wax")

def update_scene():
    # copied from Cloth.py: to update meshes
    phi_mat = cloth.positions
    phi_all = cloth.Am @ phi_mat
    for _ in range(smooth):
                phi_all = cloth.S @ phi_all

    ps.get_surface_mesh(cloth.label).update_vertex_positions(phi_all)
    ps.get_point_cloud(cloth.label).update_point_positions(phi_mat)

    V_box = get_box_vertices_world(grip.p, grip.q, grip.box_size)
    box = ps.get_surface_mesh("grasp_box")
    box.update_vertex_positions(V_box)

    if grip.is_open:   
        box.set_color(green_open)
    else:
        box.set_color(red_closed)

    # update gripper frame from current pose
    R = quat_to_rotmat(grip.q)
    try:
        ps.remove_structure("gripper_frame_live")
    except:
        pass

    origin = np.asarray(grip.p, dtype=float).reshape(1, 3)
    pc = ps.register_point_cloud("gripper_frame_live", origin, radius=0.005)
    pc.add_vector_quantity("x", (0.08 * R[:, 0]).reshape(1, 3), vectortype="ambient", enabled=True, color=[1.0, 0.0, 0.0])
    pc.add_vector_quantity("y", (0.08 * R[:, 1]).reshape(1, 3), vectortype="ambient", enabled=True, color=[0.0, 1.0, 0.0])
    pc.add_vector_quantity("z", (0.08 * R[:, 2]).reshape(1, 3), vectortype="ambient", enabled=True, color=[0.0, 0.0, 1.0])

def callback():
    global gripper_pos, yaw, jaw_open

    psim.TextUnformatted("Gripper control")

    changed, jaw_open = psim.Checkbox("Gripper open", jaw_open)

    _, gripper_pos[0] = psim.SliderFloat("px", float(gripper_pos[0]), -0.5, 0.5)
    _, gripper_pos[1] = psim.SliderFloat("py", float(gripper_pos[1]), -0.5, 0.5)
    _, gripper_pos[2] = psim.SliderFloat("pz", float(gripper_pos[2]),  0.0, 1.5)
    _, yaw = psim.SliderFloat("yaw", float(yaw), -np.pi, np.pi)

    q = quat_from_axis_angle([0.0, 0.0, 1.0], yaw)
    grip.set_pose(q, gripper_pos)

    # Edge-triggered grasp/release
    grip.set_open(jaw_open)

    psim.TextUnformatted(f"Grasped nodes = {grip.controlled}")

    # advance one physical step
    grip.step()

    update_scene()

ps.set_user_callback(callback)
ps.show()