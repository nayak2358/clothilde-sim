# Gripper chooses the nodes where the grasp box overlaps with the 
# sampled points of the traingles and selects two nodes from the edge.

import sys, os
notebook_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(notebook_dir, '..'))
sys.path.append(parent_dir + "/python_code")

from implementation.Cloth2 import Cloth
from implementation.utils import createRectangularMesh
from implementation.Gripper_live_BB_triangle import (
    SimulateGripper,
    quat_from_axis_angle,
    quat_to_rotmat,
    quat_transform_points,
    quat_normalize,
    quat_from_rotvec,
)

import numpy as np
import trimesh

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
grip = SimulateGripper(cloth, box_size=np.array([0.06, 0.01, 0.015], dtype=float))

import polyscope as ps
import polyscope.imgui as psim

smooth=2

# initial pose controls 
gripper_pos = cloth.positions.mean(axis=0).copy()
jaw_open = True
jaw_gap_open = 0.0
jaw_gap_closed = -0.001 * (30 - 6)
# 30 mm - 6 mm (extension length) is the gap between the grippers in my CAD model
YELLOW = [186/256, 142/256, 35/256]

rotvec = np.array([0.0, 0.0, 0.0], dtype=float)

q0 = quat_from_axis_angle([1.0, 0.0, 0.0], 0.0)
grip.set_pose(q0, gripper_pos)
q0 = grip.q.copy()
p0 = grip.p.copy()


### helper functions

follow_camera_enabled = True
def follow_gripper_camera():
    cam_pos = grip.p + 3.0 * np.array([0.2, -0.2, 0.15])
    target = grip.p
    ps.look_at(cam_pos, target)

# Gripper parallelopiped
debug_box_faces = np.array([
    [0,1,2], [0,2,3],
    [4,5,6], [4,6,7],
    [0,1,5], [0,5,4],
    [1,2,6], [1,6,5],
    [2,3,7], [2,7,6],
    [3,0,4], [3,4,7],
], dtype=int)

def get_box_vertices_world_offset(p, q, box_size, center_local):
    hx, hy, hz = 0.5 * np.asarray(box_size, dtype=float)
    c = np.asarray(center_local, dtype=float).reshape(3,)

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

    V_local = V_local + c.reshape(1, 3)
    return quat_transform_points(np.asarray(p, dtype=float), quat_normalize(q), V_local)

def load_mesh(path):
    m = trimesh.load_mesh(path)
    return np.asarray(m.vertices), np.asarray(m.faces)

def transform_mesh(V_local, q, p, local_offset=np.zeros(3)):
    V = np.asarray(V_local, dtype=float) + np.asarray(local_offset, dtype=float).reshape(1, 3)
    return quat_transform_points(np.asarray(p, dtype=float), quat_normalize(q), V)

###

# load 3 parts
V_base0, F_base = load_mesh("abhilash/cad_files/base.stl")
V_left0, F_left = load_mesh("abhilash/cad_files/jaw_left_2.stl")
V_right0, F_right = load_mesh("abhilash/cad_files/jaw_right_2.stl")

mesh_scale = 0.001   # change to 0.001 if CAD comes in mm
V_base0 *= mesh_scale
V_left0 *= mesh_scale
V_right0 *= mesh_scale

if cloth.polyscoped is False:
    cloth.preparePolyscope()

ps.register_surface_mesh(
    "gripper_base",
    transform_mesh(V_base0, q0, p0),
    F_base, color=YELLOW
)
ps.register_surface_mesh(
    "gripper_left",
    transform_mesh(V_left0, q0, p0, local_offset=np.array([-jaw_gap_open/2, 0, 0])),
    F_left, color=YELLOW
)
ps.register_surface_mesh(
    "gripper_right",
    transform_mesh(V_right0, q0, p0, local_offset=np.array([ jaw_gap_open/2, 0, 0])),
    F_right, color=YELLOW
)

# grasp box
grasp_box = 0.001 * np.array([6, 30, 6], dtype=float)

tip_center_local = 0.001 * np.array([0.0, 0.0, -49], dtype=float)
# 37 - 3 + 15 mm (check CAD file)  is the distance of the jaw extensions 
# from the origin of the gripper frame

V_dbg = get_box_vertices_world_offset(grip.p, grip.q, grasp_box, tip_center_local)

ps.register_surface_mesh(
    "grasp_box",
    V_dbg,
    debug_box_faces,
    color=[1.0, 1.0, 0.0],
    transparency=0.55,
    material="wax"
)

# pts_bbox, edges_bbox, candidate_faces = grip.get_candidate_face_bbox_curve_network(
#     grip.p, grip.q, grasp_box, tip_center_local, cloth
# )

# ps.register_curve_network(
#     "candidate_face_bboxes",
#     pts_bbox,
#     edges_bbox,
#     radius=0.0005,
#     color=[0.0, 1.0, 1.0]
# )

## initialize triangles

pts_tri, edges_tri = grip.get_debug_hit_triangles_world()

ps.register_curve_network(
    "hit_subtriangles",
    pts_tri,
    edges_tri,
    radius=0.0010,
    color=[0.0, 1.0, 1.0]
)

#region update scene

def update_scene():
    # copied from Cloth.py: to update meshes
    phi_mat = cloth.positions
    phi_all = cloth.Am @ phi_mat
    for _ in range(smooth):
                phi_all = cloth.S @ phi_all

    ps.get_surface_mesh(cloth.label).update_vertex_positions(phi_all)
    ps.get_point_cloud(cloth.label).update_point_positions(phi_mat)

    q = grip.q.copy()
    p = grip.p.copy()

    current_gap = jaw_gap_open if jaw_open else jaw_gap_closed


    ps.get_surface_mesh("gripper_base").update_vertex_positions(
        transform_mesh(V_base0, q, p)
    )
    ps.get_surface_mesh("gripper_left").update_vertex_positions(
        transform_mesh(V_left0, q, p, local_offset=np.array([-current_gap/2, 0, 0]))
    )
    ps.get_surface_mesh("gripper_right").update_vertex_positions(
        transform_mesh(V_right0, q, p, local_offset=np.array([ current_gap/2, 0, 0]))
    )

    # update grasp box
    V_dbg = get_box_vertices_world_offset(grip.p, grip.q, grasp_box, tip_center_local)
    ps.get_surface_mesh("grasp_box").update_vertex_positions(V_dbg)

    if jaw_open:
        ps.get_surface_mesh("grasp_box").set_color([0.0, 1.0, 0.0])
    else:
        ps.get_surface_mesh("grasp_box").set_color([1.0, 0.0, 0.0])

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
    
    ## to plot the axis of rotation
    # angle = np.linalg.norm(rotvec)
    # if angle < 1e-12:
    #     axis_vis = np.array([1.0, 0.0, 0.0], dtype=float)
    # else:
    #     axis_vis = rotvec / angle
    # pc.add_vector_quantity("rot_axis", (0.08 * axis_vis).reshape(1, 3), vectortype="ambient", enabled=True, color=[1.0, 1.0, 1.0])

    ## to plot the AABB of quads

    # pts_bbox, edges_bbox, candidate_faces = grip.get_candidate_face_bbox_curve_network(
    # grip.p, grip.q, grasp_box, tip_center_local, cloth)

    # try:
    #     ps.remove_structure("candidate_face_bboxes")
    # except:
    #     pass

    # ps.register_curve_network(
    #     "candidate_face_bboxes",
    #     pts_bbox,
    #     edges_bbox,
    #     radius=0.0005,
    #     color=[0.0, 1.0, 1.0]
    # )

    ## Plotting grasped triangles

    pts_tri, edges_tri = grip.get_debug_hit_triangles_world()

    try:
        ps.remove_structure("hit_subtriangles")
    except:
        pass

    ps.register_curve_network(
        "hit_subtriangles",
        pts_tri,
        edges_tri,
        radius=0.0010,
        color=[0.0, 1.0, 1.0]
    )

#region callback 

def callback():
    global gripper_pos, rotvec, jaw_open, jaw_gap_open, jaw_gap_closed, tip_center_local, grasp_box
    
    psim.TextUnformatted("Gripper control")

    _, jaw_open = psim.Checkbox("Gripper open", jaw_open)

    _, gripper_pos[0] = psim.SliderFloat("px", float(gripper_pos[0]), -0.5, 0.5)
    _, gripper_pos[1] = psim.SliderFloat("py", float(gripper_pos[1]), -0.5, 0.5)
    _, gripper_pos[2] = psim.SliderFloat("pz", float(gripper_pos[2]),  0.0, 1.5)
    _, rotvec[0] = psim.SliderFloat("rx", float(rotvec[0]), -np.pi, np.pi)
    _, rotvec[1] = psim.SliderFloat("ry", float(rotvec[1]), -np.pi, np.pi)
    _, rotvec[2] = psim.SliderFloat("rz", float(rotvec[2]), -np.pi, np.pi)

    q = quat_from_rotvec(rotvec)

    grip.set_pose(q, gripper_pos)

    # Edge-triggered grasp/release
    # current_gap = jaw_gap_open if jaw_open else jaw_gap_closed
    grasp_box = 0.001 * np.array([3, 10, 6], dtype=float)
    # grasp_box = 0.001 * np.array([60, 60, 30], dtype=float)

    tip_center_local = 0.001 * np.array([0.0, 0.0, -49], dtype=float)

    grip.set_open(jaw_open, box=grasp_box, center_local=tip_center_local)

    psim.TextUnformatted(f"Grasped nodes = {grip.controlled}")

    # advance one physical step
    grip.step()

    psim.TextUnformatted("Camera Controls")

    _, follow_camera_enabled = psim.Checkbox(
        "Follow Camera",
        follow_camera_enabled
    )

    if follow_camera_enabled:
         follow_gripper_camera()

    update_scene()

ps.set_user_callback(callback)
ps.show()