# Cloth.py handles all the physics
# This file provides kinematic boundary condition on some grasped nodes

import numpy as np
from implementation.Cloth import Cloth

bbox_edges = np.array([
    [0,1], [1,2], [2,3], [3,0],   # bottom
    [4,5], [5,6], [6,7], [7,4],   # top
    [0,4], [1,5], [2,6], [3,7],   # verticals
], dtype=int)

def make_aabb_vertices_local(face_min, face_max):
    xmin, ymin, zmin = face_min
    xmax, ymax, zmax = face_max

    return np.array([
        [xmin, ymin, zmin],
        [xmax, ymin, zmin],
        [xmax, ymax, zmin],
        [xmin, ymax, zmin],
        [xmin, ymin, zmax],
        [xmax, ymin, zmax],
        [xmax, ymax, zmax],
        [xmin, ymax, zmax],
    ], dtype=float)

# =========================
# Quaternion utilities
# =========================
def quat_normalize(q):
    q = np.array(q, dtype=float).reshape(4,)
    return q / (np.linalg.norm(q) + 1e-12)


def quat_conjugate(q):
    q = np.array(q, dtype=float).reshape(4,)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_mul(q1, q2):
    q1 = np.array(q1, dtype=float).reshape(4,)
    q2 = np.array(q2, dtype=float).reshape(4,)
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)


def quat_from_axis_angle(axis, angle):
    axis = np.array(axis, dtype=float).reshape(3,)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    half = 0.5 * angle
    s = np.sin(half)
    return quat_normalize(
        np.array([np.cos(half), axis[0]*s, axis[1]*s, axis[2]*s], dtype=float)
    )

# rotation: q * vec in quat form * q^{-1}
def quat_rotate_vector(q, v):
    q = quat_normalize(q)
    vq = np.array([0.0, v[0], v[1], v[2]], dtype=float)
    return quat_mul(quat_mul(q, vq), quat_conjugate(q))[1:]

def quat_rotate_points(q, X):
    X = np.array(X, dtype=float)
    return np.vstack([quat_rotate_vector(q, x) for x in X])

def quat_transform_points(p, q, X):
    X = np.array(X, dtype=float)
    return np.vstack([
        quat_rotate_vector(q, x) + p
        for x in X
    ])

def quat_inverse_transform_points(p, q, X):
    X = np.array(X, dtype=float)
    return np.vstack([
        quat_rotate_vector(quat_conjugate(q), x - p)
        for x in X
    ])

def quat_to_rotmat(q):
    q = quat_normalize(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ], dtype=float)

def quat_from_rotvec(rotvec):
    rotvec = np.asarray(rotvec, dtype=float).reshape(3,)
    angle = np.linalg.norm(rotvec)
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    axis = rotvec / angle
    return quat_from_axis_angle(axis, angle)

class SimulateGripper:

    def __init__(self, cloth: Cloth, box_size=np.array([0.03, 0.06, 0.02], dtype=float)):

        self.cloth: Cloth = cloth

        self.controlled = [] # grasped nodes
        self.local_points = [] # grasped points in local gripper frame

        # pose
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.p = np.array([0.0, 0.0, 0.0], dtype=float)

        ## Parallelopiped
        # dimensions of grasp region in gripper local frame
        # x = jaw opening direction
        # y = finger thickness direction
        # z = finger length / approach direction
        self.box_size = box_size
        
        self.is_open = True

        # So that the gripper is only shown when it grasps something
        self.origin_history = [np.array([0.0, 0.0, 0.0], dtype=float)]
        self.R_history = [np.eye(3)]
        self.visible_history = [True]
        self.grasp_history = [False]

    def set_pose(self, q=None, p=None):
        if q is None:
            q = [1.0, 0.0, 0.0, 0.0]
        if p is None:
            p = [0.0, 0.0, 0.0]

        self.q = quat_normalize(q)
        self.p = np.array(p, dtype=float).reshape(3,)

## quad center in grasp box OR quad center near to grasp box origin

    # def find_nodes_in_vicinity(self, smooth=2, box=None, center_local=None):
    #     if box is None:
    #         box = self.box_size
    #     if center_local is None:
    #         center_local = np.zeros(3, dtype=float)

    #     box = np.asarray(box, dtype=float).reshape(3,)
    #     center_local = np.asarray(center_local, dtype=float).reshape(3,)
    #     half = 0.5 * box

    #     # nodes: Take the smoothed ones rather than the actual ones
    #     phi_mat = self.cloth.positions
    #     phi_all = self.cloth.Am @ phi_mat
    #     for _ in range(smooth):
    #         phi_all = self.cloth.S @ phi_all

    #     n_nodes = self.cloth.positions.shape[0]
    #     Xw_nodes = phi_all[:n_nodes]

    #     # Xw_nodes = self.cloth.positions
    #     Xl_nodes = quat_inverse_transform_points(self.p, self.q, Xw_nodes)
    #     Xc_nodes = Xl_nodes - center_local.reshape(1, 3)

    #     # inside nodes is an array of booleans [False, True, False, ...]
    #     inside_nodes = (
    #         (np.abs(Xc_nodes[:, 0]) <= half[0]) &
    #         (np.abs(Xc_nodes[:, 1]) <= half[1]) &
    #         (np.abs(Xc_nodes[:, 2]) <= half[2])
    #     )

    #     support = set(np.where(inside_nodes)[0].tolist())

    #     # Xw_face_centers = 0.25 * (self.cloth.A2 @ self.cloth.positions)
    #     Xw_face_centers = phi_all[n_nodes:]

    # ## only accept if the quad center lies in the grasp box

    #     # # quad centers in local frame
    #     # Xl_face_centers = quat_inverse_transform_points(self.p, self.q, Xw_face_centers)
    #     # Xc_face_centers = Xl_face_centers - center_local.reshape(1, 3)

    #     # inside_face_nodes = (
    #     #         (np.abs(Xc_face_centers[:, 0]) <= half[0]) &
    #     #         (np.abs(Xc_face_centers[:, 1]) <= half[1]) &
    #     #         (np.abs(Xc_face_centers[:, 2]) <= half[2])
    #     #     )
        
    #     # support_face_nodes = np.where(inside_face_nodes)[0]

    #     # if support_face_nodes.size > 0:
    #     #     # add individual nodes as int
    #     #     support.update(
    #     #         int(i) for i in np.unique(self.cloth.faces[support_face_nodes].reshape(-1))
    #     #     )

    # ## only accept this nearest quad if it is reasonably close to the grasp point
    # ## using the half-diagonal of the grasp box as threshold

    #     grasp_point_world = quat_transform_points(
    #         self.p,
    #         self.q,
    #         center_local.reshape(1, 3)
    #     )[0]

    #     d2 = np.sum((Xw_face_centers - grasp_point_world.reshape(1, 3))**2, axis=1)
    #     face_id = int(np.argmin(d2))
    #     dist = np.sqrt(d2[face_id])

    #     max_dist = np.linalg.norm(half)

    #     if dist <= max_dist:
    #         support.update(self.cloth.faces[face_id].tolist())

    #     return sorted(support)

## overlap between grasp box and AABB of quads
    def find_nodes_in_vicinity(self, smooth=2, box=None, center_local=None):
        if box is None:
            box = self.box_size
        if center_local is None:
            center_local = np.zeros(3, dtype=float)

        box = np.asarray(box, dtype=float).reshape(3,)
        center_local = np.asarray(center_local, dtype=float).reshape(3,)
        half = 0.5 * box

        # transform all cloth vertices to gripper-local coords
        #  and shift so the grasp box center is at the local origin
        phi_mat = self.cloth.positions
        phi_all = self.cloth.Am @ phi_mat
        for _ in range(smooth):
            phi_all = self.cloth.S @ phi_all

        n_nodes = self.cloth.positions.shape[0]
        Xw_nodes = phi_all[:n_nodes]
        Xl_nodes = quat_inverse_transform_points(self.p, self.q, Xw_nodes)
        Xc_nodes = Xl_nodes - center_local.reshape(1, 3)

        # nodes inside the box
        inside_nodes = (
            (np.abs(Xc_nodes[:, 0]) <= half[0]) &
            (np.abs(Xc_nodes[:, 1]) <= half[1]) &
            (np.abs(Xc_nodes[:, 2]) <= half[2])
        )
        support = set(np.where(inside_nodes)[0].tolist())

        # build a bounding box around the quadrilateral 
        # using the minimum and maximum of x, y, z values
        F = self.cloth.faces                    # (n_faces, 4) since ech face has 4 vertices
        Xf = Xc_nodes[F]                        # (n_faces, 4, 3)

        face_min = Xf.min(axis=1)               # (n_faces, 3)
        face_max = Xf.max(axis=1)               # (n_faces, 3)

        # checking overlapping in each axes, and for the 3D boxes to overlap, 
        # every axes must overlap.
        overlaps = (
            (face_min[:, 0] <=  half[0]) & (face_max[:, 0] >= -half[0]) &
            (face_min[:, 1] <=  half[1]) & (face_max[:, 1] >= -half[1]) &
            (face_min[:, 2] <=  half[2]) & (face_max[:, 2] >= -half[2])
        )

        candidate_faces = np.where(overlaps)[0]

        # among overlapping candidates only, pick the quadrilateral whose center
        # is closest to the grasp-box center
        if candidate_faces.size > 0:
            face_centers_local = Xf.mean(axis=1)   # (n_faces, 3)
            d2 = np.sum(face_centers_local[candidate_faces]**2, axis=1)
            face_id = int(candidate_faces[np.argmin(d2)])

            # add the 4 corners of that quad
            support.update(F[face_id].tolist())

        return sorted(support)
    
    def get_candidate_face_bbox_curve_network(self, p, q, smooth, box_size, center_local, cloth):
        box_size = np.asarray(box_size, dtype=float).reshape(3,)
        center_local = np.asarray(center_local, dtype=float).reshape(3,)
        half = 0.5 * box_size

        # world -> gripper local
        phi_mat = self.cloth.positions
        phi_all = self.cloth.Am @ phi_mat
        for _ in range(smooth):
            phi_all = self.cloth.S @ phi_all

        n_nodes = self.cloth.positions.shape[0]
        Xw_nodes = phi_all[:n_nodes]
        Xl_nodes = quat_inverse_transform_points(p, q, Xw_nodes)

        F = cloth.faces                 # (n_faces, 4)
        Xf_local = Xl_nodes[F]          # (n_faces, 4, 3)

        # shifted local coords, so grasp box center is at origin
        Xf_shift = Xf_local - center_local.reshape(1, 1, 3)

        face_min_shift = Xf_shift.min(axis=1)
        face_max_shift = Xf_shift.max(axis=1)

        overlaps = (
            (face_min_shift[:, 0] <=  half[0]) & (face_max_shift[:, 0] >= -half[0]) &
            (face_min_shift[:, 1] <=  half[1]) & (face_max_shift[:, 1] >= -half[1]) &
            (face_min_shift[:, 2] <=  half[2]) & (face_max_shift[:, 2] >= -half[2])
        )

        candidate_faces = np.where(overlaps)[0]

        if candidate_faces.size == 0:
            return np.zeros((0, 3)), np.zeros((0, 2), dtype=int), candidate_faces

        all_pts = []
        all_edges = []

        for k, face_id in enumerate(candidate_faces):
            # bbox in true gripper-local frame, not shifted
            face_pts_local = Xf_local[face_id]
            face_min_local = face_pts_local.min(axis=0)
            face_max_local = face_pts_local.max(axis=0)

            V_local = make_aabb_vertices_local(face_min_local, face_max_local)
            V_world = quat_transform_points(
                np.asarray(p, dtype=float),
                quat_normalize(q),
                V_local
            )

            base = 8 * k
            E = bbox_edges + base

            all_pts.append(V_world)
            all_edges.append(E)

        all_pts = np.vstack(all_pts)
        all_edges = np.vstack(all_edges)

        return all_pts, all_edges, candidate_faces
    
    def set_open(self, is_open, smooth, box=None, center_local=None):

        is_open = bool(is_open)

        was_open = self.is_open
        self.is_open = is_open

        # open -> closed : attempt grasp
        if was_open and (not self.is_open):
            inds = self.find_nodes_in_vicinity(box=box, smooth=smooth, center_local=center_local)            
            # print(f'grasped nodes: {inds}') # only print when changing from open to closed

            if len(inds) > 0:
                self.controlled = inds
                Xw = self.cloth.positions[self.controlled].copy()

                # self.local_points = quat_inverse_transform_points(self.p, self.q, Xw)

                # the node is selected using the grasp-box center, but then it is attached using its 
                # original local position relative to the gripper frame, so it keeps that offset when lifted; 
                # the fix is to change rendering only in the zone of grasping.
                Xl = quat_inverse_transform_points(self.p, self.q, Xw)

                ## making the centroid of the quad move towards the grasp box center
                ## squeezing mainly in x-direction because that's the direction of pinching
                ## Xl is in the gripper frame, so it always happens wrt to the x-axis 
                ## of the gripper
                if Xl.shape[0] == 1:
                    Xl[0, 0] = center_local[0]
                    Xl[0, 2] = center_local[2] + 0.001
                    # Xl = center_local
                else:
                    # align patch centroid with grasp center in x 
                    # (this makes the new mean = center_local[0])
                    Xl[:, 0] += center_local[0] - Xl[:, 0].mean()
                    # squeeze patch inward around its centroid in x
                    beta = 0.7   # 0 < beta < 1 ; smaller = stronger squeeze
                    cx = Xl[:, 0].mean()
                    Xl[:, 0] = cx + beta * (Xl[:, 0] - cx)

                    Xl[:, 2] += 0.001

                self.local_points = Xl

            else:
                self.controlled = []
                self.local_points = np.zeros((0, 3))

        # closed -> open : release
        elif (not was_open) and self.is_open:
            self.controlled = []
            self.local_points = np.zeros((0, 3))

    def record_history(self):
        R = quat_to_rotmat(self.q)
        grasp_now = (not self.is_open) and (len(self.controlled) > 0)

        self.origin_history.append(self.p.copy())
        self.R_history.append(R.copy())
        self.visible_history.append(True)
        self.grasp_history.append(grasp_now)
                
    def step(self):
        if (not self.is_open) and len(self.controlled) > 0:
            u = quat_transform_points(self.p, self.q, self.local_points)
            self.cloth.simulate(u=u, control=self.controlled)
        else:
            self.cloth.simulate(u=np.zeros((0, 3)), control=[])

        self.record_history()

"""
The best long-term formulation is:

detect an arbitrary grasp point in world space
convert it to a material point on the cloth element
store:
face index
bilinear coordinates (xi, eta)
gripper-local coordinates of that material point
at each step, enforce that material point to follow the gripper
distribute that constraint to the 4 corner vertices using the quad shape functions

This is fully aligned with your quad finite-element cloth model, 
which already uses quad reference-element shape functions internally.
"""
        