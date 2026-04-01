# Cloth.py handles all the physics
# This file provides kinematic boundary condition on some grasped nodes

import numpy as np
from implementation.Cloth import Cloth

triangle_edges = np.array([
    [0,1], [1,2], [2,0]
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

def tri_aabb_overlaps_box(tri_pts_local, half):
    tri_pts_local = np.asarray(tri_pts_local, dtype=float).reshape(3, 3)
    half = np.asarray(half, dtype=float).reshape(3,)

    tri_min = tri_pts_local.min(axis=0)
    tri_max = tri_pts_local.max(axis=0)

    overlap_x = (tri_min[0] <= half[0]) and (tri_max[0] >= -half[0])
    overlap_y = (tri_min[1] <= half[1]) and (tri_max[1] >= -half[1])
    overlap_z = (tri_min[2] <= half[2]) and (tri_max[2] >= -half[2])

    return overlap_x and overlap_y and overlap_z

def tri_overlaps_box_by_sampling(tri_pts_local, half, n_samples=4):
    tri_pts_local = np.asarray(tri_pts_local, dtype=float).reshape(3, 3)
    half = np.asarray(half, dtype=float).reshape(3,)

    def inside_box(x):
        return (
            (abs(x[0]) <= half[0]) and
            (abs(x[1]) <= half[1]) and
            (abs(x[2]) <= half[2])
        )

    N = int(n_samples)
    for i in range(N + 1):
        for j in range(N + 1 - i):
            a = i / N
            b = j / N
            c = 1.0 - a - b
            x = a * tri_pts_local[0] + b * tri_pts_local[1] + c * tri_pts_local[2]
            if inside_box(x):
                return True

    return False

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

        # traingles to plot
        self.debug_hit_tri_local = np.zeros((0, 3), dtype=float)
        self.debug_hit_tri_edges = np.zeros((0, 2), dtype=int)

    def set_pose(self, q=None, p=None):
        if q is None:
            q = [1.0, 0.0, 0.0, 0.0]
        if p is None:
            p = [0.0, 0.0, 0.0]

        self.q = quat_normalize(q)
        self.p = np.array(p, dtype=float).reshape(3,)

    def find_nodes_in_vicinity(self, box=None, center_local=None):
        if box is None:
            box = self.box_size
        if center_local is None:
            center_local = np.zeros(3, dtype=float)

        box = np.asarray(box, dtype=float).reshape(3,)
        center_local = np.asarray(center_local, dtype=float).reshape(3,)
        half = 0.5 * box

        # transform all cloth vertices to gripper-local coords
        #  and shift so the grasp box center is at the local origin
        Xw_nodes = self.cloth.positions
        Xl_nodes = quat_inverse_transform_points(self.p, self.q, Xw_nodes)
        Xc_nodes = Xl_nodes - center_local.reshape(1, 3)

        # nodes inside the box
        inside_nodes = (
            (np.abs(Xc_nodes[:, 0]) <= half[0]) &
            (np.abs(Xc_nodes[:, 1]) <= half[1]) &
            (np.abs(Xc_nodes[:, 2]) <= half[2])
        )
        support = set(np.where(inside_nodes)[0].tolist())

        # -----------------------------------------
        # 4-triangle test per quad:
        #   (v0,v1,c), (v1,v2,c), (v2,v3,c), (v3,v0,c)
        # If a subtriangle overlaps the grasp box,
        # grasp the corresponding outer edge nodes.
        # -----------------------------------------
        F = self.cloth.faces          # (n_faces, 4)
        Xf = Xc_nodes[F]              # (n_faces, 4, 3), shifted local coords
        Xc_face = Xf.mean(axis=1)     # (n_faces, 3), quad center in shifted local coords

        candidate_faces = []
        candidate_edge_supports = []

        hit_tri_local = []
        hit_tri_edges = []
        hit_counter = 0

        for face_id in range(F.shape[0]):
            q0, q1, q2, q3 = Xf[face_id]
            c = Xc_face[face_id]

            # 4 subtriangles around quad center
            subtris = [
                (np.array([q0, q1, c], dtype=float), [F[face_id, 0], F[face_id, 1]]),
                (np.array([q1, q2, c], dtype=float), [F[face_id, 1], F[face_id, 2]]),
                (np.array([q2, q3, c], dtype=float), [F[face_id, 2], F[face_id, 3]]),
                (np.array([q3, q0, c], dtype=float), [F[face_id, 3], F[face_id, 0]]),
            ]

            edge_support = set()

            for tri_pts, edge_nodes in subtris:

                

                # # convert triangle to world for plotting
                # tri_world = quat_transform_points(
                #     np.asarray(self.p, dtype=float),
                #     quat_normalize(self.q),
                #     tri_local
                # )
                
                ## points on the traingle
                hit = tri_overlaps_box_by_sampling(tri_pts, half, n_samples=4)
                ## AABB of the triangle
                # hit = tri_aabb_overlaps_box(tri_pts, half)
                if hit:
                    edge_support.update(edge_nodes)   

                    # convert shifted-local triangle back to true gripper-local
                    tri_local = tri_pts + center_local.reshape(1, 3)             

                    # store only overlapping triangles separately
                    hit_tri_local.append(tri_local)
                    hit_tri_edges.append(np.array([
                        [3 * hit_counter + 0, 3 * hit_counter + 1],
                        [3 * hit_counter + 1, 3 * hit_counter + 2],
                        [3 * hit_counter + 2, 3 * hit_counter + 0],
                    ], dtype=int))
                    hit_counter += 1

            if len(edge_support) > 0:
                candidate_faces.append(face_id)
                candidate_edge_supports.append(sorted(edge_support))

        # among candidate quads only, choose the one whose center
        # is closest to the grasp-box center (which is local origin after shifting)
        if len(candidate_faces) > 0:
            cand = np.array(candidate_faces, dtype=int)
            d2 = np.sum(Xc_face[cand]**2, axis=1)
            k = int(np.argmin(d2))

            # grasp only the nodes belonging to overlapping outer edges
            support.update(candidate_edge_supports[k])

        if len(hit_tri_local) > 0:
            self.debug_hit_tri_local = np.vstack(hit_tri_local)
            self.debug_hit_tri_edges = np.vstack(hit_tri_edges)
        else:
            self.debug_hit_tri_local = np.zeros((0, 3), dtype=float)
            self.debug_hit_tri_edges = np.zeros((0, 2), dtype=int)

        return [int(i) for i in sorted(support)]  

    def get_debug_hit_triangles_world(self):
        if self.debug_hit_tri_local.shape[0] == 0:
            return (
                np.zeros((0, 3), dtype=float),
                np.zeros((0, 2), dtype=int)
            )

        pts_world = quat_transform_points(
            np.asarray(self.p, dtype=float),
            quat_normalize(self.q),
            self.debug_hit_tri_local
        )
        return pts_world, self.debug_hit_tri_edges           
    
    def set_open(self, is_open, box=None, center_local=None):

        is_open = bool(is_open)

        was_open = self.is_open
        self.is_open = is_open

        # open -> closed : attempt grasp
        if was_open and (not self.is_open):
            inds = self.find_nodes_in_vicinity(box=box, center_local=center_local)            
            # print(f'grasped nodes: {inds}') # only print when changing from open to closed

            if len(inds) > 0:
                self.controlled = inds
                Xw = self.cloth.positions[self.controlled].copy()
                self.local_points = quat_inverse_transform_points(self.p, self.q, Xw)
            else:
                self.controlled = []
                self.local_points = np.zeros((0, 3))

        # closed -> open : release
        elif (not was_open) and self.is_open:
            self.controlled = []
            self.local_points = np.zeros((0, 3))

            self.debug_hit_tri_local = np.zeros((0, 3), dtype=float)
            self.debug_hit_tri_edges = np.zeros((0, 2), dtype=int)

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
        