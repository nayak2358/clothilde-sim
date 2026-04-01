# Cloth.py handles all the physics
# This file provides kinematic boundary condition on some grasped nodes

import numpy as np
from implementation.Cloth import Cloth

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

class SimulateGripper:

    def __init__(self, cloth: Cloth, grasp_radius=0.05, max_grasped_nodes=6):
        self.cloth: Cloth = cloth

        # controlled node indices
        self.controlled = []

        # selected cloth points in local gripper frame
        self.local_points = np.zeros((0, 3))

        # pose
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.p = np.array([0.0, 0.0, 0.0], dtype=float)

        # Gripper open or close
        self.is_closed = False
        self.grasp_radius = grasp_radius
        self.max_grasped_nodes = max_grasped_nodes
        

        # So that the gripper is only shown when it grasps something
        self.origin_history = [np.array([0.0, 0.0, 0.0], dtype=float)]
        self.R_history = [np.eye(3)]
        self.visible_history = [False]
        self.grasp_history = [False]

    def set_pose(self, q=None, p=None):
        if q is None:
            q = [1.0, 0.0, 0.0, 0.0]
        if p is None:
            p = [0.0, 0.0, 0.0]

        self.q = quat_normalize(q)
        self.p = np.array(p, dtype=float).reshape(3,)
    
    def find_nodes_in_vicinity(self, radius=None, max_nodes=None):
        if radius is None:
            radius = self.grasp_radius
        if max_nodes is None:
            max_nodes = self.max_grasped_nodes

        Xw = self.cloth.positions
        d = np.linalg.norm(Xw - self.p[None, :], axis=1)

        inds = np.where(d <= radius)[0]
        if inds.size == 0:
            return []

        inds = inds[np.argsort(d[inds])]
        inds = inds[:max_nodes]

        return inds.tolist()
    
    def close(self, radius=None, max_nodes=None):
        '''
        Finds nodes in vicinity and closes only if found some and 
        accordingly changes the boolean self.closed 
        '''
        inds = self.find_nodes_in_vicinity(radius=radius, max_nodes=max_nodes)

        if len(inds) == 0:
            self.controlled = []
            self.local_points = np.zeros((0, 3))
            self.is_closed = False
            return False

        self.controlled = inds
        Xw = self.cloth.positions[self.controlled].copy()
        self.local_points = quat_inverse_transform_points(self.p, self.q, Xw)
        self.is_closed = True
        return True
    
    def open(self):
        self.controlled = []
        self.local_points = np.zeros((0, 3))
        self.is_closed = False

    def record_history(self):
        R = quat_to_rotmat(self.q)

        self.origin_history.append(self.p.copy())
        self.R_history.append(R.copy())

        ## show gripper only when grasping

        # visible_now = self.is_closed and (len(self.controlled) > 0)
        # self.visible_history.append(visible_now)

        ## show gripper all the time

        self.visible_history.append(True)

        grasp_now = self.is_closed and (len(self.controlled) > 0)
        self.grasp_history.append(grasp_now)
        

        

    def step(self):
        if self.is_closed and len(self.controlled) > 0:
            u = quat_transform_points(self.p, self.q, self.local_points)
            self.cloth.simulate(u=u, control=self.controlled)
        else:
            self.cloth.simulate(u=np.zeros((0, 3)), control=[])

        self.record_history()
    

#A better model later would be an oriented box or two fingertip spheres in the gripper local frame, 
# because that uses gripper orientation and avoids grabbing nodes “behind” the gripper. 
# But for your present code, the sphere is the cleanest patch.