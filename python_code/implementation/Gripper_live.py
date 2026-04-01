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

def quat_from_rotvec(rotvec):
    rotvec = np.asarray(rotvec, dtype=float).reshape(3,)
    angle = np.linalg.norm(rotvec)
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    axis = rotvec / angle
    return quat_from_axis_angle(axis, angle)

class SimulateGripper:

    def __init__(self, cloth: Cloth, box_size=np.array([0.03, 0.06, 0.02], dtype=float), max_grasped_nodes=6):

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
        
        self.max_grasped_nodes = max_grasped_nodes # How much the gripper can grasp depending on its size
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
    
### parallelopiped version
    # def find_nodes_in_vicinity(self, box=None, max_nodes=None):
    #     if box is None:
    #         box = self.box_size
    #     if max_nodes is None:
    #         max_nodes = self.max_grasped_nodes

    #     box = np.asarray(box, dtype=float).reshape(3,)
    #     half = 0.5 * box 

    #     Xw = self.cloth.positions
    #     Xl = quat_inverse_transform_points(self.p, self.q, Xw)  # world -> gripper local

    #     inside = (
    #         (np.abs(Xl[:, 0]) <= half[0]) &
    #         (np.abs(Xl[:, 1]) <= half[1]) &
    #         (np.abs(Xl[:, 2]) <= half[2])
    #     )

    #     inds = np.where(inside)[0]
    #     if inds.size == 0:
    #         return []

    #     # prefer nodes near the grasp-box center
    #     d = np.linalg.norm(Xl[inds], axis=1)
    #     inds = inds[np.argsort(d)]

    #     if max_nodes is not None:
    #         inds = inds[:max_nodes] 

        # return inds.tolist()
    
    def find_nodes_in_vicinity(self, box=None, center_local=None, max_nodes=None):
        if box is None:
            box = self.box_size
        if max_nodes is None:
            max_nodes = self.max_grasped_nodes
        if center_local is None:
            center_local = np.zeros(3, dtype=float)

        box = np.asarray(box, dtype=float).reshape(3,)
        center_local = np.asarray(center_local, dtype=float).reshape(3,)
        half = 0.5 * box # because the box is centered at the gripper frame origin
        # and I want to be able to grasp nodes on either side of it

        Xw = self.cloth.positions
        Xl = quat_inverse_transform_points(self.p, self.q, Xw)  # world -> gripper local

        Xc = Xl - center_local  # shift box center to desired local position

        inside = (
            (np.abs(Xc[:, 0]) <= half[0]) &
            (np.abs(Xc[:, 1]) <= half[1]) &
            (np.abs(Xc[:, 2]) <= half[2])
        )

        inds = np.where(inside)[0]
        if inds.size == 0:
            return []

        # prefer nodes near the shifted grasp-box center
        d = np.linalg.norm(Xc[inds], axis=1)
        inds = inds[np.argsort(d)]

        if max_nodes is not None:
            inds = inds[:max_nodes]

        return inds.tolist()
    
    # def set_open(self, is_open, radius=None, max_nodes=None):
    def set_open(self, is_open, box=None, center_local=None, max_nodes=None):

        is_open = bool(is_open)

        was_open = self.is_open
        self.is_open = is_open

        # open -> closed : attempt grasp
        if was_open and (not self.is_open):
            # inds = self.find_nodes_in_vicinity(radius=radius, max_nodes=max_nodes)
            inds = self.find_nodes_in_vicinity(box=box, center_local=center_local, max_nodes=max_nodes)            # print(f'grasped nodes: {inds}') # only print when changing from open to closed

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
        