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

def quat_to_rotmat(q):
    q = quat_normalize(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ], dtype=float)

# =========================
# Dual quaternion utilities
# =========================
def dq_normalize(dq):
    """
    Normalize a rigid dual quaternion using only the rotation part norm.
    dq = [qr(4), qt(4)]
    """
    dq = np.array(dq, dtype=float).reshape(8,)
    qr = dq[:4]
    qt = dq[4:]
    n = np.linalg.norm(qr) + 1e-12
    qr = qr / n
    qt = qt / n
    return np.hstack([qr, qt])


def dq_from_rt(q, t):
    """
    Build unit dual quaternion from rotation quaternion q and translation t.
    dq = qr + eps * qt
    qt = 0.5 * [0, t] * qr
    """
    q = quat_normalize(q)
    t = np.array(t, dtype=float).reshape(3,)
    tq = np.array([0.0, t[0], t[1], t[2]], dtype=float)
    qt = 0.5 * quat_mul(tq, q)
    return dq_normalize(np.hstack([q, qt]))


def dq_to_rt(dq):
    """
    Extract rotation quaternion and translation vector from unit dual quaternion.
    t_quat = 2 * qt * conj(qr)
    """
    dq = dq_normalize(dq)
    qr = dq[:4]
    qt = dq[4:]
    t_quat = 2.0 * quat_mul(qt, quat_conjugate(qr))
    t = t_quat[1:]
    return qr, t


def dq_conjugate(dq):
    """
    Quaternion conjugate on both parts.
    For unit rigid DQ, this is the inverse transform.
    """
    dq = np.array(dq, dtype=float).reshape(8,)
    qr = dq[:4]
    qt = dq[4:]
    return np.hstack([quat_conjugate(qr), quat_conjugate(qt)])


def dq_mul(dq1, dq2):
    """
    (qr1 + eps qt1)(qr2 + eps qt2)
      = qr1 qr2 + eps (qr1 qt2 + qt1 qr2)
    """
    dq1 = np.array(dq1, dtype=float).reshape(8,)
    dq2 = np.array(dq2, dtype=float).reshape(8,)

    qr1, qt1 = dq1[:4], dq1[4:]
    qr2, qt2 = dq2[:4], dq2[4:]

    qr = quat_mul(qr1, qr2)
    qt = quat_mul(qr1, qt2) + quat_mul(qt1, qr2)
    return np.hstack([qr, qt])


def dq_transform_point(dq, x):
    """
    Apply rigid transform encoded by dq to point x.
    Uses extracted (q, t):
        x' = q * q* + t
    """
    q, t = dq_to_rt(dq)
    return quat_rotate_vector(q, x) + t


def dq_transform_points(dq, X):
    """
    From gripper coordinate frame to world frame
    """
    X = np.array(X, dtype=float)
    return np.vstack([dq_transform_point(dq, x) for x in X])


def dq_inverse_transform_point(dq, x):
    """
    From world coordinate frame of the cloth to gripper local frame
    """
    q, t = dq_to_rt(dq)
    return quat_rotate_vector(quat_conjugate(q), x - t)


def dq_inverse_transform_points(dq, X):
    X = np.array(X, dtype=float)
    return np.vstack([dq_inverse_transform_point(dq, x) for x in X])


# Optional helper for pure translation update
def dq_from_translation(t):
    return dq_from_rt([1.0, 0.0, 0.0, 0.0], t)


# Optional helper for pure rotation update
def dq_from_rotation(q):
    return dq_from_rt(q, [0.0, 0.0, 0.0])


class SimulateGripper:
    """
    Minimal controller with dual quaternions:
    - select cloth nodes
    - define a rigid transform frame as a unit dual quaternion
    - move those nodes according to the frame

    To check: for screw motions, interpolate pose_dq with ScLERP.
    """

    def __init__(self, cloth: Cloth):
        self.cloth: Cloth = cloth

        # controlled node indices
        self.controlled = []

        # selected cloth points in local gripper frame
        self.local_points = np.zeros((0, 3))

        # pose as unit dual quaternion
        # identity: qr = [1,0,0,0], qt = [0,0,0,0]
        self.pose_dq = np.array(
            [1.0, 0.0, 0.0, 0.0,   0.0, 0.0, 0.0, 0.0],
            dtype=float
        )

        # So that the gripper is only shown when it grasps something
        self.origin_history = [np.array([0.0, 0.0, 0.0], dtype=float)]
        self.R_history = [np.eye(3)]
        self.visible_history = [False]
        self.has_ever_grasped = False

    def set_pose(self, dq=None, q=None, p=None):
        """
        Set pose either directly from dq, or from (q, p).
        """
        if dq is not None:
            self.pose_dq = dq_normalize(dq)
        else:
            if q is None:
                q = [1.0, 0.0, 0.0, 0.0]
            if p is None:
                p = [0.0, 0.0, 0.0]
            self.pose_dq = dq_from_rt(q, p)

    def get_pose_rt(self):
        """
        Returns current pose as (q, p).
        Useful for debugging or compatibility.
        """
        return dq_to_rt(self.pose_dq)
    
# select_nodes() stores where the grasped nodes are relative to the gripper, 
# step() keeps recomputing where those same relative points should be in the world as the gripper pose changes.

    def select_nodes(self, inds):
        """
        Store selected cloth nodes in the local frame:
            x_local = T^{-1}(x_world)
        """
        self.controlled = list(inds)

        if len(self.controlled) == 0:
            self.local_points = np.zeros((0, 3))
            return
        
        self.has_ever_grasped = True

        Xw = self.cloth.positions[self.controlled].copy()
        # transform the selected nodes from world frame to gripper frame
        self.local_points = dq_inverse_transform_points(self.pose_dq, Xw)

    def record_history(self):
        q, p = dq_to_rt(self.pose_dq)
        R = quat_to_rotmat(q)

        self.origin_history.append(p.copy())
        self.R_history.append(R.copy())
        self.visible_history.append(self.has_ever_grasped)

    def clear_nodes(self):
        self.controlled = []
        self.local_points = np.zeros((0, 3))

    def current_targets(self):
        """
        Reconstruct selected cloth points in world frame:
            x_world = T(x_local)
        """
        if len(self.controlled) == 0:
            return np.zeros((0, 3))

        return dq_transform_points(self.pose_dq, self.local_points)

    def step(self):
        u = self.current_targets()
        self.cloth.simulate(u=u, control=self.controlled)
        self.record_history()

    def free_step(self):
        """
        Simulate cloth without control, while keeping the gripper pose frozen.
        """
        self.cloth.simulate(u=np.zeros((0, 3)), control=[])
        self.record_history()
