# SPDX-License-Identifier: MIT
# Port of recover_root_rot_pos / recover_from_ric (HumanML3D motion_process.py, Guo et al., MIT,
# as vendored by MDM and Kirin) and the quaternion helpers it uses.
"""From the 407-dimensional motion features back to 3D joint positions.

Feature layout per frame (34-joint animal skeleton, root = joint 0)::

    [0]        root angular velocity about +Y (rad/frame)
    [1:3]      root linear velocity on the XZ plane, in the root's heading frame
    [3]        root height
    [4:103]    33 joints × xyz relative to the root, in the root's heading frame ("ric")
    [103:301]  33 joints × 6D rotations           (unused here, as upstream)
    [301:403]  34 joints × xyz velocities         (unused here, as upstream)
    [403:407]  foot contacts                      (unused here, as upstream)

Integrating the root yaw and velocity over time and rotating the "ric" positions out of the
heading frame gives world positions ``[..., T, 34, 3]``; this is all upstream's ``generate.py``
does after sampling. Everything here is MLX ops on float32, so it runs on the GPU as part of the
generation graph.
"""

from __future__ import annotations

import mlx.core as mx

from .config import NUM_JOINTS


def _cross(a: mx.array, b: mx.array) -> mx.array:
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    return mx.stack([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx], axis=-1)


def qrot(q: mx.array, v: mx.array) -> mx.array:
    """Rotate ``v [..., 3]`` by unit quaternions ``q [..., 4]`` (w, x, y, z), as upstream ``qrot``."""
    qvec = q[..., 1:]
    uv = _cross(qvec, v)
    uuv = _cross(qvec, uv)
    return v + 2 * (q[..., :1] * uv + uuv)


def qinv(q: mx.array) -> mx.array:
    return q * mx.array([1.0, -1.0, -1.0, -1.0], dtype=q.dtype)


def recover_root_rot_pos(data: mx.array) -> tuple[mx.array, mx.array]:
    """``data [..., T, F]`` -> (root yaw quaternions ``[..., T, 4]``, root positions ``[..., T, 3]``)."""
    rot_vel = data[..., 0]
    # r_rot_ang[..., 1:] = rot_vel[..., :-1]; cumsum over time
    angle = mx.cumsum(mx.concatenate([mx.zeros_like(rot_vel[..., :1]), rot_vel[..., :-1]], axis=-1), axis=-1)
    zeros = mx.zeros_like(angle)
    r_rot_quat = mx.stack([mx.cos(angle), zeros, mx.sin(angle), zeros], axis=-1)
    # r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    vel = mx.concatenate([mx.zeros_like(data[..., :1, 1:3]), data[..., :-1, 1:3]], axis=-2)
    r_pos = mx.stack([vel[..., 0], mx.zeros_like(vel[..., 0]), vel[..., 1]], axis=-1)
    r_pos = mx.cumsum(qrot(qinv(r_rot_quat), r_pos), axis=-2)
    r_pos = mx.stack([r_pos[..., 0], data[..., 3], r_pos[..., 2]], axis=-1)
    return r_rot_quat, r_pos


def recover_from_ric(data: mx.array, joints_num: int = NUM_JOINTS) -> mx.array:
    """``data [..., T, F]`` (denormalized features) -> joint positions ``[..., T, joints_num, 3]``."""
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.reshape(*positions.shape[:-1], joints_num - 1, 3)
    q = mx.broadcast_to(qinv(r_rot_quat)[..., None, :], (*positions.shape[:-1], 4))
    positions = qrot(q, positions)
    offset = mx.stack([r_pos[..., 0], mx.zeros_like(r_pos[..., 0]), r_pos[..., 2]], axis=-1)
    positions = positions + offset[..., None, :]
    return mx.concatenate([r_pos[..., None, :], positions], axis=-2)
