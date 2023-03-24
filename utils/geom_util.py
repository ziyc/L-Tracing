import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.qhull import QhullError
import torch
import torch.nn as nn
from . import math as mathutil

"""
    these functions borrowed and modified from NeRFactor
    https://github.com/google/nerfactor
"""

def get_convex_hull(pts):
    try:
        hull = ConvexHull(pts)
    except QhullError:
        hull = None
    return hull


def in_hull(hull, pts):
    verts = hull.points[hull.vertices, :]
    hull = Delaunay(verts)
    return hull.find_simplex(pts) >= 0


def rad2deg(rad):
    return 180 / np.pi * rad


def slerp(p0, p1, t):
    assert p0.ndim == p1.ndim == 2, "Vectors must be 2D"

    if p0.shape[0] == 1:
        cos_omega = p0 @ tf.transpose(p1)
    elif p0.shape[1] == 1:
        cos_omega = tf.transpose(p0) @ p1
    else:
        raise ValueError("Vectors should have one singleton dimension")

    omega = mathutil.safe_acos(cos_omega)

    z0 = p0 * tf.sin((1 - t) * omega) / tf.sin(omega)
    z1 = p1 * tf.sin(t * omega) / tf.sin(omega)

    z = z0 + z1
    return z


def gen_world2local(normal, eps=1e-6):
    """Generates rotation matrices that transform world normals to local +Z
    (world tangents to local +X, and world binormals to local +Y).

    `normal`: Nx3
    """
    # normal = mathutil.safe_l2_normalize(normal, axis=1)
    normal = nn.functional.normalize(normal, p=2, dim=1, eps=1e-6)
    

    # To avoid colinearity with some special normals that may pop up
    # z = tf.convert_to_tensor((0, 0, 1), dtype=tf.float32) + eps
    # z = tf.tile(z[None, :], (tf.shape(normal)[0], 1))

    z = torch.tensor((0, 0, 1), dtype=torch.float32) + eps
    z = z.cuda()
    z = torch.tile(z[None, :], (normal.shape[0], 1))
    

    # Tangents
    # t = tf.linalg.cross(normal, z)
    t = torch.cross(normal, z, dim=-1)
    # tf.debugging.assert_greater(
    #     tf.linalg.norm(t, axis=1), 0., message=(
    #         "Found zero-norm tangents, either because of colinearity "
    #         "or zero-norm normals"))
    assert torch.gt(torch.linalg.norm(t, dim=1), 0.).min(), \
        "Found zero-norm tangents, either because of colinearity "

    # t = mathutil.safe_l2_normalize(t, axis=1)
    t = nn.functional.normalize(t, p=2, dim=1, eps=1e-6)

    # print(t,t.dtype)
    # exit()

    # Binormals
    # No need to normalize because normals and tangents are orthonormal
    b = torch.cross(normal, t, dim=-1)
    # b = mathutil.safe_l2_normalize(b, axis=1)
    b = nn.functional.normalize(b, p=2, dim=1, eps=1e-6)

    # Rotation matrices
    rot = torch.stack((t, b, normal), dim=1)
    # So that at each pixel, we have a 3x3 matrix whose ROWS are world
    # tangents, binormals, and normals

    return rot


def dir2rusink(a, b):
    """Adapted from
    third_party/nielsen2015on/coordinateFunctions.py->DirectionsToRusink().

    `a` and `b` should be both Nx3.
    """
    # a = mathutil.safe_l2_normalize(a, axis=1)
    # b = mathutil.safe_l2_normalize(b, axis=1)
    # h = mathutil.safe_l2_normalize((a + b) / 2, axis=1)

    a = nn.functional.normalize(a, p=2, dim=1, eps=1e-6)
    b = nn.functional.normalize(b, p=2, dim=1, eps=1e-6)
    h = nn.functional.normalize((a + b) / 2, p=2, dim=1, eps=1e-6)

    theta_h = mathutil.safe_acos(h[:, 2])
    phi_h = mathutil.safe_atan2(h[:, 1], h[:, 0])

    binormal = torch.tensor((0, 1, 0), dtype=torch.float32).cuda()
    normal = torch.tensor((0, 0, 1), dtype=torch.float32).cuda()

    def rot_vec(vector, axis, angle):
        """Rotates vector around arbitrary axis.
        """
        cos_ang = torch.reshape(torch.cos(angle), (-1,))
        sin_ang = torch.reshape(torch.sin(angle), (-1,))
        vector = vector.reshape(-1, 3)
        axis = axis.reshape(-1, 3)
        return vector * cos_ang[:, None] + \
            axis * torch.matmul(
                vector, axis.transpose(1, 0)) * (1 - cos_ang)[:, None] + \
            torch.cross(
                torch.tile(axis, (vector.shape[0], 1)), vector, dim=-1) * sin_ang[:, None]

    # What is the incoming/outgoing direction in the Rusink. frame?
    diff = rot_vec(rot_vec(b, normal, -phi_h), binormal, -theta_h)
    diff0, diff1, diff2 = diff[:, 0], diff[:, 1], diff[:, 2]
    # NOTE: when a and b are the same, diff will lie along +h, so theta_d=0
    # and phi_d is meaningless. This is fine in forward pass, but creates
    # NaN's in backward pass. Avoiding this problem by using safe_atan2
    theta_d = mathutil.safe_acos(diff2)
    phi_d = torch.remainder(mathutil.safe_atan2(diff1, diff0), np.pi)
    rusink = torch.stack((phi_d, theta_h, theta_d)).transpose(1,0)

    return rusink
