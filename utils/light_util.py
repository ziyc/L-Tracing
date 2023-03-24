from os.path import join
import numpy as np
from xiuminglib import xiuminglib as xm

"""
    these functions borrowed and modified from NeRFactor
    https://github.com/google/nerfactor
"""


def gen_light_xyz(envmap_h, envmap_w, envmap_radius=1e2):
    """Additionally returns the associated solid angles, for integration.
    """
    # OpenEXR "latlong" format
    # lat = pi/2
    # lng = pi
    #     +--------------------+
    #     |                    |
    #     |                    |
    #     +--------------------+
    #                      lat = -pi/2
    #                      lng = -pi
    lat_step_size = np.pi / (envmap_h + 2)
    lng_step_size = 2 * np.pi / (envmap_w + 2)
    # Try to exclude the problematic polar points
    lats = np.linspace(
        np.pi / 2 - lat_step_size, -np.pi / 2 + lat_step_size, envmap_h)
    lngs = np.linspace(
        np.pi - lng_step_size, -np.pi + lng_step_size, envmap_w)
    lngs, lats = np.meshgrid(lngs, lats)

    # To Cartesian
    rlatlngs = np.dstack((envmap_radius * np.ones_like(lats), lats, lngs))
    rlatlngs = rlatlngs.reshape(-1, 3)
    xyz = xm.geometry.sph.sph2cart(rlatlngs)
    xyz = xyz.reshape(envmap_h, envmap_w, 3)

    # Calculate the area of each pixel on the unit sphere (useful for
    # integration over the sphere)
    sin_colat = np.sin(np.pi / 2 - lats)
    areas = 4 * np.pi * sin_colat / np.sum(sin_colat)

    assert 0 not in areas, \
        "There shouldn't be light pixel that doesn't contribute"

    return xyz, areas


def load_light(envmap_path, envmap_inten=1., envmap_h=None, vis_path=None):
    if envmap_path == 'white':
        h = 16 if envmap_h is None else envmap_h
        envmap = np.ones((h, 2 * h, 3), dtype=float)

    elif envmap_path == 'point':
        h = 16 if envmap_h is None else envmap_h
        envmap = np.zeros((h, 2 * h, 3), dtype=float)
        i = -envmap.shape[0] // 4
        j = -int(envmap.shape[1] * 7 / 8)
        d = 2
        envmap[(i - d):(i + d), (j - d):(j + d), :] = 1

    else:
        envmap = xm.io.exr.read(envmap_path)

    # Optionally resize
    if envmap_h is not None:
        envmap = xm.img.resize(envmap, new_h=envmap_h)

    # Scale by intensity
    envmap = envmap_inten * envmap

    # visualize the environment map in effect
    if vis_path is not None:
        xm.io.img.write_arr(envmap, vis_path, clip=True)

    return envmap