import torch
import numpy as np

from xiuminglib import xiuminglib as xm

from . import img_util

def vis_light(light_probe, outpath=None, h=None):
    # In case we are predicting too low of a resolution
    if h is not None:
        light_probe = img_util.resize(light_probe, new_h=h)
    
    # We need NumPy array onwards
    if isinstance(light_probe, torch.Tensor):
        light_probe = light_probe.cpu().numpy()

    # Tonemap
    img = xm.img.tonemap(light_probe, method='gamma', gamma=4) # [0, 1]
    # srgb = xm.img.linear2srgb(linear)
    # srgb_uint = xm.img.denormalize_float(srgb)
    img_uint = xm.img.denormalize_float(img)

    # Optionally, write to disk
    if outpath is not None:
        xm.io.img.write_img(img_uint, outpath)

    return img_uint

def one_hot_img(h, w, c, i, j):
    """Makes a float32 HxWxC tensor with 1s at (i, j, *) and 0s everywhere else.
    """
    # ind = [(i, j, x) for x in range(c)]
    # ind = tf.convert_to_tensor(ind)
    # updates = tf.ones((c,), dtype=tf.float32)
    # one_hot = tf.scatter_nd(ind, updates, (h, w, c))

    one_hot = torch.zeros((h, w, c),dtype=torch.float32)
    one_hot[i,j,:] = torch.ones((c,), dtype=torch.float32)
    return one_hot

def read(path):
    """Reads an HDR map from disk.

    Args:
        path (str): Path to the .hdr file.

    Returns:
        numpy.ndarray: Loaded (float) HDR map with RGB channels in order.
    """
    import cv2

    with open(path, 'rb') as h:
        buffer_ = np.fromstring(h.read(), np.uint8)
    bgr = cv2.imdecode(buffer_, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return rgb