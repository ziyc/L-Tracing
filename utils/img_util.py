import numpy as np
import cv2
from PIL import Image
import torch

"""
    these functions borrowed and modified from NeRFactor
    https://github.com/google/nerfactor
"""

def _clip_0to1_warn(tensor_0to1):
    """Enforces [0, 1] on a tensor/array that should be already [0, 1].
    """
    msg = "Some values outside [0, 1], so clipping happened"
    if isinstance(tensor_0to1, tf.Tensor):
        if tf.reduce_min(tensor_0to1) < 0 or tf.reduce_max(tensor_0to1) > 1:
            logger.debug(msg)
            tensor_0to1 = tf.clip_by_value(
                tensor_0to1, clip_value_min=0, clip_value_max=1)
    else:
        if tensor_0to1.min() < 0 or tensor_0to1.max() > 1:
            logger.debug(msg)
            tensor_0to1 = np.clip(tensor_0to1, 0, 1)
    return tensor_0to1


def alpha_blend(tensor1, alpha, tensor2=None):
    """Alpha-blend two tensors. If the second tensor is `None`, the first
    tensor will be blended with a all-zero tensor, equivalent to masking if
    alpha is binary.

    Alpha should range from 0 to 1.
    """
    if isinstance(tensor1, torch.Tensor):
        if tensor2 is None:
            tensor2 = torch.zeros_like(tensor1)

        if len(tensor1.shape) == 3 and len(alpha.shape) == 2:
            alpha = alpha.reshape(tuple(torch.shape(alpha)) + (1,))
            alpha = torch.tile(alpha, (1, 1, torch.shape(tensor1)[2]))

        result = torch.mul(tensor1, alpha) + torch.mul(tensor2, 1. - alpha)
        # result = torch.clip(result, 0., 1.)

        return result
    else:
        if tensor2 is None:
            tensor2 = np.zeros_like(tensor1)

        if len(np.shape(tensor1)) == 3 and len(np.shape(alpha)) == 2:
            alpha = np.reshape(alpha, tuple(np.shape(alpha)) + (1,))
            alpha = np.tile(alpha, (1, 1, np.shape(tensor1)[2]))

        result = np.multiply(tensor1, alpha) + np.multiply(tensor2, 1. - alpha)
        # result = np.clip(result, 0., 1.)

        return result


def resize(img, new_h=None, new_w=None):
    """Uses TensorFlow's bilinear, antialiasing resizing, accepting an
    HxWxC tensor or array.

    Compatible with graph execution.
    """
    if isinstance(img, torch.Tensor):
        input_is_tensor = True
        tensor = img
    else:
        input_is_tensor = False
        tensor = torch.tensor(img)

    # Original size
    hw = tensor.shape
    h, w = hw[0], hw[1] # both 0D tensors

    # Figure out new size
    if new_h is not None and new_w is not None:
        if not (h / w * new_w)==new_h:
            logger.warn((
                "Aspect ratio changed in resizing: original size is %s; "
                "new size is %s"), (h, w), (new_h, new_w))
    elif new_h is None and new_w is not None:
        new_h = int(h / w * new_w)
    elif new_h is not None and new_w is None:
        new_w = int(w / h * new_h)
    else:
        raise ValueError("At least one of new height or width must be given")
    # new_shape = (new_h, new_w)
    # cv2 resize (width, height)
    new_shape = (new_w, new_h)

    resized = cv2.resize(tensor.cpu().numpy(), new_shape, interpolation = cv2.INTER_LINEAR)
    resized = torch.from_numpy(resized).cuda()

    # resized = tf.image.resize(
    #     tensor, new_shape, method='bilinear', antialias=True)

    if input_is_tensor:
        return resized

    # Restore the original data type if input is not a tensor
    orig_dtype = img.dtype if isinstance(img, np.ndarray) else img.numpy().dtype
    return resized.numpy().astype(orig_dtype)


def linear2srgb(tensor_0to1):
    if isinstance(tensor_0to1, torch.Tensor):
        pow_func = torch.pow
        where_func = torch.where
    else:
        pow_func = np.pow
        where_func = np.where

    srgb_linear_thres = 0.0031308
    srgb_linear_coeff = 12.92
    srgb_exponential_coeff = 1.055
    srgb_exponent = 2.4

    tensor_0to1 = torch.clip(tensor_0to1, min=0.001, max=1.)

    assert torch.gt(tensor_0to1, 0).all().item(), "input of pow func should be greater than 0"
    
    tensor_linear = tensor_0to1 * srgb_linear_coeff
    tensor_nonlinear = srgb_exponential_coeff * (
        pow_func(tensor_0to1, 1 / srgb_exponent)
    ) - (srgb_exponential_coeff - 1)

    is_linear = tensor_0to1 <= srgb_linear_thres
    tensor_srgb = where_func(is_linear, tensor_linear, tensor_nonlinear)

    return tensor_srgb


def to_uint(tensor_0to1, target_type='uint8'):
    """Converts a float tensor/array with values in [0, 1] to an unsigned
    integer type for visualization.
    """
    if isinstance(tensor_0to1, tf.Tensor):
        target_type = tf.as_dtype(target_type)
        tensor_0to1 = _clip_0to1_warn(tensor_0to1)
        tensor_uint = tf.cast(tensor_0to1 * target_type.max, target_type)
    else:
        tensor_0to1 = _clip_0to1_warn(tensor_0to1)
        tensor_uint = (np.iinfo(target_type).max * tensor_0to1).astype(
            target_type)
    return tensor_uint


def rot90(img, counterclockwise=False):
    if isinstance(img, np.ndarray):
        from_to = (0, 1) if counterclockwise else (1, 0)
        img_ = np.rot90(img, axes=from_to)
    elif isinstance(img, tf.Tensor):
        k = 1 if counterclockwise else 3
        img_ = tf.image.rot90(img, k=k)
    else:
        raise TypeError(img)
    return img_


def set_left_top_corner(tensor, val):
    """Thanks to "eager tensor doesn't support assignment."
    """
    mask = np.ones(tensor.shape)
    mask[:, 0, 0, :] = val
    mask = tf.convert_to_tensor(mask, dtype=tf.float32)
    return tf.multiply(mask, tensor)


def hconcat(img_list, out_w=None):
    total = []
    for img in img_list:
        if img.ndim == 2:
            img = np.dstack([img] * 3)
        if total:
            prev = total[-1]
            img = resize(img, new_h=prev.shape[0])
        total.append(img)
    total = np.hstack(total)
    if out_w is not None:
        total = resize(total, new_w=out_w)
    return total


def vconcat(img_list, out_h=None):
    total = []
    for img in img_list:
        if img.ndim == 2:
            img = np.dstack([img] * 3)
        if total:
            prev = total[-1]
            img = resize(img, new_w=prev.shape[1])
        total.append(img)
    total = np.vstack(total)
    if out_h is not None:
        total = resize(total, new_h=out_h)
    return total


def frame_image(img, rgb=None, width=4):
    img_dtype_str = str(img.dtype)
    if img_dtype_str.startswith('float'):
        dtype_max = 1.
    elif img_dtype_str.startswith('uint'):
        dtype_max = np.iinfo(img.dtype).max
    else:
        raise NotImplementedError(img_dtype_str)

    if rgb is None:
        rgb = (0, 0, 1)
    rgb = np.array(rgb, dtype=img.dtype) * dtype_max

    img[:width, :, :] = rgb
    img[-width:, :, :] = rgb
    img[:, :width, :] = rgb
    img[:, -width:, :] = rgb


def embed_into(inset, img, inset_scale=0.2):
    inset_h = int(inset_scale * img.shape[0])
    inset_w = int(inset_h / inset.size[1] * inset.size[0])
    inset = inset.resize((inset_w, inset_h))
    bg = Image.fromarray(img)
    bg.paste(inset, (bg.size[0] - inset.size[0], 0), inset) # inset's A
    # channel will be used as mask
    composite = np.array(bg)
    return composite

def vis_light(light_rgb: torch.Tensor, light_xyz: torch.Tensor, H=16, W=32):
    xyz = light_xyz.reshape(-1, 3)
    r = torch.norm(xyz, dim=-1)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    phi = torch.arccos(torch.sqrt(x**2 + y**2) / r) * torch.sign(y)  
    lamb = torch.arccos(x / torch.sqrt(x**2 + y**2)) * torch.sign(z)
    latitude = 180/3.1415829 * phi
    longitude = 180/3.1415829 * lamb
    
    lat_gird = torch.linspace(-90 + 180/H, 90, steps=H, device=xyz.device)
    long_grid = torch.linspace(-180 + 360/W, 180, steps=W, device=xyz.device)
    
    lat_idx = torch.searchsorted(lat_gird, latitude)
    long_idx = torch.searchsorted(long_grid, longitude)
    
    rgb = torch.zeros((H * W, 3), dtype = light_rgb.dtype, device = light_rgb.device)
    
    idx = lat_idx * W + long_idx
    
    rgb.index_add_(0, idx, light_rgb.reshape(-1,3))
    
    rgb = torch.clip(rgb, min=0., max=1.)
    
    return rgb.reshape(1, H, W, 3).permute(0, 3, 1, 2)    

def tonemap(hdr, method='gamma', gamma=2.2):
    r"""Tonemaps an HDR image.

    Args:
        hdr (numpy.ndarray): HDR image.
        method (str, optional): Values accepted: ``'gamma'`` and ``'reinhard'``.
        gamma (float, optional): Gamma value used if method is ``'gamma'``.

    Returns:
        numpy.ndarray: Tonemapped image :math:`\in [0, 1]`.
    """
    if method == 'reinhard':
        import cv2
        tonemapper = cv2.createTonemapReinhard(1, 1, 0, 0)
        img = tonemapper.process(hdr)
    elif method == 'gamma':
        img = (hdr / hdr.max()) ** (1 / gamma)
    else:
        raise ValueError(method)

    # Clip, if necessary, to guard against numerical errors
    minv, maxv = img.min(), img.max()
    if minv < 0:
        print("Clipping negative values (min.: %f)", minv)
        img = np.clip(img, 0, np.inf)
    if maxv > 1:
        print("Clipping >1 values (max.: %f)", maxv)
        img = np.clip(img, -np.inf, 1)

    return img 

def write_arr(arr_0to1, path, clip = True):
    arr_min, arr_max = arr_0to1.min(), arr_0to1.max()
    if clip:
        if arr_max > 1:
            print("Maximum before clipping: %f", arr_max)
        if arr_min < 0:
            print("Minimum before clipping: %f", arr_min)
        arr_0to1 = np.clip(arr_0to1, 0, 1)
    else:
        assert arr_min >= 0 and arr_max <= 1, \
            "Input should be in [0, 1], or allow it to be clipped"

    arr_0to1 = arr_0to1 * 255
    arr_0to1 = cv2.cvtColor(arr_0to1.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, arr_0to1)

    return arr_0to1

def rgb2gray(rgb):

    r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
