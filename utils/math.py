"""
    these functions borrowed and modified from NeRFactor
    https://github.com/google/nerfactor
"""

import torch

def log10(x):
    num = tf.math.log(x)
    denom = tf.math.log(tf.constant(10, dtype=num.dtype))
    return num / denom


# @tf.custom_gradient
def tf_safe_atan2(x, y, eps=1e-6):
    """Numerically stable version to safeguard against (0, 0) input, which
    causes the backward of tf.atan2 to go NaN.
    """
    z = torch.atan2(x, y)

    # def grad(dz):
    #     denom = x ** 2 + y ** 2
    #     denom += eps
    #     dzdx = y / denom
    #     dzdy = -x / denom
    #     return dz * dzdx, dz * dzdy

    return z#, grad


# @tf.custom_gradient
def tf_safe_acos(x, eps=1e-6):
    """Numerically stable version to safeguard against +/-1 input, which
    causes the backward of tf.acos to go inf.

    Simply clipping the input at +/-1-/+eps gives 0 gradient at the clipping
    values, but analytically, the gradients there should be large.
    """
    x_clip = torch.clip(x, min=-1., max=1.)
    y = torch.acos(x_clip)

    # def grad(dy):
    #     in_sqrt = 1. - x_clip ** 2
    #     in_sqrt += eps
    #     denom = tf.sqrt(in_sqrt)
    #     denom += eps
    #     dydx = -1. / denom
    #     return dy * dydx

    return y #, grad

eps = 1e-6


class Safe_atan2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, y):
        z = torch.atan2(x, y)
        ctx.save_for_backward(x, y)
        return z

    @staticmethod
    def backward(ctx, dz):
        x, y = ctx.saved_tensors
        denom = torch.pow(x, 2) + torch.pow(y, 2)
        denom += eps
        dzdx = y / denom
        dzdy = -x / denom
        return dz * dzdx, dz * dzdy


class Safe_acos(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        x_clip = torch.clip(input, min=-1., max=1.)
        y = torch.acos(x_clip)
        ctx.save_for_backward(x_clip)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x_clip, = ctx.saved_tensors
        in_sqrt = 1. - torch.pow(x_clip, 2)
        in_sqrt += eps
        denom = torch.sqrt(in_sqrt)
        denom += eps
        dydx = -1. / denom
        return grad_output * dydx


def safe_acos(input):
    return Safe_acos().apply(input)

def safe_atan2(x, y):
    return Safe_atan2().apply(x, y)

def safe_l2_normalize(x, axis=None, eps=1e-6):
    return tf.linalg.l2_normalize(x, axis=axis, epsilon=eps)

def l2_normalize(x, dim = None):
    n = x.shape[-1]
    x_norm = (x ** 2).sum(dim)
    x_norm = torch.clip(x_norm, min = 1e-6)
    x_norm = torch.sqrt(x_norm)
    x_norm = x_norm.unsqueeze(1).repeat(1, n)
    x = x / x_norm
    return x

def safe_cumprod(x, eps=1e-6):
    return tf.math.cumprod(x + eps, axis=-1, exclusive=True)


def inv_transform_sample(val, weights, n_samples, det=False, eps=1e-5):
    denom = tf.reduce_sum(weights, -1, keepdims=True)
    denom += eps
    pdf = weights / denom
    cdf = tf.cumsum(pdf, -1)
    cdf = tf.concat((tf.zeros_like(cdf[:, :1]), cdf), -1)

    if det:
        u = tf.linspace(0., 1., n_samples)
        u = tf.broadcast_to(u, cdf.shape[:-1] + (n_samples,))
    else:
        u = tf.random.uniform(cdf.shape[:-1] + (n_samples,))

    ind = tf.searchsorted(cdf, u, side='right') # (n_rays, n_samples)
    below = tf.maximum(0, ind - 1)
    above = tf.minimum(ind, cdf.shape[-1] - 1)
    ind_g = tf.stack((below, above), -1) # (n_rays, n_samples, 2)
    cdf_g = tf.gather(cdf, ind_g, axis=-1, batch_dims=len(ind_g.shape) - 2)
    val_g = tf.gather(val, ind_g, axis=-1, batch_dims=len(ind_g.shape) - 2)
    denom = cdf_g[:, :, 1] - cdf_g[:, :, 0] # (n_rays, n_samples)
    denom = tf.where(denom < eps, tf.ones_like(denom), denom)
    t = (u - cdf_g[:, :, 0]) / denom
    samples = val_g[:, :, 0] + t * (val_g[:, :, 1] - val_g[:, :, 0])
    return samples # (n_rays, n_samples)
