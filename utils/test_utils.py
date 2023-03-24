import cv2
from glob import glob

import imageio
import skimage
from skimage.transform import rescale
import torch

def load_rgb(path, downscale=1):
    img = imageio.imread(path)
    img = skimage.img_as_float32(img)
    if downscale != 1:
        img = rescale(img, 1. / downscale, anti_aliasing=False, multichannel=True)
    return img

def get_albedo_scale(exp_dir, no_scale=False, gamma=2.2):
    if no_scale:
        return torch.tensor([1.,1.,1.])
    else:
        path = glob(exp_dir+'/imgs/epoch000000000/batch000000000/mask_albedo*.png')[0]
        img = load_rgb(path)
        H = img.shape[0]
        half_H = int(H/2)
        gt_img = img[:half_H,:,:].reshape(-1,3)
        pred_img = img[half_H:,:,:].reshape(-1,3)
        pred_img = pred_img ** gamma # undo gamma
        opt_scale=[]
        for i in range(3):
            x_hat = pred_img[:,i]
            x = gt_img[:,i]
            scale = x_hat.dot(x) / x_hat.dot(x_hat)
            opt_scale.append(scale)
        opt_scale = torch.tensor(opt_scale, dtype=torch.float32)
        return opt_scale