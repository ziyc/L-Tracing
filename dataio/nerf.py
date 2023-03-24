import os
import json
import torch
import numpy as np
from tqdm import tqdm

from utils.io_util import load_rgb
from glob import glob

def gen_intrinsic(cam_angle_x, imh, imw, downscale):
    fl = .5 * imw / np.tan(.5 * cam_angle_x)
    intrinsic = np.eye(4, dtype = np.float64)
    
    intrinsic[0, 0] = fl
    intrinsic[1, 1] = fl
    intrinsic[0, 2] = imw/2
    intrinsic[1, 2] = imh/2
    
    intrinsic[0, 2] /= downscale
    intrinsic[1, 2] /= downscale
    intrinsic[0, 0] /= downscale
    intrinsic[1, 1] /= downscale
    
    return intrinsic
    

class SceneDataset(torch.utils.data.Dataset):
    def __init__(self,
                 train_cameras,
                 data_dir,
                 downscale=1.,   # [H, W]
                 val=False,
                 scale_radius=1,
                 train_cam_min_norm=1,
                 rgb_file='rgba.png'):
        
        self.instance_dir = data_dir
        self.train_cameras = train_cameras
        self.is_val = val
        if self.is_val:
            self.img_paths = sorted(glob(os.path.join(data_dir, 'val_*')))
        else:
            self.img_paths = sorted(glob(os.path.join(data_dir, 'train_*')))
        
        self.n_images = len(self.img_paths)
        self.downscale = downscale
        self.c2w_all = []
        self.intrinsics_all = []
        self.images = []
        self.albedo = []
        self.object_masks = []
        self.vis_masks = []
        
        for img_path in tqdm(self.img_paths, desc='loading images...'):
            rgba = load_rgb(os.path.join(img_path, rgb_file), downscale)
            
            # Load images
            image = torch.from_numpy(rgba[:3, :, :])
            self.images.append(image)
            
            # Load mask for object rendering
            mask = torch.from_numpy(rgba[-1, :, :])
            self.object_masks.append(mask[None, ...])
            
            if self.is_val:
                # Load mask for visual evaluation
                # set alpha_thres=0.8 for visualization this part refer to: 
                # https://github.com/google/nerfactor/blob/19651eb72af7f6174a4d9fb68c987047ba351980/nerfactor/models/nerfactor.py#L564
                vis_mask = mask.clone()
                vis_mask[vis_mask<0.8]=0
                self.vis_masks.append(vis_mask)

                # Load albedo
                albedo = load_rgb(os.path.join(img_path, 'albedo.png'), downscale)
                self.albedo.append(torch.from_numpy(albedo[:3, :, :]))
            
            # Load meta data
            meta_path = os.path.join(img_path, 'metadata.json')
            with open(meta_path, 'r') as h:
                metadata = json.load(h)
            
            # Load camera to world matrix
            cam_to_world = np.array([
                float(x) for x in metadata['cam_transform_mat'].split(',')
            ]).reshape(4, 4)
            self.c2w_all.append(torch.from_numpy(cam_to_world))
            
            # Load camera intrinsic matrix
            imh, imw = metadata['imh'], metadata['imw']
            cam_angle_x = metadata['cam_angle_x']
            intrinsic = gen_intrinsic(cam_angle_x, imh, imw, downscale) 
            self.intrinsics_all.append(torch.from_numpy(intrinsic))

        self.c2w_all = torch.stack(self.c2w_all, 0)
        if scale_radius > 0:
            for i in range(len(self.c2w_all)):
                self.c2w_all[i][:3, 3] *= (scale_radius / train_cam_min_norm)
        
        self.intrinsics_all = torch.stack(self.intrinsics_all, 0)
        self.images = torch.stack(self.images, 0)
        self.object_masks = torch.stack(self.object_masks, 0)

        self.H, self.W = self.images.shape[2], self.images.shape[3]
        self.images = self.images.permute(0, 2, 3, 1).reshape(self.n_images, -1, 3)
        self.object_masks = self.object_masks.permute(0, 2, 3, 1).reshape(self.n_images, -1)
        
        if self.is_val:
            self.albedo = torch.stack(self.albedo, 0)
            self.vis_masks = torch.stack(self.vis_masks, 0).reshape(self.n_images, -1)
            self.albedo = self.albedo.permute(0, 2, 3, 1).reshape(self.n_images, -1, 3)
        
    def __len__(self):
        return self.n_images
    
    def __getitem__(self, idx):
        sample = {
            "intrinsics": self.intrinsics_all[idx],
            "object_mask": self.object_masks[idx],
            "c2w":self.c2w_all[idx]
        }

        ground_truth = {
            "rgb": self.images[idx],
        }
        
        if self.is_val:
            ground_truth["albedo"] = self.albedo[idx]
            ground_truth["vis_masks"] = self.vis_masks[idx]

        return idx, sample, ground_truth
    
    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

if __name__ == '__main__':
    data = SceneDataset(True, '/data/chenziyu/myprojects/datasets/nerfactor/vasedeck', downscale=1)