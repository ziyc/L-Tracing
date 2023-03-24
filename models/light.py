import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
from utils.light_util import gen_light_xyz
from utils import nf_util, img_util
from glob import glob
import os

# nerfactor light model : Light Probe Image
# Fixed position and area
# Trainable RGB
class source(nn.Module):
    def __init__(self, 
                 light_h,
                 light_w=None,
                 maxv=1,
                 light_map_radius=100.,):
        super().__init__()
        
        self.radius = light_map_radius
        self.modelname = "source"
        
        self.h = light_h
        if light_w is None:
            self.w = 2 * light_h
        else:
            self.w = light_w
        self.light_res = (light_h, self.w)
        self.light_num = light_h * self.w
        
        lxyz, lareas = gen_light_xyz(*self.light_res, light_map_radius)
        self.lxyz = torch.tensor(lxyz, dtype=torch.float32).cuda()
        self.lareas = torch.tensor(lareas, dtype=torch.float32).cuda()

        light_init = torch.rand(self.light_res + (3,), dtype=torch.float32) * maxv
        light_init = torch.clip(light_init, min=0., max=np.inf)
        self.rgb = nn.Parameter(data=light_init, requires_grad=True)        
        
class OLAT(nn.Module):
    def __init__(self, 
                 light_h,
                 light_w=None,
                 olat_inten=200,
                 ambi_inten=0,
                 embed_light_h=32,
                 white_bg=True):
        super().__init__()
        novel_olat = collections.OrderedDict()
        
        self.h = light_h
        if self.h < 0:
            return None
        
        if light_w is None:
            self.w = 2 * light_h
        else:
            self.w = light_w
        self.light_res = (light_h, self.w)
        light_shape = self.light_res + (3,)
        
        if white_bg:
            # Add some ambient lighting to better match perception
            ambient = ambi_inten * torch.ones(light_shape, dtype=torch.float32)
        else:
            ambient = torch.zeros(light_shape, dtype=torch.float32)
        
        for i in range(self.light_res[0]):
            for j in range(self.light_res[1]):
                one_hot = nf_util.one_hot_img(*ambient.shape, i, j)
                envmap = olat_inten * one_hot + ambient
                novel_olat['%04d-%04d' % (i, j)] = envmap.cuda()
        
        self.novel_olat = novel_olat

        self.novel_olat_uint = {}
        for k, v in self.novel_olat.items():
            vis_light = nf_util.vis_light(v, h=embed_light_h)
            self.novel_olat_uint[k] = vis_light
            
class Probes(nn.Module):
    def __init__(self,
        test_envmap_dir,
        light_h,
        light_w=None,
        embed_light_h=32):
        super().__init__()    
        novel_probes = collections.OrderedDict()
        
        self.h = light_h
        if light_w is None:
            self.w = 2 * light_h
        else:
            self.w = light_w
        self.light_res = (light_h, self.w)
        light_shape = self.light_res + (3,)
        
        for path in sorted(glob(os.path.join(test_envmap_dir,'*.hdr'))):
            name = os.path.basename(path)[:-len('.hdr')]
            envmap = self._load_light(path)
            novel_probes[name] = envmap
        self.novel_probes = novel_probes
        
        self.novel_probes_uint = {}
        for k, v in self.novel_probes.items():
            vis_light = nf_util.vis_light(v, h=embed_light_h)
            self.novel_probes_uint[k] = vis_light
        
    def _load_light(self, path):
        arr = nf_util.read(path)
        tensor = torch.from_numpy(arr)
        resized = img_util.resize(tensor, new_h=self.light_res[0])
        return resized
    
class EnvMap(nn.Module):
    def __init__(self,
        light_h,
        env_map_path,
        light_w=None):
        super().__init__()
        
        self.h = light_h
        if light_w is None:
            self.w = 2 * light_h
        else:
            self.w = light_w
        self.light_res = (light_h, self.w)
        light_shape = self.light_res + (3,)
        
        env_map = torch.load(env_map_path)
        self.rgb = env_map
        
        
def get_light(light_cfg):
    modelname = light_cfg.pop('modelname')
    if modelname == 'source':
        return source(**light_cfg)
    elif modelname == 'OLAT':
        return OLAT(**light_cfg)
    elif modelname == 'Probes':
        return Probes(**light_cfg)
    elif modelname == 'Envmap':
        return EnvMap(**light_cfg)
    else:
        raise NotImplementedError
    # TODO: support more light models

if __name__=='__main__':
    light = source(16, '_')
    torch.save(light.lxyz.cpu(),'./light_direct_16X32_unit.pt')