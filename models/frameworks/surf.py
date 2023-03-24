from models.base import ImplicitSurface, MLP
from models.light import get_light
from models.ray_casting import sphere_tracing_surface_points
from brdf.brdf import Model as BRDFModel
from utils import rend_util, train_util, geom_util, img_util

import copy
import functools
import numpy as np
from tqdm import tqdm
from typing import Optional
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA
from utils import rend_util, train_util, io_util

"""
    This model is mainly built similarly to NeRFactor.
    The surface factorization parts borrowed and modified from NeRFactor:
    https://github.com/google/nerfactor
    We reimplemented these parts with PyTorch.
"""

def cdf_Phi_s(x, s):
    # den = 1 + torch.exp(-s*x)
    # y = 1./den
    # return y
    return torch.sigmoid(x*s)


def sdf_to_alpha(sdf: torch.Tensor, s):
    # [(B), N_rays, N_pts]
    cdf = cdf_Phi_s(sdf, s)
    # [(B), N_rays, N_pts-1]
    # TODO: check sanity.
    opacity_alpha = (cdf[..., :-1] - cdf[..., 1:]) / (cdf[..., :-1] + 1e-10)
    opacity_alpha = torch.clamp_min(opacity_alpha, 0)
    return cdf, opacity_alpha


def sdf_to_w(sdf: torch.Tensor, s):
    device = sdf.device
    # [(B), N_rays, N_pts-1]
    cdf, opacity_alpha = sdf_to_alpha(sdf, s)

    # [(B), N_rays, N_pts]
    shifted_transparency = torch.cat(
        [
            torch.ones([*opacity_alpha.shape[:-1], 1], device=device),
            1.0 - opacity_alpha + 1e-10,
        ], dim=-1)
    
    # [(B), N_rays, N_pts-1]
    visibility_weights = opacity_alpha *\
        torch.cumprod(shifted_transparency, dim=-1)[..., :-1]

    return cdf, opacity_alpha, visibility_weights


def chunk_apply(func, x, dim, chunk_size):
        n = x.shape[0]
        # y = torch.zeros((n, dim), dtype=torch.float32).cuda()
        y = None
        for i in range(0, n, chunk_size):
            end_i = min(n, i + chunk_size)
            x_chunk = x[i:end_i]
            y_chunk = func(x_chunk)
            # y[i : end_i] += y_chunk
            if y is None:
                y = y_chunk
            else:
                y = torch.cat([y, y_chunk], dim=0)
        return y

def scatter_nd(index, updates, shape):
    new_tensor = torch.zeros(shape, dtype=updates.dtype, requires_grad=True, device=updates.device)
    new_tensor = new_tensor.index_add(0, index, updates)

    return new_tensor[None, ...]

class Surf(nn.Module):
    def __init__(self,
                 variance_init=0.05,
                 speed_factor=1.0,

                 input_ch=3,
                 W_geo_feat=0,
                 obj_bounding_radius=1.0,
                 z_dim = 3,
                 tracing_iter = 20,
                 noise_std = -1,
                 albedo_scale = 0.77,
                 albedo_bias = 0.03,
                 lvis_mode = 'None',

                 surface_cfg=dict(),
                 albedo_cfg=dict(),
                 brdf_cfg=dict(),
                 light_cfg=dict(),
                 render_cfg=dict(),
                 olat_cfg=dict(),
                 probe_cfg=dict(),
                 env_map_cfg=dict(),
                 loss_cfg=dict()):
        super().__init__()
        
        self.ln_s = nn.Parameter(data=torch.Tensor([-np.log(variance_init) / speed_factor]), requires_grad=True)
        self.speed_factor = speed_factor
        
        self.z_dim = z_dim
        self.tracing_iter = tracing_iter
        self.render_cfg = render_cfg
        self.noise_std = noise_std
        self.albedo_scale = albedo_scale
        self.albedo_bias = albedo_bias
        self.loss = loss_cfg
        self.lvis_mode = lvis_mode

        #------- surface network
        self.implicit_surface = ImplicitSurface(
            W_geo_feat=W_geo_feat, input_ch=input_ch, obj_bounding_size=obj_bounding_radius, **surface_cfg)
        
        #------- radiance network
        if W_geo_feat < 0:
            W_geo_feat = self.implicit_surface.W
        
        self.albedo = MLP(
            W_geo_feat=W_geo_feat, **albedo_cfg)
        self.brdf_z = MLP(
            W_geo_feat=W_geo_feat, **brdf_cfg)
        
        #------- Load BRDF model of NeRFactor
        #------- Pre-trained on MERL dataset by NeRFactor
        config_brdf = io_util.read_config('./brdf/lr1e-2.ini')
        self.brdf_model = BRDFModel(config_brdf)
        
        #------- Light model
        self.light = get_light(light_cfg)
        
        #------- Load multiple light models for visulization
        if self.render_cfg['relight_olat']:
            self.olat = get_light(olat_cfg)
        if self.render_cfg['relight_probes']:
            self.probe = get_light(probe_cfg)
        if self.render_cfg['relight_env_map']:
            self.env_map = get_light(env_map_cfg)


    def _cal_ldir(self, pts: torch.Tensor, light_posi: torch.Tensor):
        # [N_rays*N_pts, light_num, 3]
        surf2l = light_posi.reshape(1, -1, 3) - pts.permute(1, 0, 2)
        surf2l = F.normalize(surf2l, dim=-1)
        
        assert (torch.gt(torch.linalg.norm(surf2l, dim=-1), 0)).min(), \
            "Found zero-norm light directions"
        
        return surf2l
    
    def _cal_vdir(self, pts: torch.Tensor, cam_loc: torch.Tensor):
        # [1, N_rays*N_pts, 3]
        surf2c = cam_loc - pts
        surf2c = F.normalize(surf2c.squeeze(0), dim=-1)
        
        assert (torch.gt(torch.linalg.norm(surf2c, dim=-1), 0)).min(), \
            "Found zero-norm view directions"
        
        return surf2c

    @torch.no_grad()
    def _cal_lvis(self, pts: torch.Tensor, light_posi: torch.Tensor, method='ltracing', lvis_iter=20):
        # Official light visibility estimation method
        if method=='ltracing':
            # [light_num, N_rays*N_pts, 3]
            l2pts = pts - light_posi.reshape(-1, 1, 3)
            light_num, pts_num = l2pts.shape[0], l2pts.shape[1]
            
            # [light_num, N_rays*N_pts, 3]
            lxyz_flat = light_posi.reshape(-1, 1, 3).repeat(1, pts_num, 1)
            
            # [1, light_num*N_rays*N_pts]
            far = LA.norm(l2pts.reshape(-1, 3), ord=2, dim=-1).unsqueeze(0) - 1e-3 
            near = far - 0.5 # the 'near' is about out of the bounding sphere #TODO
            
            rayo_lvis = lxyz_flat.reshape(1, -1, 3)
            rayd_lvis = F.normalize(l2pts, dim=-1).reshape(1, -1, 3)
            
            _, _, mask_lvis = sphere_tracing_surface_points(self.implicit_surface, rayo_lvis, rayd_lvis, near, far, N_iters=lvis_iter)
            
            # [1, light_num*N_rays*N_pts]
            mask_lvis = 1. - mask_lvis.float() # 1 means visible, 0 means invisible
        
            # [N_lights, N_hit_rays]
            mask_lvis = mask_lvis.reshape(light_num, pts_num)
        
        # Implemented for ablation study: L-Tracing vs. Volume Rendering
        elif method=='volume_rendering':
            # init volume_rendering parameters
            lpix_chunk = 1
            lvis_near = 0.001
            lvis_far = 0.5
            fixed_s_recp = 1/100
            n_samples_coarse = 64
            n_samples_fine = 128
            device = pts.device
            
            # pts[1, N_ray_hit, 3]
            pts2l = light_posi.reshape(-1, 1, 3) - pts
            pts2l = F.normalize(pts2l, dim=-1)
            # pts2l [n_light, n_ray_hit, 3]
            light_num, pts_num = pts2l.shape[0], pts2l.shape[1]
            # [N_lights, N_hit_rays]
            mask_lvis = torch.zeros((light_num, pts_num), dtype=torch.float32, device = device)
            
            for i in range(0, light_num, lpix_chunk):                    
                end_i = min(light_num, i + lpix_chunk)
                
                # [lpix_chunk, N_ray_hit, 3]
                pts2l_chunk = pts2l[i:end_i, :, :]
                _t = torch.linspace(0, 1-1/n_samples_coarse, n_samples_coarse).float().to(device)
                d_coarse = lvis_near * (1 - _t) + lvis_far * _t
                d_coarse = d_coarse[None, None, :].repeat(1, pts_num, 1)
            
                # [(B), N_rays, N_samples, 3]
                pts_coarse = pts.unsqueeze(-2) + d_coarse.unsqueeze(-1) * pts2l_chunk.unsqueeze(-2)
                # query network to get sdf
                # [(B), N_rays, N_samples]
                sdf_coarse = self.implicit_surface.forward(pts_coarse)
                # [(B), N_rays, N_samples-1]
                *_, w_coarse = sdf_to_w(sdf_coarse, 1./fixed_s_recp)
                # Fine points
                # [(B), N_rays, N_importance]
                d_fine = rend_util.sample_pdf(d_coarse, w_coarse, n_samples_fine, det=False)
                # Gather points
                
                d_all = torch.cat([d_coarse, d_fine], dim=-1)
                d_all, d_sort_indices = torch.sort(d_all, dim=-1)
                
                sampled_lvis_pts = pts.unsqueeze(-2) + d_all.unsqueeze(-1) * pts2l_chunk.unsqueeze(-2)
                sdf_all = self.implicit_surface.forward(sampled_lvis_pts)
                
                *_, w_all = sdf_to_w(sdf_all, 1./fixed_s_recp)
                occu = torch.sum(w_all, -1)
                mask_lvis[i:end_i, :] = 1. - occu
        else:
            raise NotImplementedError
        
        return mask_lvis # [N_lights, N_hit_rays]
        
    def _eval_brdf_at(self,
                      pts2l: torch.Tensor,
                      pts2c: torch.Tensor,
                      normal: torch.Tensor, 
                      albedo: torch.Tensor, 
                      brdf_prop: torch.Tensor):
        """
            pts2l (torch.Tensor):       [pts_num, light_num, 3]
            pts2c (torch.Tensor):       [pts_num, 3]
            normal (torch.Tensor):      [pts_num, 3]
            albedo (torch.Tensor):      [pts_num, 3]
            brdf_prop (torch.Tensor):   [pts_num, 3]
        """
        
        z = brdf_prop 
        world2local = geom_util.gen_world2local(normal)
            
        # Transform directions into local frames
        vdir = torch.einsum('jkl,jl->jk', world2local, pts2c)
        ldir = torch.einsum('jkl,jnl->jnk', world2local, pts2l)
        
        # Directions to Rusink.
        ldir_flat = ldir.reshape(-1, 3)
        vdir_rep = torch.tile(vdir[:, None, :], (1, ldir.shape[1], 1))
        vdir_flat = vdir_rep.reshape(-1, 3)
        rusink = geom_util.dir2rusink(ldir_flat, vdir_flat) # NLx3
        
        # Repeat BRDF Z
        z_rep = torch.tile(z[:, None, :], (1, ldir.shape[1], 1))
        z_flat = z_rep.reshape(-1, self.z_dim)
        
        # Mask out back-lit directions for speed
        local_normal = torch.tensor((0, 0, 1), dtype=torch.float32, device=normal.device)
        local_normal = local_normal.reshape(3, 1)
        cos = ldir_flat @ local_normal
        front_lit = cos.reshape(-1,) > 0
        rusink_fl = rusink[front_lit]
        z_fl = z_flat[front_lit]
        
        embedder = self.brdf_model.embedder
        
        def chunk_func(rusink_z):
            rusink, z = rusink_z[:, :3], rusink_z[:, 3:]
            rusink_embed = embedder(rusink)
            z_rusink = torch.cat((z, rusink_embed), dim=1)
            brdf = self.brdf_model.net(z_rusink)
            return brdf

        rusink_z = torch.cat((rusink_fl, z_fl), 1)
        brdf_fl = chunk_apply(chunk_func, rusink_z, 1, 65536)
        
        # Put front-lit BRDF values back into an all-zero flat tensor, ...
        if brdf_fl is not None:
            brdf_flat = scatter_nd(torch.where(front_lit)[0], brdf_fl, (front_lit.shape[0], 1))
        else:
            brdf_flat = torch.zeros_like(cos)

        # and then reshape the resultant flat tensor
        spec = brdf_flat.reshape(ldir.shape[0], ldir.shape[1], 1)
        spec = torch.tile(spec, (1, 1, 3)) # becasue they are achromatic
        # Combine specular and Lambertian components
        brdf = albedo[:, None, :] / torch.tensor(np.pi, dtype=torch.float32, device=albedo.device) + spec
        
        return brdf # [N_rays*N_pts, light_num, 3]
    
    def _render_pts(self, light_vis, brdf, l, n,
                    relight_olat=False, relight_probes=False, relight_env_map=False,
                    white_light_override=False, white_lvis_override=False,
                    linear2srgb=True):
        light = self.light.rgb

        if white_light_override:
            light = np.ones_like(self.light.rgb)
        if white_lvis_override:
            light_vis = np.ones_like(light_vis)

        cos = torch.einsum('ijk,ik->ij', l, n) # NxL
        # Areas for intergration
        areas = self.light.lareas.reshape(1, -1, 1) # 1xLx1
        # NOTE: unnecessary if light_vis already encodes it, but won't hurt
        front_lit = (cos > 0).float()
        lvis = front_lit * light_vis # NxL

        def integrate(light):
            light_flat = light.reshape(-1, 3) # Lx3
            light = lvis[:, :, None] * light_flat[None, :, :] # NxLx3
            light_pix_contrib = brdf * light * cos[:, :, None] * areas # NxLx3
            rgb = light_pix_contrib.sum(1) # Nx3
            rgb = torch.clip(rgb, min=0., max=1.) 
            
            # Colorspace transform
            if linear2srgb:
                rgb = img_util.linear2srgb(rgb)
            return rgb

        # ------ Render under original lighting
        rgb = integrate(light)
        
        # ------ Continue to render EnvMap results
        rgb_env_map = None
        if relight_env_map:
            rgb_env_map = integrate(self.env_map.rgb.to(rgb.device))

        # ------ Continue to render OLAT-relit results
        rgb_olat = None
        if relight_olat:
            rgb_olat = []
            with torch.no_grad():
                for _, light in self.olat.novel_olat.items():
                    rgb_relit = integrate(light)
                    rgb_olat.append(rgb_relit)
                rgb_olat = torch.cat([x[:, None, :] for x in rgb_olat], dim=1)
                assert not(torch.any(torch.isinf(rgb_olat)) and torch.any(torch.isnan(rgb_olat))), "inf or nan in rgb_olat"

        # ------ Continue to render light probe-relit results
        rgb_probes = None
        if relight_probes:
            rgb_probes = []
            with torch.no_grad():
                probe_intensity = 0
                for _, light in self.probe.novel_probes.items():
                    probe_intensity += img_util.rgb2gray(light).sum()
                light_scale = img_util.rgb2gray(self.light.rgb).sum() * 8 / probe_intensity
                light_scale = light_scale ** 2
                for _, light in self.probe.novel_probes.items():
                    rgb_relit = integrate(light * light_scale)
                    rgb_probes.append(rgb_relit)
                rgb_probes = torch.cat([x[:, None, :] for x in rgb_probes], dim=1)
                assert not(torch.any(torch.isinf(rgb_probes)) and torch.any(torch.isnan(rgb_probes))), "inf or nan in rgb_probes"

        return rgb, rgb_olat, rgb_probes, rgb_env_map # Nx3
    
    def render_surf_pts(self, x_hit, albedo_hit, brdf_z_hit, normal_hit, rayo_hit):
        light_position = self.light.radius * F.normalize(self.light.lxyz, dim=-1)
        
        # [hit_rays, light_num, 3]
        pts2l = self._cal_ldir(x_hit, light_position) 
        # [1, hit_rays, 3]
        pts2c = self._cal_vdir(x_hit, rayo_hit) 
        
        # [light_num, hit_rays]
        light_vis = self._cal_lvis(x_hit, light_position, method=self.lvis_mode, lvis_iter=self.tracing_iter)
        # [hit_rays, light_num]
        light_vis = light_vis.permute(1,0)
        
        # [N_rays, light_num, 3]
        brdf = self._eval_brdf_at(pts2l, pts2c, normal_hit, albedo_hit, brdf_z_hit)
        
        # [B, N_rays_hit, N_lights, 3]
        rgb_hit, rgb_olat_hit, rgb_probe_hit, rgb_env_map_hit = self._render_pts(light_vis, brdf, 
                                                                                 pts2l, normal_hit,
                                                                                 **self.render_cfg)
        
        # [N_rays, 1] average on all lights
        light_vis = light_vis.mean(dim=1, keepdim=True)

        return rgb_hit, rgb_olat_hit, rgb_probe_hit, rgb_env_map_hit, light_vis
            
    def forward_radiance(self, x: torch.Tensor, rayo: torch.Tensor, rayd: torch.Tensor):
        """
            x (torch.Tensor): [B, N_rays*N_pts,3]
            rayo (torch.Tensor): [B, N_rays*N_pts, 3]
            rayd (torch.Tensor): [B, N_rays*N_pts, 3]
        """
        _, _, geometry_feature = self.implicit_surface.forward_with_nablas(x)

        albedo = self.albedo(x, geometry_feature)   # [1, N_rays*N_pts, 3]
        albedo = self.albedo_scale * albedo + self.albedo_bias
        brdf_prop = self.brdf_z(x, geometry_feature)# [1, N_rays*N_pts, 3]

        if self.noise_std > 0:
            x_noise = torch.randn(x.shape, device=x.device) * self.noise_std
            _, _, geometry_feature_noise = self.implicit_surface.forward_with_nablas(x_noise)
            albedo_jitter = self.albedo(x_noise, geometry_feature_noise)   # [1, N_rays*N_pts, 3]
            albedo_jitter = self.albedo_scale * albedo_jitter + self.albedo_bias
            brdf_prop_jitter = self.brdf_z(x_noise, geometry_feature_noise)# [1, N_rays*N_pts, 3]
            return albedo, brdf_prop, albedo_jitter, brdf_prop_jitter
            
        return albedo, brdf_prop

    def forward_s(self):
        return torch.exp(self.ln_s * self.speed_factor)

    def forward(self, x: torch.Tensor, view_dirs: torch.Tensor):
        sdf, nablas, geometry_feature = self.implicit_surface.forward_with_nablas(x)
        albedo = self.albedo.forward(x, geometry_feature)
        brdf_z = self.brdf_z.forward(x, geometry_feature)
        return albedo, brdf_z, sdf, nablas
    
    def vis_batch(self, ret, gamma=2.2):
        # same as nerfactor, visualize albedo and BRDF
        albedo = ret['albedo']
        brdf = ret['brdf']
        albedo = albedo ** (1 / gamma)
        
        def _brdf_prop_as_img(brdf_prop):
            """Z in learned BRDF.

            Input and output are both NumPy arrays, not tensors.
            """
            # Get min. and max. from seen BRDF Zs
            seen_z = self.brdf_model.latent_code._z
            seen_z = seen_z.numpy()
            seen_z_rgb = seen_z[:, :3]
            min_ = seen_z_rgb.min()
            max_ = seen_z_rgb.max()
            range_ = max_ - min_
            assert range_ > 0, "Range of seen BRDF Zs is 0"
            
            # Clip predicted values and scale them to [0, 1]
            z_rgb = brdf_prop[:, :, :3]
            z_rgb = torch.clip(z_rgb, min_, max_)
            z_rgb = (z_rgb - min_) / range_
            return z_rgb

        brdf = _brdf_prop_as_img(brdf)
        
        vis = {'albedo': albedo,
               'brdf': brdf}
        
        return vis


def volume_render(
    rays_o, 
    rays_d,
    model: Surf,
    obj_bounding_radius=1.0,
    batched = False,
    
    # render algorithm config
    calc_normal = False,
    rayschunk = 65536,
    netchunk = 1048576,
    white_bkgd = False,
    near_bypass: Optional[float] = None,
    far_bypass: Optional[float] = None,

    # render function config
    detailed_output = True,
    show_progress = False,
    
    # scale albedo if use synthetic dataset with gt albedos
    albedo_scales = None,

    **dummy_kwargs  # just place holder
):
    """
    input: 
        rays_o: [(B,) N_rays, 3]
        rays_d: [(B,) N_rays, 3] NOTE: not normalized. contains info about ratio of len(this ray)/len(principle ray)
    """
    device = rays_o.device
    if batched:
        DIM_BATCHIFY = 1
        B = rays_d.shape[0]  # batch_size
        flat_vec_shape = [B, -1, 3]
    else:
        DIM_BATCHIFY = 0
        flat_vec_shape = [-1, 3]

    rays_o = torch.reshape(rays_o, flat_vec_shape).float()
    rays_d = torch.reshape(rays_d, flat_vec_shape).float()
    # Normalize the rays direction for rendering
    rays_d = F.normalize(rays_d, dim=-1)
    
    batchify_query = functools.partial(train_util.batchify_query, chunk=netchunk, dim_batchify=DIM_BATCHIFY)
    
    # ------------------
    # Render a ray chunk
    # ------------------
    def render_rayschunk(rays_o: torch.Tensor, rays_d: torch.Tensor):
        # rays_o: [(B), N_rays, 3]
        # rays_d: [(B), N_rays, 3]
        
        # [(B), N_rays] x 2
        near, far = rend_util.near_far_from_sphere(rays_o, rays_d, r=obj_bounding_radius)
        if near_bypass is not None:
            near = near_bypass * torch.ones_like(near).to(device)
        if far_bypass is not None:
            far = far_bypass * torch.ones_like(far).to(device)
        
        # compute intersect points and masks
        with torch.no_grad():
            depth, x, intersect_mask = sphere_tracing_surface_points(model.implicit_surface, rays_o, rays_d, N_iters=50)
            
        # [(B), N_rays_hit, 3]
        intersect_mask = intersect_mask.reshape(-1)
        x_hit = x[:, intersect_mask, :]
        rayo_hit = rays_o[:, intersect_mask, :]
        rayd_hit = rays_d[:, intersect_mask, :]
        
        if intersect_mask.any():
            # make sure the surface pts x is on the outside of the implicit surface
            x_hit = x_hit - F.normalize(rayd_hit, dim=-1) * 2e-3
            
            # compute normal
            sdf, nablas, _ = model.implicit_surface.forward_with_nablas(x_hit)
            normals_hit = F.normalize(nablas, dim=-1)
            normals_hit = normals_hit.squeeze(0)
            # if sdf.min()<0:
            #     print("Find {} point inside the surface".format((sdf<0).sum()))
            
            if model.noise_std > 0:
                albedo, brdf_z, albedo_jitter, brdf_z_jitter = batchify_query(model.forward_radiance,
                                        x_hit.unsqueeze(-2),
                                        rayo_hit.unsqueeze(-2),
                                        rayd_hit.unsqueeze(-2))
            else:
                albedo, brdf_z = batchify_query(model.forward_radiance,
                                        x_hit.unsqueeze(-2),
                                        rayo_hit.unsqueeze(-2),
                                        rayd_hit.unsqueeze(-2))
    
            if albedo_scales is not None:
                albedo = albedo_scales.reshape(1, 3) * albedo

            albedo_hit = albedo.squeeze(-2)
            albedo_hit = albedo_hit.squeeze(0)
            brdfprop_hit = brdf_z.squeeze(-2)
            brdfprop_hit = brdfprop_hit.squeeze(0)
            if model.noise_std > 0:
                albedo_jitter_hit = albedo_jitter.squeeze(-2)
                albedo_jitter_hit = albedo_jitter_hit.squeeze(0)
                brdf_z_jitter_hit = brdf_z_jitter.squeeze(-2)
                brdf_z_jitter_hit = brdf_z_jitter_hit.squeeze(0)
                
            # ---------------------------------------
            # Surface rendering, similar to nerfactor
            # ---------------------------------------
            rgb_hit, rgb_olat_hit, rgb_probe_hit, rgb_env_map_hit, lvis_hit = model.render_surf_pts(x_hit, albedo_hit, brdfprop_hit, normals_hit, rayo_hit)

            hit_where = torch.where(intersect_mask)[0]
            hit_num = intersect_mask.shape[0]
            rgb_map = scatter_nd(hit_where, rgb_hit, (hit_num, 3))
            lvis_map = scatter_nd(hit_where, lvis_hit, (hit_num, 1))
            normals_map = scatter_nd(hit_where, normals_hit, (hit_num, 3))
            albedo_map = scatter_nd(hit_where, albedo_hit, (hit_num, 3))
            brdfprop_map = scatter_nd(hit_where, brdfprop_hit, (hit_num, 3))
            
            if model.noise_std > 0:
                albedo_jitter_map = scatter_nd(hit_where, albedo_jitter_hit, (hit_num, 3))
                brdf_z_jitter_map = scatter_nd(hit_where, brdf_z_jitter_hit, (hit_num, 3))
            
            olat_map, probe_map, env_map = None, None, None
            if model.render_cfg['relight_olat']:
                olat_map = scatter_nd(hit_where, rgb_olat_hit, (hit_num, rgb_olat_hit.shape[1], 3))
            if model.render_cfg['relight_probes']:
                probe_map = scatter_nd(hit_where, rgb_probe_hit, (hit_num, rgb_probe_hit.shape[1], 3))
            if model.render_cfg['relight_env_map']:
                env_map = scatter_nd(hit_where, rgb_env_map_hit, (hit_num, 3))

        else:
            rgb_map = torch.zeros_like(x)
            normals_map = torch.zeros_like(x)
            albedo_map = torch.zeros_like(x)
            brdfprop_map = torch.zeros_like(x)
            lvis_map = torch.zeros_like(x[..., 0, None])
            
            if model.noise_std > 0:
                albedo_jitter_map = torch.zeros_like(x)
                brdf_z_jitter_map = torch.zeros_like(x)
            
            olat_map, probe_map, env_map = None, None, None
            if model.render_cfg['relight_olat']:
                olat_map = torch.zeros_like(x).unsqueeze(-2).repeat(1, 1, 512, 1)
            if model.render_cfg['relight_probes']:
                probe_map = torch.zeros_like(x).unsqueeze(-2).repeat(1, 1, 8, 1)
            if model.render_cfg['relight_env_map']:
                env_map = torch.zeros_like(x)

        # SDF depth
        depth_map = depth
        acc_map = intersect_mask.unsqueeze(0).float()
        
        if white_bkgd:
            rgb_map = rgb_map + (1.0 - acc_map[..., None])

        ret_i = OrderedDict([
            ('rgb', rgb_map),               # [(B), N_rays, 3]
            ('albedo', albedo_map),         # [(B), N_rays, 3]
            ('brdf', brdfprop_map),         # [(B), N_rays, 3]
            ('depth_volume', depth_map),    # [(B), N_rays]
            ('mask_volume', acc_map),       # [(B), N_rays]
        ])
        
        if olat_map is not None:
            ret_i['olat_map'] = olat_map
        if probe_map is not None:
            ret_i['probe_map'] = probe_map
        if env_map is not None:
            ret_i['env_map'] = env_map
        if calc_normal:
            ret_i['normals_volume'] = normals_map # SDF surface normal

        if detailed_output:
            ret_i['light_visibility'] = lvis_map
            if model.noise_std > 0:
                ret_i['albedo_jitter_map'] = albedo_jitter_map
                ret_i['brdf_z_jitter_map'] = brdf_z_jitter_map

        return ret_i
        
    ret = {}
    for i in tqdm(range(0, rays_o.shape[DIM_BATCHIFY], rayschunk), disable=not show_progress):
        ret_i = render_rayschunk(
            rays_o[:, i:i+rayschunk] if batched else rays_o[i:i+rayschunk],
            rays_d[:, i:i+rayschunk] if batched else rays_d[i:i+rayschunk],
        )
        for k, v in ret_i.items():
            if k not in ret:
                ret[k] = []
            ret[k].append(v)
    for k, v in ret.items():
        ret[k] = torch.cat(v, DIM_BATCHIFY)
    
    return ret['rgb'], ret['depth_volume'], ret

class SingleRenderer(nn.Module):
    def __init__(self, model: Surf):
        super().__init__()
        self.model = model

    def forward(self, rays_o, rays_d, **kwargs):
        return volume_render(rays_o, rays_d, self.model, **kwargs)


class Trainer(nn.Module):
    def __init__(self, model: Surf, device_ids=[0], batched=True):
        super().__init__()
        self.model = model
        self.renderer = SingleRenderer(model)
        if len(device_ids) > 1:
            self.renderer = nn.DataParallel(self.renderer, device_ids=device_ids, dim=1 if batched else 0)
        self.device = device_ids[0]
        
    def forward(self, 
            args,
            indices,
            model_input,
            ground_truth,
            render_kwargs_train: dict,
            it: int,
            device='cuda'):

        intrinsics = model_input["intrinsics"].to(device)
        c2w = model_input['c2w'].to(device)
        H = render_kwargs_train['H']
        W = render_kwargs_train['W']
        rays_o, rays_d, select_inds = rend_util.get_rays(args.data.type,
            c2w, intrinsics, H, W, N_rays=args.data.N_rays)
        # [B, N_rays, 3]
        target_rgb = torch.gather(ground_truth['rgb'].to(device), 1, torch.stack(3*[select_inds],-1))
        
        rgb, _, extras = self.renderer(rays_o, rays_d, detailed_output=True, **render_kwargs_train)
        
        hit_mask = extras['mask_volume'][..., None]
        mask_volume: torch.Tensor = extras['mask_volume']
        # # NOTE: when predicted mask is close to 1 but GT is 0, exploding gradient.
        # # mask_volume = torch.clamp(mask_volume, 1e-10, 1-1e-10)
        mask_volume = torch.clamp(mask_volume, 1e-3, 1-1e-3)
        extras['mask_volume_clipped'] = mask_volume

        losses = OrderedDict()

        # [B, N_rays, 3]
        losses['loss_img'] = F.mse_loss(rgb * hit_mask, target_rgb * hit_mask, reduction='none')

        if self.model.noise_std > 0:
            smooth_loss_func = F.l1_loss if True else F.mse_loss
            albedo_smooth_loss = smooth_loss_func(extras['albedo'] * hit_mask, extras['albedo_jitter_map'] * hit_mask) # N
            brdf_smooth_loss = smooth_loss_func(extras['brdf'] * hit_mask, extras['brdf_z_jitter_map'] * hit_mask) # N
            losses['smooth_loss'] = self.model.loss['brdf_smooth_weight'] * brdf_smooth_loss + \
                self.model.loss['albedo_smooth_weight'] * albedo_smooth_loss
        
        assert self.model.light.modelname == 'source', \
            'The trainable light source must be source'
        # Spatial TV penalty, same as nerfactor
        light = self.model.light.rgb
        light_tv_weight = self.model.loss['light_tv_weight']
        light_achro_weight = self.model.loss['light_achro_weight']
        if light_tv_weight > 0:
            losses['light_tv'] = 0.
            dx = light - torch.roll(light, 1, 1)
            dy = light - torch.roll(light, 1, 0)
            tv = torch.sum(dx ** 2 + dy ** 2)
            losses['light_tv'] += light_tv_weight * tv
        # Cross-channel TV penalty, same as nerfactor
        if light_achro_weight > 0:
            losses['light_achro'] = 0.  
            dc = light - torch.roll(light, 1, 2)
            tv = torch.sum(dc ** 2)
            losses['light_achro'] += light_achro_weight * tv

        losses['loss_img'] = losses['loss_img'].mean()

        loss = 0
        for k, v in losses.items():
            losses[k] += losses[k] * args.training.loss_scale_factor
            loss += losses[k]

        losses['total'] = loss
        extras['scalars'] = {'1/s': 1./self.model.forward_s().data}
        extras['select_inds'] = select_inds
        
        return OrderedDict(
            [('losses', losses),
             ('extras', extras)])


def get_model(args):
    model_config = {
        'obj_bounding_radius':  args.model.obj_bounding_radius,
        'W_geo_feat':       args.model.setdefault('W_geometry_feature', 256),
        'speed_factor':     args.training.setdefault('speed_factor', 1.0),
        'variance_init':    args.model.setdefault('variance_init', 0.05),
        'z_dim':            args.model.setdefault('z_dim', 3),
        'tracing_iter':     args.model.setdefault('tracing_iter', 20),
        'noise_std':        args.model.noise_std,
        'albedo_scale':     args.model.albedo_scale,
        'albedo_bias':      args.model.albedo_bias,
        'lvis_mode':        args.lvis_mode,
    }
    
    model_config['surface_cfg'] = args.model.surface
    model_config['albedo_cfg'] = args.model.albedo
    model_config['brdf_cfg'] = args.model.brdf
    model_config['light_cfg'] = args.model.light.source
    model_config['olat_cfg'] = args.model.light.olat
    model_config['probe_cfg'] = args.model.light.probe
    model_config['env_map_cfg'] = args.model.light.env_map
    model_config['render_cfg'] = args.model.render
    model_config['loss_cfg'] = args.loss
    
    model = Surf(**model_config)
    
    ## render kwargs
    render_kwargs_train = {      
        'obj_bounding_radius': args.model.setdefault('obj_bounding_radius', 1.0),
        'batched': args.data.batch_size is not None,
        'perturb': args.model.setdefault('perturb', True),   # config whether do stratified sampling
        'white_bkgd': args.model.setdefault('white_bkgd', False),
    }
    render_kwargs_test = copy.deepcopy(render_kwargs_train)
    render_kwargs_train['debug'] = args.training.debug
    
    render_kwargs_test['rayschunk'] = args.data.val_rayschunk
    render_kwargs_test['perturb'] = False
    
    trainer = Trainer(model, device_ids=args.device_ids, batched=render_kwargs_train['batched'])
    
    return model, trainer, render_kwargs_train, render_kwargs_test, trainer.renderer

if __name__=='__main__':
    parser = io_util.create_args_parser()
    args, unknown = parser.parse_known_args()
    config = io_util.load_config(args, unknown)
    model, trainer, render_kwargs_train, render_kwargs_test, renderer_fn  = get_model(config)
    model = model.cuda()
    rayo = torch.tensor([1.,1.,1.])[None, None, :].repeat(1,16,1).cuda()
    rayd = torch.rand(1,16,3).cuda()
    renderer_fn(rayo, rayd, **render_kwargs_train)