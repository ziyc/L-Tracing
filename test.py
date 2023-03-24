import os
import cv2
import functools
import torch
from tqdm import tqdm
from dataio import get_data
from models.light import get_light
from models.frameworks import get_model
from utils import rend_util, train_util, io_util, img_util
from utils.print_fn import log
from utils.logger import Logger
from utils.checkpoints import CheckpointIO
from utils.io_util import load_rgb
from utils.test_utils import get_albedo_scale


def main_function(args, config_path):
    device = torch.device('cuda:0')
    exp_dir = args.training.exp_dir
    print('=> restore from dir: ', exp_dir)
    
    down_scale = args.data.get('val_downscale', 4.0)
    vis_name = 'val_downscale={}'.format(down_scale)

    # logger
    logger = Logger(log_dir=exp_dir, img_dir=os.path.join(exp_dir, vis_name), is_train=False)
    log.info("=> Experiments dir: {}".format(exp_dir))
    
    # activate light probes to relight with light probe images
    args.model.render.relight_probes = True
        
    # Create model
    model, _, _, render_kwargs_test, volume_render_fn = get_model(args)
    model.to(device)
    print(model)
    log.info("=> Nerf params: " + str(train_util.count_trainable_parameters(model)))
    
    # Load Dataset
    val_dataset = get_data(args, return_val=True, only_val=True, val_downscale=args.data.get('val_downscale', 4.0))
    render_kwargs_test['H'] = val_dataset.H
    render_kwargs_test['W'] = val_dataset.W

    # Register modules to checkpoint and oad checkpoints
    checkpoint_io = CheckpointIO(checkpoint_dir=os.path.join(args.training.exp_dir, 'ckpts'))
    checkpoint_io.register_modules(model=model)
    checkpoint_io.load_file(args.training.ckpt_file,
                            ignore_keys=args.training.ckpt_ignore_keys,
                            only_use_keys=args.training.ckpt_only_use_keys,
                            map_location=device)
    
    if args.model.light.probe.light_h != args.model.light.source.light_h:
        print('=> changing the light resolution')
        light_dict = args.model.light.source
        light_dict['light_h'] = args.model.light.probe.light_h
        model.light = get_light(args.model.light.source)
        model.light.to(device)
    
    # if run NeRF synthesis dataset, rescale the albedo
    # TODO: cite nerfactor for the same application
    albedo_scale = get_albedo_scale(exp_dir)
    albedo_scale = albedo_scale.to(device)

    log.info('=> Start testing..., in {}'.format(exp_dir))
    for idx in tqdm(range(8)):
        with torch.no_grad():
            (val_ind, val_in, val_gt) = val_dataset[idx]
            val_ind = torch.tensor(val_ind)
            
            intrinsics = val_in["intrinsics"][None, ...].to(device)
            c2w = val_in['c2w'][None, ...].to(device)
            
            # N_rays=-1 for rendering full image
            rays_o, rays_d, _ = rend_util.get_rays(args.data.type,
                                                   c2w, intrinsics,
                                                   render_kwargs_test['H'],
                                                   render_kwargs_test['W'],
                                                   N_rays=-1)
            target_rgb = val_gt['rgb'][None, ...].to(device)
            
            rgb, depth_v, ret = volume_render_fn(rays_o, rays_d, calc_normal=True,
                                                 detailed_output=True, 
                                                 albedo_scales=albedo_scale,
                                                 **render_kwargs_test)
            
            to_img = functools.partial(rend_util.lin2img,
                                       H=render_kwargs_test['H'], W=render_kwargs_test['W'],
                                       batched=render_kwargs_test['batched'])
            cat_img = functools.partial(rend_util.lin2img,
                                        H=render_kwargs_test['H'] * 2, W=render_kwargs_test['W'] ,
                                        batched=render_kwargs_test['batched'])
            vis = model.vis_batch(ret)
            
            if args.data.type=='nerf':
                is_synthetic = True
                vis_masks = val_gt['vis_masks'][None, ..., None].to(device) # [1, H*W, 1]
            is_synthetic = False
    
            # target_rgb [1, H*W, 3] rgb [1, H*W, 3]
            if is_synthetic:
                # save and vis novel view rgb
                masked_pred_rgb = img_util.alpha_blend(rgb, vis_masks , torch.ones_like(rgb))
                masked_gt_rgb = img_util.alpha_blend(target_rgb, vis_masks , torch.ones_like(target_rgb))
                
                img_A = (masked_pred_rgb.reshape(render_kwargs_test['H'], render_kwargs_test['W'], 3) * 255).cpu().numpy().astype('uint8')
                img_B = (masked_gt_rgb.reshape(render_kwargs_test['H'], render_kwargs_test['W'], 3) * 255).cpu().numpy().astype('uint8')
                
                masked_psnr = cv2.PSNR(img_A, img_B)
                
                vis_masked_rgb = torch.cat((masked_gt_rgb, masked_pred_rgb), 1)
                logger.add_imgs(cat_img(vis_masked_rgb), 'mask_rgb', metric=masked_psnr, batch_idx=val_ind)
                logger.add_imgs(to_img(masked_pred_rgb), 'pred_rgb', batch_idx=val_ind)
                logger.add_imgs(to_img(target_rgb), 'gt_rgb', batch_idx=val_ind)
                
                # save and vis albedo
                pred_albedo = vis['albedo'] # [1, H*W, 3]
                true_albedo = val_gt['albedo'].to(device).unsqueeze(0) # [1, H*W, 3]

                masked_pred_albedo = img_util.alpha_blend(pred_albedo, vis_masks , torch.ones_like(pred_albedo))
                masked_true_albedo = img_util.alpha_blend(true_albedo, vis_masks , torch.ones_like(true_albedo))

                pred_img = (masked_pred_albedo.reshape(render_kwargs_test['H'], render_kwargs_test['W'], 3) * 255).cpu().numpy().astype('uint8')
                true_img = (masked_true_albedo.reshape(render_kwargs_test['H'], render_kwargs_test['W'], 3) * 255).cpu().numpy().astype('uint8')

                masked_albedo_psnr = cv2.PSNR(pred_img, true_img)

                vis_gt_albedo = torch.cat((masked_true_albedo, masked_pred_albedo), 1)
                logger.add_imgs(cat_img(vis_gt_albedo), 'mask_albedo', metric=masked_albedo_psnr, batch_idx=val_ind)
                logger.add_imgs(to_img(masked_pred_albedo), 'pred_albedo', batch_idx=val_ind)
                logger.add_imgs(to_img(masked_true_albedo), 'gt_albedo', batch_idx=val_ind)
                
                # save and vis brdf
                masked_brdf = img_util.alpha_blend(vis['brdf'], vis_masks , torch.ones_like(vis['brdf']))
                logger.add_imgs(to_img(masked_brdf), 'masked_brdf', batch_idx=val_ind)
                
                # save and vis normal
                vis_normal = ret['normals_volume']/2.+0.5
                masked_normal = img_util.alpha_blend(vis_normal, vis_masks , torch.ones_like(vis_normal))
                logger.add_imgs(to_img(masked_normal), 'masked_normals', batch_idx=val_ind)
                
                # save and vis light visibility
                masked_lvis = img_util.alpha_blend(ret['light_visibility'], vis_masks , torch.ones_like(ret['light_visibility']))
                logger.add_imgs(to_img(masked_lvis), 'masked_lvis', batch_idx=val_ind)
            else:
                # save and vis novel view rgb
                img_A = (rgb.reshape(render_kwargs_test['H'], render_kwargs_test['W'], 3) * 255).cpu().numpy().astype('uint8')
                img_B = (target_rgb.reshape(render_kwargs_test['H'], render_kwargs_test['W'], 3) * 255).cpu().numpy().astype('uint8')
                masked_psnr = cv2.PSNR(img_A, img_B)
                
                vis_rgb = torch.cat((target_rgb, rgb), 1)
                logger.add_imgs(cat_img(vis_rgb), 'vis_rgb', metric=masked_psnr, batch_idx=val_ind)
                logger.add_imgs(to_img(rgb), 'pred_rgb', batch_idx=val_ind)
                logger.add_imgs(to_img(target_rgb), 'gt_rgb', batch_idx=val_ind)
                
                # save and vis albedo
                logger.add_imgs(to_img(vis['albedo']), 'albedo', batch_idx=val_ind)
                
                # save and vis brdf
                logger.add_imgs(to_img(vis['brdf']), 'brdf', batch_idx=val_ind)
                
                # save and vis normal
                logger.add_imgs(to_img(ret['normals_volume']/2.+0.5), 'val_predicted_normals', batch_idx=val_ind)
                logger.add_imgs(to_img(ret['light_visibility']), 'val_predicted_lvis', batch_idx=val_ind)


            logger.add_imgs(to_img((depth_v/(depth_v.max()+1e-10)).unsqueeze(-1)), 'val_pred_depth_volume', batch_idx=val_ind)
            logger.add_imgs(to_img(ret['mask_volume'].unsqueeze(-1)), 'val_pred_mask_volume', batch_idx=val_ind)


            if 'depth_surface' in ret:
                logger.add_imgs(to_img((ret['depth_surface']/ret['depth_surface'].max()).unsqueeze(-1)), 'val_pred_depth_surface', batch_idx=val_ind)
            if 'mask_surface' in ret:
                logger.add_imgs(to_img(ret['mask_surface'].unsqueeze(-1).float()), 'val_predicted_mask', batch_idx=val_ind)
                
                
            if 'probe_map' in ret.keys():
                for probe_idx, probe_name in enumerate(model.probe.novel_probes):
                    pred_probe = ret['probe_map'][:, :, probe_idx, :]
                    gt_dir = os.path.join(args.data.data_dir,'val_{:03d}'.format(idx),'rgba_'+probe_name+'.png')
                    if os.path.exists(gt_dir):
                        raw_probe = load_rgb(gt_dir, args.data.val_downscale)
                        gt_probe = torch.from_numpy(raw_probe[:3, :, :]).to(pred_probe.device).permute(1,2,0).reshape(1,-1,3)
                        
                        masked_gt_probe = img_util.alpha_blend(gt_probe, vis_masks , torch.ones_like(gt_probe))
                        masked_pred_probe = img_util.alpha_blend(pred_probe, vis_masks , torch.ones_like(pred_probe))
                        
                        img_A = (masked_pred_probe.reshape(render_kwargs_test['H'], render_kwargs_test['W'], 3) * 255).cpu().numpy().astype('uint8')
                        img_B = (masked_gt_probe.reshape(render_kwargs_test['H'], render_kwargs_test['W'], 3) * 255).cpu().numpy().astype('uint8')
                        
                        masked_psnr = cv2.PSNR(img_A, img_B)
                        vis_masked_probe = torch.cat((masked_gt_probe, masked_pred_probe), 1)
                        logger.add_imgs(cat_img(vis_masked_probe), 'vis_'+probe_name, metric=masked_psnr, batch_idx=val_ind)
                        logger.add_imgs(to_img(masked_pred_probe), 'mask_'+probe_name, batch_idx=val_ind)
                        logger.add_imgs(to_img(masked_gt_probe), 'gt_mask_'+probe_name, batch_idx=val_ind)
                    else:
                        logger.add_imgs(to_img(pred_probe), 'vis_'+probe_name, metric=0, batch_idx=val_ind)
            
            
            if 'env_map' in ret.keys():
                env_map_relight_rgb = ret['env_map']
                env_map_relight_img = img_util.alpha_blend(env_map_relight_rgb, vis_masks , torch.ones_like(env_map_relight_rgb))
                env_map_relight_img = torch.cat([env_map_relight_img, vis_masks>=0.8], dim=-1)
                logger.add_imgs(to_img(env_map_relight_img), 'env_map', batch_idx=val_ind)
                                    

if __name__ == "__main__":
    # Arguments
    parser = io_util.create_args_parser()
    
    args, unknown = parser.parse_known_args()
    config_path = args.config
    config = io_util.load_config(args, unknown)
    main_function(config, config_path)