from models.frameworks import get_model
from models.base import get_optimizer, get_scheduler
from utils import rend_util, train_util, mesh_util, io_util, img_util
from utils.dist_util import get_local_rank, init_env, is_master, get_rank, get_world_size
from utils.print_fn import log
from utils.logger import Logger
from utils.checkpoints import CheckpointIO
from dataio import get_data
import os
import sys
import time
import functools
import cv2
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


def main_function(args):

    init_env(args)
    
    #----------------------------
    #-------- shortcuts ---------
    rank = get_rank()
    local_rank = get_local_rank()
    world_size = get_world_size()
    i_backup = int(args.training.i_backup // world_size) if args.training.i_backup > 0 else -1
    i_val = int(args.training.i_val // world_size) if args.training.i_val > 0 else -1
    i_val_mesh = int(args.training.i_val_mesh // world_size) if args.training.i_val_mesh > 0 else -1
    special_i_val_mesh = [int(i // world_size) for i in [3000, 5000, 7000]]
    exp_dir = args.training.exp_dir
    mesh_dir = os.path.join(exp_dir, 'meshes')
    
    device = torch.device('cuda', local_rank)

    # logger
    logger = Logger(
        log_dir=exp_dir,
        img_dir=os.path.join(exp_dir, 'imgs'),
        monitoring=args.training.get('monitoring', 'tensorboard'),
        monitoring_dir=os.path.join(exp_dir, 'events'),
        rank=rank, is_master=is_master(), multi_process_logging=(world_size > 1))

    log.info("=> Experiments dir: {}".format(exp_dir))

    if is_master():
        # backup codes
        io_util.backup(os.path.join(exp_dir, 'backup'))

        # save configs
        io_util.save_config(args, os.path.join(exp_dir, 'config.yaml'))
    
    dataset, val_dataset = get_data(args, return_val=True, val_downscale=args.data.get('val_downscale', 4.0))
    bs = args.data.get('batch_size', None)
    if args.ddp:
        train_sampler = DistributedSampler(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=bs)
        val_sampler = DistributedSampler(val_dataset)
        valloader = torch.utils.data.DataLoader(val_dataset, sampler=val_sampler, batch_size=bs)
    else:
        dataloader = DataLoader(dataset,
            batch_size=bs,
            shuffle=True,
            pin_memory=args.data.get('pin_memory', False))
        valloader = DataLoader(val_dataset,
            batch_size=1,
            shuffle=False)
    
    # Create model
    model, trainer, render_kwargs_train, render_kwargs_test, volume_render_fn = get_model(args)
    # if args.training.debug:
    #     model = DebugModule(model)
    model.to(device)
    print(model)
    log.info("=> Nerf params: " + str(train_util.count_trainable_parameters(model)))

    render_kwargs_train['H'] = dataset.H
    render_kwargs_train['W'] = dataset.W
    render_kwargs_test['H'] = val_dataset.H
    render_kwargs_test['W'] = val_dataset.W

    # build optimizer
    optimizer = get_optimizer(args, model)

    # checkpoints
    checkpoint_io = CheckpointIO(checkpoint_dir=os.path.join(exp_dir, 'ckpts'), allow_mkdir=is_master())
    if world_size > 1:
        dist.barrier()
    # Register modules to checkpoint
    checkpoint_io.register_modules(
        model=model,
        optimizer=optimizer,
    )

    # Load checkpoints
    load_dict = checkpoint_io.load_file(
        args.training.ckpt_file,
        ignore_keys=args.training.ckpt_ignore_keys,
        only_use_keys=args.training.ckpt_only_use_keys,
        map_location=device)

    logger.load_stats('stats.p')    # this will be used for plotting
    it = load_dict.get('global_step', 0)
    epoch_idx = load_dict.get('epoch_idx', 0)

    # pretrain if needed. must be after load state_dict, since needs 'is_pretrained' variable to be loaded.
    #---------------------------------------------
    #-------- init perparation only done in master
    #---------------------------------------------
    if is_master():
        pretrain_config = {'logger': logger}
        if 'lr_pretrain' in args.training:
            pretrain_config['lr'] = args.training.lr_pretrain
            if(model.implicit_surface.pretrain_hook(pretrain_config)):
                checkpoint_io.save(filename='latest.pt'.format(it), global_step=it, epoch_idx=epoch_idx)

    # Parallel training
    if args.ddp:
        trainer = DDP(trainer, device_ids=args.device_ids, output_device=local_rank, find_unused_parameters=True)

    time_100_cnt = 0
    # build scheduler
    scheduler = get_scheduler(args, optimizer, last_epoch=it-1)
    t0 = time.time()
    log.info('=> Start training..., it={}, lr={}, in {}'.format(it, optimizer.param_groups[0]['lr'], exp_dir))
    end = (it >= args.training.num_iters)
    with tqdm(range(args.training.num_iters), disable=not is_master()) as pbar:
        if is_master():
            pbar.update(it)
        while it <= args.training.num_iters and not end:
            try:
                if args.ddp:
                    train_sampler.set_epoch(epoch_idx)
                for (indices, model_input, ground_truth) in dataloader:
                    int_it = int(it // world_size)
                    #-------------------
                    # validate
                    #-------------------
                    if i_val > 0 and int_it % i_val == 0:
                        with torch.no_grad():
                            (val_ind, val_in, val_gt) = next(iter(valloader))

                            intrinsics = val_in["intrinsics"].to(device)
                            c2w = val_in['c2w'].to(device)

                            # N_rays=-1 for rendering full image
                            rays_o, rays_d, select_inds = rend_util.get_rays(args.data.type,
                                c2w, intrinsics, render_kwargs_test['H'], render_kwargs_test['W'], N_rays=-1)
                            target_rgb = val_gt['rgb'].to(device)

                            rgb, depth_v, ret = volume_render_fn(rays_o, rays_d, calc_normal=True, detailed_output=True, **render_kwargs_test)
                            # print(model.light.rgb)

                            to_img = functools.partial(
                                rend_util.lin2img,
                                H=render_kwargs_test['H'], W=render_kwargs_test['W'],
                                batched=render_kwargs_test['batched'])

                            if args.model.framework == 'Surf':
                                vis = model.vis_batch(ret)
                                if args.data.type=='nerf':
                                    is_synthetic = True
                                    vis_masks = val_gt['vis_masks'][..., None].to(device) # [1, H*W, 1]
                            is_synthetic = False
                                
                            cat_img = functools.partial(
                                rend_util.lin2img,
                                H=render_kwargs_test['H'] * 2, W=render_kwargs_test['W'] ,
                                batched=render_kwargs_test['batched'])

                            if is_synthetic:
                                # save and vis novel view rgb
                                masked_pred_rgb = img_util.alpha_blend(rgb, vis_masks , torch.zeros_like(rgb))
                                masked_gt_rgb = img_util.alpha_blend(target_rgb, vis_masks , torch.zeros_like(target_rgb))
                                
                                img_A = (masked_pred_rgb.reshape(render_kwargs_test['H'], render_kwargs_test['W'], 3) * 255).cpu().numpy().astype('uint8')
                                img_B = (masked_gt_rgb.reshape(render_kwargs_test['H'], render_kwargs_test['W'], 3) * 255).cpu().numpy().astype('uint8')
                                
                                masked_psnr = cv2.PSNR(img_A, img_B)
                                
                                vis_masked_rgb = torch.cat((masked_gt_rgb, masked_pred_rgb), 1)
                                logger.add_imgs(cat_img(vis_masked_rgb), 'mask_rgb', it, metric=masked_psnr, batch_idx=val_ind)
                                logger.add_imgs(to_img(masked_pred_rgb), 'pred_rgb', it, batch_idx=val_ind)
                                logger.add_imgs(to_img(target_rgb), 'gt_rgb', it, batch_idx=val_ind)
                                
                                # save and vis albedo
                                pred_albedo = vis['albedo'] # [1, H*W, 3]
                                true_albedo = val_gt['albedo'].to(device) # [1, H*W, 3]

                                masked_pred_albedo = img_util.alpha_blend(pred_albedo, vis_masks , torch.zeros_like(pred_albedo))
                                masked_true_albedo = img_util.alpha_blend(true_albedo, vis_masks , torch.zeros_like(true_albedo))

                                pred_img = (masked_pred_albedo.reshape(render_kwargs_test['H'], render_kwargs_test['W'], 3) * 255).cpu().numpy().astype('uint8')
                                true_img = (masked_true_albedo.reshape(render_kwargs_test['H'], render_kwargs_test['W'], 3) * 255).cpu().numpy().astype('uint8')

                                masked_albedo_psnr = cv2.PSNR(pred_img, true_img)

                                vis_gt_albedo = torch.cat((masked_true_albedo, masked_pred_albedo), 1)
                                logger.add_imgs(cat_img(vis_gt_albedo), 'mask_albedo', it, metric=masked_albedo_psnr, batch_idx=val_ind)
                                logger.add_imgs(to_img(masked_pred_albedo), 'pred_albedo', it, batch_idx=val_ind)
                                logger.add_imgs(to_img(masked_true_albedo), 'gt_albedo', it, batch_idx=val_ind)
                                
                                # save and vis brdf
                                masked_brdf = img_util.alpha_blend(vis['brdf'], vis_masks , torch.zeros_like(vis['brdf']))
                                logger.add_imgs(to_img(masked_brdf), 'masked_brdf', it, batch_idx=val_ind)
                                
                                # save and vis normal
                                vis_normal = ret['normals_volume']/2.+0.5
                                masked_normal = img_util.alpha_blend(vis_normal, vis_masks , torch.zeros_like(vis_normal))
                                logger.add_imgs(to_img(masked_normal), 'masked_normals', it, batch_idx=val_ind)
                                
                                # save and vis light visibility
                                masked_lvis = img_util.alpha_blend(ret['light_visibility'], vis_masks , torch.zeros_like(ret['light_visibility']))
                                logger.add_imgs(to_img(masked_lvis), 'masked_lvis', it, batch_idx=val_ind)
                            else:
                                # save and vis novel view rgb
                                img_A = (rgb.reshape(render_kwargs_test['H'], render_kwargs_test['W'], 3) * 255).cpu().numpy().astype('uint8')
                                img_B = (target_rgb.reshape(render_kwargs_test['H'], render_kwargs_test['W'], 3) * 255).cpu().numpy().astype('uint8')
                                masked_psnr = cv2.PSNR(img_A, img_B)
                                
                                vis_rgb = torch.cat((target_rgb, rgb), 1)
                                logger.add_imgs(cat_img(vis_rgb), 'vis_rgb', it, metric=masked_psnr, batch_idx=val_ind)
                                logger.add_imgs(to_img(rgb), 'pred_rgb', it, batch_idx=val_ind)
                                logger.add_imgs(to_img(target_rgb), 'gt_rgb', it, batch_idx=val_ind)

                                if args.model.framework == 'Surf':
                                    # save and vis albedo
                                    logger.add_imgs(to_img(vis['albedo']), 'albedo', it, batch_idx=val_ind)
                                    
                                    # save and vis brdf
                                    logger.add_imgs(to_img(vis['brdf']), 'brdf', it, batch_idx=val_ind)
                                    
                                    # save and vis normal
                                    logger.add_imgs(to_img(ret['normals_volume']/2.+0.5), 'val_predicted_normals', it, batch_idx=val_ind)
                                    logger.add_imgs(to_img(ret['light_visibility']), 'val_predicted_lvis', it, batch_idx=val_ind)
                                
                            logger.add_imgs(to_img((depth_v/(depth_v.max()+1e-10)).unsqueeze(-1)), 'val_pred_depth_volume', it, batch_idx=val_ind)
                            logger.add_imgs(to_img(ret['mask_volume'].unsqueeze(-1)), 'val_pred_mask_volume', it, batch_idx=val_ind)

                            if 'depth_surface' in ret:
                                logger.add_imgs(to_img((ret['depth_surface']/ret['depth_surface'].max()).unsqueeze(-1)), 'val_pred_depth_surface', it, batch_idx=val_ind)
                            if 'mask_surface' in ret:
                                logger.add_imgs(to_img(ret['mask_surface'].unsqueeze(-1).float()), 'val_predicted_mask', it, batch_idx=val_ind)
                            if hasattr(trainer, 'val'):
                                trainer.val(logger, ret, to_img, it, render_kwargs_test)

                            logger.add_imgs(to_img(ret['normals_volume']/2.+0.5), 'val_predicted_normals', it, batch_idx=val_ind)

                    # if i_val_light > 0 and int_it % i_val_light == 0:
                    #     # NOTE: remeber to delete after training
                    #     with torch.no_grad():
                    #         if args.model.framework == 'Surf':
                    #             resized = cv2.resize(model.light.rgb.cpu().numpy(), (256 * 2, 256),
                    #                                     interpolation=cv2.INTER_LINEAR)
                    #             light_map = img_util.tonemap(resized, method='gamma', gamma=4)
                    #             light_map = torch.from_numpy(light_map[None, ...]).permute(0, 3, 1, 2)
                    #             light_map = torch.clip(light_map, min=0., max=1.)
                    #             logger.add_imgs(light_map, 'light', it, batch_idx=0)
                    
                    #-------------------
                    # validate mesh
                    #-------------------
                    if is_master():
                        # NOTE: not validating mesh before 3k, as some of the instances of DTU for NeuS training will have no large enough mesh at the beginning.
                        if i_val_mesh > 0 and (int_it % i_val_mesh == 0 or int_it in special_i_val_mesh) and it != 0:
                            with torch.no_grad():
                                io_util.cond_mkdir(mesh_dir)
                                mesh_util.extract_mesh(
                                    model.implicit_surface, 
                                    filepath=os.path.join(mesh_dir, '{:08d}.ply'.format(it)),
                                    volume_size=args.data.get('volume_size', 2.0),
                                    show_progress=is_master())

                    if it >= args.training.num_iters:
                        end = True
                        break
                    
                    #-------------------
                    # train
                    #-------------------
                    start_time = time.time()
                    model.implicit_surface.eval()
                    
                    ret = trainer.forward(args, indices, model_input, ground_truth, render_kwargs_train, it)
                    
                    losses = ret['losses']
                    extras = ret['extras']
                    for k, v in losses.items():
                        # log.info("{}:{} - > {}".format(k, v.shape, v.mean().shape))
                        losses[k] = torch.mean(v)
                    
                    optimizer.zero_grad()

                    losses['total'].backward()
                    if True:
                        grad_norms = train_util.calc_grad_norm(model=model)
                    optimizer.step()
                    scheduler.step(it)  # NOTE: important! when world_size is not 1

                    #-------------------
                    # logging
                    #-------------------
                    # done every i_save seconds
                    if (args.training.i_save > 0) and (time.time() - t0 > args.training.i_save):
                        if is_master():
                            checkpoint_io.save(filename='time_{:08d}.pt'.format(args.training.i_save*time_100_cnt), global_step=it, epoch_idx=epoch_idx)
                        # this will be used for plotting
                        time_100_cnt += 1
                        logger.save_stats('stats.p')
                        t0 = time.time()
                    
                    if is_master():
                        #----------------------------------------------------------------------------
                        #------------------- things only done in master -----------------------------
                        #----------------------------------------------------------------------------
                        pbar.set_postfix(lr=optimizer.param_groups[0]['lr'], loss_total=losses['total'].item(), loss_img=losses['loss_img'].item())

                        if i_backup > 0 and int_it % i_backup == 0 and it > 0:
                            checkpoint_io.save(filename='{:08d}.pt'.format(it), global_step=it, epoch_idx=epoch_idx)

                    #----------------------------------------------------------------------------
                    #------------------- things done in every child process ---------------------------
                    #----------------------------------------------------------------------------

                    #-------------------
                    # log grads and learning rate
                    for k, v in grad_norms.items():
                        logger.add('grad', k, v, it)
                    logger.add('learning rates', 'whole', optimizer.param_groups[0]['lr'], it)

                    #-------------------
                    # log losses
                    for k, v in losses.items():
                        logger.add('losses', k, v.data.cpu().numpy().item(), it)
                    
                    #-------------------
                    # log extras
                    names = ["radiance", "alpha", "implicit_surface", "implicit_nablas_norm", "sigma_out", "radiance_out"]
                    for n in names:
                        p = "whole"
                        # key = "raw.{}".format(n)
                        key = n
                        if key in extras:
                            logger.add("extras_{}".format(n), "{}.mean".format(p), extras[key].mean().data.cpu().numpy().item(), it)
                            logger.add("extras_{}".format(n), "{}.min".format(p), extras[key].min().data.cpu().numpy().item(), it)
                            logger.add("extras_{}".format(n), "{}.max".format(p), extras[key].max().data.cpu().numpy().item(), it)
                            logger.add("extras_{}".format(n), "{}.norm".format(p), extras[key].norm().data.cpu().numpy().item(), it)
                    if 'scalars' in extras:
                        for k, v in extras['scalars'].items():
                            logger.add('scalars', k, v.mean(), it)                           

                    #---------------------
                    # end of one iteration
                    end_time = time.time()
                    log.debug("=> One iteration time is {:.2f}".format(end_time - start_time))
                    
                    it += world_size
                    if is_master():
                        pbar.update(world_size)
                #---------------------
                # end of one epoch
                epoch_idx += 1

            except KeyboardInterrupt:
                if is_master():
                    checkpoint_io.save(filename='latest.pt'.format(it), global_step=it, epoch_idx=epoch_idx)
                    # this will be used for plotting
                logger.save_stats('stats.p')
                sys.exit()

    if is_master():
        checkpoint_io.save(filename='final_{:08d}.pt'.format(it), global_step=it, epoch_idx=epoch_idx)
        logger.save_stats('stats.p')
        log.info("Everything done.")

if __name__ == "__main__":
    # Arguments
    parser = io_util.create_args_parser()
    parser.add_argument("--ddp", action='store_true', help='whether to use DDP to train.')
    parser.add_argument("--port", type=int, default=None, help='master port for multi processing. (if used)')
    args, unknown = parser.parse_known_args()
    config = io_util.load_config(args, unknown)
    main_function(config)