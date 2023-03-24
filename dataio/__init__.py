def get_data(args, return_val=False, only_val=False, val_downscale=4.0, **overwrite_cfgs):
    dataset_type = args.data.get('type', 'DTU')
    cfgs = {
        'scale_radius': args.data.get('scale_radius', -1),
        'downscale': args.data.downscale,
        'data_dir': args.data.data_dir,
        'train_cameras': False
    }
    
    if dataset_type == 'DTU':
        from .DTU import SceneDataset
        cfgs['cam_file'] = args.data.get('cam_file', None)
    elif dataset_type == 'nerf':
        from .nerf import SceneDataset
        # pre-computed minimum distance from train cameras' center to the origin
        cfgs['train_cam_min_norm'] = args.data.get('train_cam_min_norm', 4.031128660578481)
        cfgs['rgb_file'] = args.data.get('rgb_file', 'rgba.png')
    else:
        raise NotImplementedError

    cfgs.update(overwrite_cfgs)
    if only_val:
        cfgs['downscale'] = val_downscale
        cfgs['val'] = True
        val_dataset = SceneDataset(**cfgs)
        return val_dataset
    
    dataset = SceneDataset(**cfgs)
    if return_val:
        cfgs['downscale'] = val_downscale
        cfgs['val'] = True
        val_dataset = SceneDataset(**cfgs)
        return dataset, val_dataset
    else:
        return dataset