import os
import torch
import torch.distributed as dist

rank = 0        # process id, for IPC
local_rank = 0  # local GPU device id
world_size = 1  # number of processes

def init_env(args):
    global rank, local_rank, world_size
    if args.ddp:
        #------------- multi process running, using DDP
        dist.init_process_group(backend='nccl')
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(local_rank)
        args.device_ids = [local_rank]
        print("=> Init Env @ DDP: rank={}, world_size={}, local_rank={}.\n\tdevice_ids set to {}".format(rank, world_size, local_rank, args.device_ids))
        # NOTE: important!
    else:
        #------------- single process running, using single GPU or DataParallel
        # torch.cuda.set_device(args.device_ids[0])
        print("=> Init Env @ single process: use device_ids = {}".format(args.device_ids))
        rank = 0
        local_rank = args.device_ids[0]
        world_size = 1
        torch.cuda.set_device(args.device_ids[0])

def is_master():
    return rank == 0

def get_rank():
    return int(os.environ.get('SLURM_PROCID', rank))

def get_local_rank():
    return int(os.environ.get('LOCAL_RANK', local_rank))

def get_world_size():
    return int(os.environ.get('SLURM_NTASKS', world_size))