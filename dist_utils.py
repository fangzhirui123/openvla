import os
import socket
import subprocess
from datetime import timedelta

import deepspeed
import torch
import torch.multiprocessing as mp
from torch import distributed as dist

timeout = timedelta(minutes=60)


def _find_free_port():
    # Copied from detectron2/detectron2/engine/launch.py at main · facebookresearch/detectron2 # noqa: E501
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def _is_free_port(port):
    ips = socket.gethostbyname_ex(socket.gethostname())[-1]
    ips.append('localhost')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return all(s.connect_ex((ip, port)) != 0 for ip in ips)


def init_dist(launcher, backend='nccl', **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    #print("launcher",launcher)
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'mpi':
        _init_dist_mpi(backend, **kwargs)
    elif launcher == 'slurm':
        #print("launcher == 'slurm'",launcher)
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')


def _init_dist_pytorch(backend, port=None):
    # TODO: use local_rank instead of rank % num_gpus
    #rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    num_gpus = torch.cuda.device_count()
    proc_id = int(os.environ['LOCAL_RANK'])#int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    #num_gpus = torch.cuda.device_count()
    print("proc_id",proc_id,"ntasks",ntasks,"node_list",node_list,"num_gpus",num_gpus,"local_rank",local_rank,"port",port)
    # proc_id 4 ntasks 8 node_list SH-IDCA1404-10-140-54-44 num_gpus 8
    #print("num_gpus",num_gpus)# num_gpus 0
    torch.cuda.set_device(local_rank % num_gpus)
    ###############################################
    addr = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    print("addr",addr)
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # if torch.distributed default port(29500) is available
        # then use it, else find a free port
        if _is_free_port(29500):
            os.environ['MASTER_PORT'] = '29500'
        else:
            os.environ['MASTER_PORT'] = str(_find_free_port())
    print("os.environ['MASTER_PORT']",os.environ['MASTER_PORT'])
    # use MASTER_ADDR in the environment variable if it already exists
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    print("os.environ['MASTER_ADDR']",os.environ['MASTER_ADDR'])
    # os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    # os.environ['RANK'] = str(proc_id)
    #########################################################
    # dist.init_process_group(backend=backend, **kwargs)
    deepspeed.init_distributed(dist_backend=backend, timeout=timedelta(minutes=int(60)))


def _init_dist_mpi(backend, **kwargs):
    local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    if 'MASTER_PORT' not in os.environ:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    if 'MASTER_ADDR' not in os.environ:
        raise KeyError('The environment variable MASTER_ADDR is not set')
    os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
    os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
    dist.init_process_group(backend=backend, **kwargs)


def _init_dist_slurm(backend, port=None):
    """Initialize slurm distributed training environment.

    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.

    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    print("proc_id",proc_id,"ntasks",ntasks,"node_list",node_list,"num_gpus",num_gpus,"port",port)
    #   proc_id 1 ntasks 8 node_list SH-IDCA1404-10-140-54-44 num_gpus 8 port None
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # if torch.distributed default port(29500) is available
        # then use it, else find a free port
        if _is_free_port(29500):
            os.environ['MASTER_PORT'] = '29500'
        else:
            os.environ['MASTER_PORT'] = str(_find_free_port())
    print("os.environ['MASTER_PORT']",os.environ['MASTER_PORT'])
    # use MASTER_ADDR in the environment variable if it already exists
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)
    # dist.init_process_group(backend=backend, timeout=timeout)
    print(backend, 'init')
    deepspeed.init_distributed(dist_backend=backend)
    
    #proc_id 6 ntasks 8 node_list SH-IDCA1404-10-140-54-121 num_gpus 2