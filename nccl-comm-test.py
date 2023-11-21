import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from enum import Enum
import os

class TransferTag(Enum):
    FEAT = 0

def init_process(rank, size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend='nccl', rank=rank, world_size=size)

def worker(rank, world_size):
    init_process(rank, world_size)

    tensor_size = (50000, 600)
    send_tensors = {dst: torch.ones(tensor_size, device=f'cuda:{rank}') for dst in range(world_size) if dst != rank}
    recv_tensors = {src: torch.zeros(tensor_size, device=f'cuda:{rank}') for src in range(world_size) if src != rank}

    reqs = []
    for dst, tensor in send_tensors.items():
        req = dist.isend(tensor, dst=dst)
        reqs.append(req)

    for src in recv_tensors.keys():
        dist.recv(recv_tensors[src], src=src)

    for req in reqs:
        req.wait()

    print(f'Worker {rank} received tensors:', recv_tensors)

def run(rank, size):
    init_process(rank, size)
    print(f'device cnt: {torch.cuda.device_count()}')
    device = None
    if dist.get_backend() == 'gloo':
        device = torch.device('cpu')
    elif dist.get_backend() == 'nccl':
        device = torch.device(rank)
    else:
        raise ValueError(f'{dist.get_backend()} backend not supported')

    # Reddit 2 partitions comm simulation
    if rank == 0:
        t0_send = torch.ones((42242, 602), dtype=torch.float, device=torch.device(rank))
        req = dist.isend(t0_send, dst=1, tag=TransferTag.FEAT.value)
        req.wait()
        t0_recv = torch.zeros((59891, 602), dtype=torch.float, device=torch.device(rank))
        dist.recv(t0_recv, src=1, tag=TransferTag.FEAT.value)
        print(f't0_recv {t0_recv}')
        
    elif rank == 1:
        t1_recv = torch.zeros((42242, 602), dtype=torch.float, device=torch.device(rank))
        dist.recv(t1_recv, src=0, tag=TransferTag.FEAT.value)
        print(f't1_recv {t1_recv}')
        t1_send = torch.ones((59891, 602), dtype=torch.float, device=torch.device(rank))
        req = dist.isend(t1_send, dst=0, tag=TransferTag.FEAT.value)
        req.wait()
    # dist.barrier()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(worker, args=(world_size,), nprocs=world_size)
