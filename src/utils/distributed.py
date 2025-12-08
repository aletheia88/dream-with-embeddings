import os
from typing import Tuple

import torch
import torch.distributed as dist


def init_distributed_mode(backend: str = "nccl", require_env: bool = False) -> Tuple[int, int, torch.device]:
    """
    Initialize torch.distributed if environment variables are set.
    Falls back to single-process execution when no distributed context is present.
    """
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device_index = torch.cuda.current_device() if torch.cuda.is_available() else -1
        device = torch.device("cuda", device_index) if device_index >= 0 else torch.device("cpu")
        return rank, world_size, device

    has_env = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not has_env and require_env:
        raise RuntimeError(
            "Distributed environment variables (RANK, WORLD_SIZE) are not set. "
            "Launch via torchrun/torch.distributed to enable DDP."
        )

    if has_env:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, world_size, device


def seed_everything(base_seed: int, world_size: int, rank: int) -> int:
    """
    Apply a deterministic seed based on the base_seed, world size, and rank.
    Returns the per-rank seed that was applied.
    """
    seed = base_seed * max(world_size, 1) + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def cleanup_distributed() -> None:
    """Destroy the distributed process group if it was initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    """Convenience barrier that is a no-op when distributed is not initialized."""
    if dist.is_initialized():
        dist.barrier()


def get_world_size() -> int:
    """Safely get world size without requiring initialization."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Safely get rank without requiring initialization."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0
