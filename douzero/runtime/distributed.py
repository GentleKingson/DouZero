"""Explicit DistributedDataParallel runtime helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistributedContext:
    """Process identity and device for one DDP learner rank."""

    enabled: bool
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    backend: str = "gloo"
    device: torch.device = torch.device("cpu")

    @property
    def is_rank_zero(self) -> bool:
        """Whether this process owns checkpoint and logging side effects."""
        return self.rank == 0

    def shard_indices(self, size: int) -> range:
        """Return non-overlapping strided sample indices for this rank."""
        return range(self.rank, size, self.world_size)

    def wrap(self, model: torch.nn.Module) -> torch.nn.Module:
        """Wrap a model in DDP, or return it unchanged for one process."""
        if not self.enabled:
            return model
        kwargs = {}
        if self.device.type == "cuda":
            kwargs = {"device_ids": [self.local_rank], "output_device": self.local_rank}
        # ModelV2 has stable parameter usage across optimizer iterations. A
        # static reducer avoids the per-step autograd traversal of
        # find_unused_parameters=True and correctly handles its structured
        # ModelOutput when only selected objectives feed the loss.
        return DistributedDataParallel(model, static_graph=True, **kwargs)

    def all_true(self, local_value: bool) -> bool:
        """Return true only when every learner rank reports true.

        Optimizer-step control flow must agree across ranks before any rank
        enters DDP backward. The disabled context preserves the same API for
        single-process callers without initializing a process group.
        """
        if not self.enabled:
            return bool(local_value)
        flag = torch.tensor(
            1 if local_value else 0,
            dtype=torch.int32,
            device=self.device,
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    def close(self) -> None:
        """Destroy the process group when initialized.

        Do not add a final barrier here: if one rank is exiting because another
        rank failed, a cleanup barrier would turn the original error into a
        secondary hang.
        """
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()


def initialize_distributed(*, enabled: bool, backend: str = "auto",
                           timeout_seconds: int = 180) -> DistributedContext:
    """Initialize a torchrun process group with one GPU per process."""
    if not enabled:
        return DistributedContext(enabled=False)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size < 2:
        raise ValueError("DDP requires WORLD_SIZE >= 2; launch with torchrun")
    resolved = backend
    if resolved == "auto":
        resolved = "nccl" if torch.cuda.is_available() else "gloo"
    if resolved not in {"nccl", "gloo"}:
        raise ValueError("ddp backend must be 'auto', 'nccl', or 'gloo'")
    if resolved == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL DDP requires CUDA")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    dist.init_process_group(backend=resolved, init_method="env://", rank=rank,
                            world_size=world_size,
                            timeout=timedelta(seconds=timeout_seconds))
    return DistributedContext(True, rank, world_size, local_rank, resolved, device)


def checkpoint_map_location(device: torch.device) -> torch.device:
    """Return a rank-local map_location suitable for checkpoint restore."""
    return torch.device(device)
