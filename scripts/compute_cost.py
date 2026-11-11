"""Count registered PyTorch FLOPs on a separate pass from latency measurements."""

import time

import torch
from torch.utils.flop_counter import FlopCounterMode


def count_flops(run):
    with FlopCounterMode(display=False) as counter:
        result = run()
    torch.cuda.synchronize()
    return result, int(counter.get_total_flops())


def measure(run, profile=False):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = run()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    if profile:
        return result, seconds, count_flops(run)
    return result, seconds, (None, None)
