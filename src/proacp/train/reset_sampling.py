"""Return-based reset sampling helpers.

From supermimic_vibed (Chong Zhang, https://github.com/zita-ch/supermimic_vibed, MIT License).
"""

import torch


def update_bin_returns(bin_G: torch.Tensor, bins: torch.Tensor, returns: torch.Tensor, old_weight: float = 0.8):
    sums = torch.zeros_like(bin_G)
    counts = torch.zeros_like(bin_G)
    sums.index_add_(0, bins, returns)
    counts.index_add_(0, bins, torch.ones_like(returns))
    updated = counts > 0
    bin_G[updated] = old_weight * bin_G[updated] + (1 - old_weight) * sums[updated] / counts[updated]
    return bin_G
