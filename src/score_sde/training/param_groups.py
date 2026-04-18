"""Optimizer parameter-group split: bias and norm parameters get weight_decay=0."""

import torch.nn as nn

# Norm layers whose parameters should never be decayed
_NO_DECAY_MODULE_TYPES = (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d, nn.InstanceNorm2d)
# Parameter names that should never be decayed regardless of module type
_NO_DECAY_PARAM_NAMES = {"bias"}


def get_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Return two param groups: one with decay, one without.

    No-decay set: bias parameters and all parameters of norm layers.
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []

    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        leaf_name = param_name.rsplit(".", 1)[-1]
        if leaf_name in _NO_DECAY_PARAM_NAMES:
            no_decay.append(param)
        elif isinstance(_get_parent_module(model, param_name), _NO_DECAY_MODULE_TYPES):
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _get_parent_module(root: nn.Module, param_name: str) -> nn.Module:
    """Walk the module tree and return the direct parent of `param_name`."""
    parts = param_name.split(".")
    module = root
    for part in parts[:-1]:
        module = getattr(module, part)
    return module
