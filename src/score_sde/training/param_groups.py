"""Optimizer parameter-group split: bias and norm parameters get weight_decay=0."""

import torch.nn as nn

# Norm layers whose parameters should never be decayed
_NO_DECAY_MODULE_TYPES = (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d, nn.InstanceNorm2d)
# Parameter names that should never be decayed regardless of module type
_NO_DECAY_PARAM_NAMES = {"bias"}


def get_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Return two optimizer parameter groups: one with decay and one without.

    The no-decay set contains bias parameters and all parameters of norm layers.

    Args:
        model (nn.Module): model whose parameters are to be split.
        weight_decay (float): L2 regularisation coefficient for the decay group.

    Returns:
        list[dict]: two-element list of parameter group dicts compatible with
            torch optimizers, the first with weight_decay and the second with 0.0.
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
    """Walk the module tree and return the direct parent module of a parameter.

    Args:
        root (nn.Module): root module to start traversal from.
        param_name (str): fully-qualified parameter name as returned by
            ``named_parameters()``, e.g. ``"layer.0.weight"``.

    Returns:
        nn.Module: immediate parent module that owns the parameter leaf.
    """
    parts = param_name.split(".")
    module = root
    for part in parts[:-1]:
        module = getattr(module, part)
    return module
