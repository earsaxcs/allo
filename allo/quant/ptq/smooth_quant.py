"""Fixed-factor SmoothQuant fusion helpers.

This module intentionally provides a minimal, calibration-free variant:
callers provide a fixed smoothing factor (scalar or per-channel vector),
and the scaling is fused into static parameters only.
"""

from __future__ import annotations

from typing import Sequence, Union

import torch
import torch.nn as nn


ScaleLike = Union[float, int, torch.Tensor]


def _as_channel_scale(scale: ScaleLike, channels: int, device, dtype, name: str) -> torch.Tensor:
    if isinstance(scale, (int, float)):
        s = torch.full((channels,), float(scale), device=device, dtype=dtype)
    elif isinstance(scale, torch.Tensor):
        if scale.ndim == 0:
            s = torch.full((channels,), float(scale.item()), device=device, dtype=dtype)
        elif scale.ndim == 1 and scale.numel() == channels:
            s = scale.to(device=device, dtype=dtype)
        else:
            raise ValueError(
                f"{name} must be scalar or 1D tensor with {channels} elements, "
                f"got shape={tuple(scale.shape)}"
            )
    else:
        raise TypeError(f"{name} must be float/int/tensor, got {type(scale)}")

    if torch.any(s <= 0):
        raise ValueError(f"{name} must be strictly positive")
    return s


def _validate_linear_in_features(linear: nn.Linear, expected_in_features: int, name: str) -> None:
    if linear.in_features != expected_in_features:
        raise ValueError(
            f"{name}.in_features={linear.in_features} does not match expected "
            f"channels={expected_in_features}"
        )


@torch.no_grad()
def apply_fixed_smoothquant_ln_linear(
    layernorm: nn.LayerNorm,
    linears: Union[nn.Linear, Sequence[nn.Linear]],
    smooth_scale: ScaleLike,
) -> None:
    """Fuse fixed SmoothQuant scaling into LayerNorm -> Linear path.

    Math:
    - x' = x / s
    - W' = W * s

    Static fusion used here:
    - LayerNorm weight/bias are divided by s
    - each following Linear input-channel weight is multiplied by s
    """
    if isinstance(linears, nn.Linear):
        linear_list = [linears]
    else:
        linear_list = list(linears)

    channels = int(layernorm.normalized_shape[-1])
    s = _as_channel_scale(
        smooth_scale,
        channels=channels,
        device=layernorm.weight.device,
        dtype=layernorm.weight.dtype,
        name="smooth_scale",
    )

    layernorm.weight.data.div_(s)
    if layernorm.bias is not None:
        layernorm.bias.data.div_(s)

    s_in = s.view(1, -1)
    for idx, linear in enumerate(linear_list):
        _validate_linear_in_features(linear, channels, f"linears[{idx}]")
        linear.weight.data.mul_(s_in)


@torch.no_grad()
def apply_fixed_smoothquant_qk(
    linear_q: nn.Linear,
    linear_k: nn.Linear,
    qk_scale: ScaleLike,
) -> None:
    """Fuse fixed SmoothQuant scaling for q @ k^T into Q/K linears.

    Math-preserving transform:
    - q' = q / s
    - k' = k * s
    Then q' @ k'^T == q @ k^T.

    Because q and k are outputs of linear layers, this can be statically fused by
    scaling the output channels (rows) of linear_q and linear_k weights/biases.
    """
    if linear_q.out_features != linear_k.out_features:
        raise ValueError(
            f"linear_q.out_features={linear_q.out_features} and "
            f"linear_k.out_features={linear_k.out_features} must match"
        )

    channels = linear_q.out_features
    s = _as_channel_scale(
        qk_scale,
        channels=channels,
        device=linear_q.weight.device,
        dtype=linear_q.weight.dtype,
        name="qk_scale",
    )

    s_row = s.view(-1, 1)
    inv_s_row = (1.0 / s).view(-1, 1)

    linear_q.weight.data.mul_(inv_s_row)
    linear_k.weight.data.mul_(s_row)

    if linear_q.bias is not None:
        linear_q.bias.data.mul_(1.0 / s)
    if linear_k.bias is not None:
        linear_k.bias.data.mul_(s)


@torch.no_grad()
def apply_fixed_smoothquant_vit_block(
    block,
    ln_to_attn_scale: ScaleLike = 1.0,
    ln_to_ffn_scale: ScaleLike = 1.0,
    qk_scale: ScaleLike = 1.0,
) -> None:
    """Apply fixed-factor SmoothQuant fusion to a ViTBlock-like module.

    Expected attributes (same names as allo.ops.vit.ViTBlock):
    - block.norm1, block.norm2
    - block.attention.linear_q/linear_k/linear_v
    - block.ffn.fc1
    """
    apply_fixed_smoothquant_ln_linear(
        layernorm=block.norm1,
        linears=[block.attention.linear_q, block.attention.linear_k, block.attention.linear_v],
        smooth_scale=ln_to_attn_scale,
    )
    apply_fixed_smoothquant_ln_linear(
        layernorm=block.norm2,
        linears=block.ffn.fc1,
        smooth_scale=ln_to_ffn_scale,
    )
    apply_fixed_smoothquant_qk(
        linear_q=block.attention.linear_q,
        linear_k=block.attention.linear_k,
        qk_scale=qk_scale,
    )


@torch.no_grad()
def apply_fixed_smoothquant_vit_model(
    model: nn.Module,
    ln_to_attn_scale: ScaleLike = 1.0,
    ln_to_ffn_scale: ScaleLike = 1.0,
    qk_scale: ScaleLike = 1.0,
) -> int:
    """Apply fixed-factor SmoothQuant fusion to all ViT blocks in a model.

    Returns the number of transformed blocks.
    """
    transformed = 0
    for module in model.modules():
        has_attrs = all(
            hasattr(module, attr)
            for attr in ("norm1", "norm2", "attention", "ffn")
        )
        if not has_attrs:
            continue

        attn = getattr(module, "attention")
        ffn = getattr(module, "ffn")
        if not (
            hasattr(attn, "linear_q")
            and hasattr(attn, "linear_k")
            and hasattr(attn, "linear_v")
            and hasattr(ffn, "fc1")
        ):
            continue

        apply_fixed_smoothquant_vit_block(
            block=module,
            ln_to_attn_scale=ln_to_attn_scale,
            ln_to_ffn_scale=ln_to_ffn_scale,
            qk_scale=qk_scale,
        )
        transformed += 1
    return transformed
