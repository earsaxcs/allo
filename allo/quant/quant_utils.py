import torch

import torch
import torch.nn as nn
import math

# Debug switch for printing quantization messages
DEBUG_QUANT = False

def max_min_quantize_params(
    input_tensor: torch.Tensor | None,
    bitwidth: int = 8,
    quant_mode: str = 'sym', # 'sym' or 'asym'
    per_channel: bool = False,
    is_weight: bool = True,       # New: Indicate if the tensor is a weight
    is_seq_x: bool = False,        # When `input_tensor` is 4D, False means that `input_tensor` is conv feature, True means that `input_tensor` is multi-hhead sequence attention score in Transformer
    channel_dim: int | None = None, # Optional: Manually specify channel dimension. If None and per_channel is True, infer based on ndim and is_weight.
    dtype: torch.dtype = torch.float32 # Dtype for scale and zero_point
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    计算张量的最大最小值量化参数 (Scale 和 Zero Point)，支持根据阶数和是否为权重推断 Per-Channel 维度。

    Args:
        input_tensor: 输入的 PyTorch 张量。目前支持 1D, 2D, 3D, 4D。
        bitwidth: 量化的位宽 (例如，8 表示 INT8)。
        quant_mode: 量化类型，'sym' 或 'asym'。
        per_channel: 是否使用 per-channel 量化。注意：1D 张量不支持 per-channel。
        is_weight: 布尔值，指示张量是否为模型权重。用于 2D/4D 张量推断 per-channel 维度。
        channel_dim: 可选参数。如果使用 per-channel 并手动指定通道维度，则使用此值。
                     如果 per_channel 为 True 且 channel_dim 为 None，函数将根据 input_tensor 的阶数和 is_weight 推断通道维度。
                     默认推断规则见函数说明。
        dtype: 输出的 scale 和 zero_point 张量的数据类型 (例如，torch.float32)。

    Returns:
        一个元组 (scale, zero_point)。
        scale 是量化缩放因子张量。
        zero_point 是量化零点张量。如果使用对称量化，zero_point 为 None。

    Raises:
        ValueError: 如果量化类型、位宽或张量阶数无效 (>4D)，或手动指定的 channel_dim 无效。
    """
    if input_tensor is None:
        return None, None

    if quant_mode not in ['sym', 'asym']:
        raise ValueError("quant_mode must be 'sym' or 'asym'")
    if not 2 <= bitwidth <= 32: # Arbitrary but reasonable bitwidth range
        if DEBUG_QUANT:
            print(f"Warning: Using bitwidth {bitwidth} which is outside the typical range (2-32).")

    ndim = input_tensor.ndim
    if ndim < 1 or ndim > 4: # Now supports 1D, 2D, 3D, 4D
        raise ValueError(f"Input tensor must be 1D, 2D, 3D, or 4D for this function. Got {ndim}D tensor.")

    # 确定整数范围
    if quant_mode == 'sym':
        int_min_q = -(2**(bitwidth - 1))
        int_max_q = (2**(bitwidth - 1)) - 1
    elif quant_mode == 'asym':
        int_min_q = 0
        int_max_q = (2**bitwidth) - 1

    # 确定 Per-Channel 量化维度和 Reduction 维度
    actual_channel_dim = None
    _per_channel_effective = per_channel # Use an internal flag for effective per_channel state
    _is_seq_x_effective = is_seq_x # Use an internal flag for effective is_seq_x state

    if ndim == 1:
        # 1D 张量 (Bias, Norm Weight) 不支持 Per-Channel
        # 不过Bias很多时候是直接被动继承中间值的scale的，不会单独统计其min、max了
        # Norm Weight这里只适用于Transformer里的Norm，只针对hidden_dim这一维
        if per_channel:
            if DEBUG_QUANT:
                print(f"Warning: 1D tensor (shape {input_tensor.shape}) does not support per-channel quantization. Forcing per-tensor.")
        _per_channel_effective = False

    if _per_channel_effective:
        if channel_dim is not None:
            # 使用用户指定的维度
            actual_channel_dim = channel_dim
        else:
            # 根据阶数和是否为权重推断维度
            if ndim == 2:
                # Linear Weight [O, I] (dim 0) vs Linear Activation [B, I] (dim 1)
                actual_channel_dim = 0 if is_weight else 1
                if DEBUG_QUANT:
                    print(f"Info: Inferring channel_dim={actual_channel_dim} for {ndim}D tensor based on is_weight={is_weight}.")
            elif ndim == 3:
                # NOTE: is_weight==False means its per-token currently
                actual_channel_dim = 1
                if DEBUG_QUANT:
                    print(f"Info: Inferring channel_dim={actual_channel_dim} for {ndim}D tensor.")
                if is_weight:
                    # Typical for Transformer/RNN Activation [B, S, E] (dim 2)
                     actual_channel_dim = 2
                     if DEBUG_QUANT:
                         print("Info: Assuming 3D tensor is activation-like for per-channel inference. If it's a 3D weight, manually set channel_dim.")
            elif ndim == 4:
                if _is_seq_x_effective and not is_weight:
                    # Transformer Attention Score [B, H, L, L]
                    actual_channel_dim = 2
                    if DEBUG_QUANT:
                        print(f"Info: Inferring channel_dim={actual_channel_dim} for {ndim}D tensor based on is_weight={is_weight}. It's Transformer Attention Score")
                else:
                    # Conv Weight [O, I, K, K] (dim 0) vs Conv Activation [B, C, H, W] (dim 1)
                    actual_channel_dim = 0 if is_weight else 1
                    if DEBUG_QUANT:
                        print(f"Info: Inferring channel_dim={actual_channel_dim} for {ndim}D tensor based on is_weight={is_weight} (assuming channels-first for activations). For channels-last activation ([B, H, W, C]), please manually set channel_dim=3.")

        # 验证确定的维度是否有效
        if actual_channel_dim < 0 or actual_channel_dim >= ndim:
             raise ValueError(f"Invalid channel_dim {actual_channel_dim} determined for a {ndim}D tensor (is_weight={is_weight}).")

        # 找出需要 reduction 的维度（除了 channel_dim）
        permute_list = list(range(ndim))
        try:
            permute_list.remove(actual_channel_dim)
        except:
            raise ValueError(f"Invalid channel_dim {actual_channel_dim} determined for a {ndim}D tensor (is_weight={is_weight}).")
        permute_list.append(actual_channel_dim)

        input_tensor = input_tensor.permute(permute_list)
        input_tensor = input_tensor.reshape(-1, input_tensor.shape[-1])

        # 计算每个通道的 Min 和 Max
        # 使用 detach() 避免梯度计算对 min/max 查找的影响
        min_val = torch.min(input_tensor.detach(), dim=0, keepdim=False)[0]
        max_val = torch.max(input_tensor.detach(), dim=0, keepdim=False)[0]

    else: # Per-tensor quantization
        # 计算整个张量的 Min 和 Max (Scalar)
        min_val = torch.min(input_tensor.detach()).unsqueeze(0)
        max_val = torch.max(input_tensor.detach()).unsqueeze(0)

    # 将 min_val 和 max_val 转换为目标 dtype
    min_val = min_val.to(dtype)
    max_val = max_val.to(dtype)

    # 计算 Scale 和 Zero Point (使用浮点数计算)
    scale = torch.tensor(1.0, dtype=dtype, device=input_tensor.device) # Default scale
    zero_point = None # Default zero point for symmetric

    if quant_mode == 'sym':
        # 对称量化，范围基于最大绝对值
        abs_max = torch.max(torch.abs(min_val), torch.abs(max_val))

        # Avoid division by zero if tensor is all zeros
        scale = torch.where(
            abs_max == 0,
            torch.tensor(1e-8, dtype=dtype, device=input_tensor.device), # Use a small epsilon if range is 0
            abs_max / int_max_q # Map abs_max to int_max_q (2^(B-1)-1)
        )
        zero_point = None # 对称量化通常不返回 zero_point

    elif quant_mode == 'asym':
        # 非对称量化，范围基于 max - min

        # Ensure min_val <= max_val
        min_val = torch.min(min_val, max_val)

        # Handle case where min_val == max_val (constant tensor)
        range_flt = max_val - min_val
        scale = torch.where(
            range_flt == 0,
            torch.tensor(1e-8, dtype=dtype, device=input_tensor.device), # Use a small epsilon if range is 0
            range_flt / (int_max_q - int_min_q)
        )

        # 计算 Zero Point
        # ZeroPoint = int_min_q - round(min_val / Scale) # Using round directly might be problematic with gradients
        # A better way: ZeroPoint = int_min_q - min_val / Scale, then round the float result
        zero_point = int_min_q - min_val / scale
        zero_point = torch.round(zero_point)

        # Handle zero_point when range was zero (min_val == max_val)
        zero_point = torch.where(
             range_flt == 0,
             torch.tensor(float(int_min_q), dtype=dtype, device=input_tensor.device), # Map to int_min_q as float
             zero_point
        )

        # Clamp zero_point to be within the target integer range [int_min_q, int_max_q]
        zero_point = torch.clamp(zero_point, int_min_q, int_max_q)
        # zero_point is returned as float dtype

    return scale, zero_point

# TODO: follow max_min_quantize_params set seq_int_x
def mean_std_quantize_params(
    input_tensor: torch.Tensor | None,
    bitwidth: int = 8,
    quant_mode: str = 'sym', # 'sym' or 'asym'
    per_channel: bool = False,
    is_weight: bool = False,      # Indicate if the tensor is a weight (helps infer channel_dim)
    is_seq_x: bool = False,       # For 4D activation: False=conv feature, True=Transformer attention score
    channel_dim: int | None = None, # Optional: Manually specify channel dimension. If None and per_channel is True, infer based on ndim and is_weight.
    n_sigmas: float = 3.0,        # Number of standard deviations to consider for the range
    dtype: torch.dtype = torch.float32 # Dtype for scale and zero_point
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    计算张量的均值标准差量化参数 (Scale 和 Zero Point)。
    量化范围设定为 [mean - n_sigmas * std, mean + n_sigmas * std]。

    Args:
        input_tensor: 输入的 PyTorch 张量。目前支持 1D, 2D, 3D, 4D。
        bitwidth: 量化的位宽 (例如，8 表示 INT8)。
        quant_mode: 量化类型，'sym' 或 'asym'。
        per_channel: 是否使用 per-channel 量化。注意：1D 张量不支持 per-channel。
        is_weight: 布尔值，指示张量是否为模型权重。用于 2D/4D 张量推断 per-channel 维度。
        channel_dim: 可选参数。如果使用 per-channel 并手动指定通道维度，则使用此值。
                     如果 per_channel 为 True 且 channel_dim 为 None，函数将根据 input_tensor 的阶数和 is_weight 推断通道维度。
                     默认推断规则见函数说明。
        n_sigmas: 确定量化范围的倍数，范围设定为 [mean - n_sigmas * std, mean + n_sigmas * std]。
        dtype: 输出的 scale 和 zero_point 张量的数据类型 (例如，torch.float32)。

    Returns:
        一个元组 (scale, zero_point)。
        scale 是量化缩放因子张量。
        zero_point 是量化零点张量。如果使用对称量化，zero_point 为 None。

    Raises:
        ValueError: 如果量化类型、位宽或张量阶数无效 (>4D)，或手动指定的 channel_dim 无效。
    """
    if input_tensor is None:
        return None, None

    if quant_mode not in ['sym', 'asym']:
        raise ValueError("quant_mode must be 'sym' or 'asym'")
    if not 2 <= bitwidth <= 32:
        if DEBUG_QUANT:
            print(f"Warning: Using bitwidth {bitwidth} which is outside the typical range (2-32).")
    if n_sigmas <= 0:
         raise ValueError("n_sigmas must be positive.")

    ndim = input_tensor.ndim
    if ndim < 1 or ndim > 4: # Supports 1D, 2D, 3D, 4D
        raise ValueError(f"Input tensor must be 1D, 2D, 3D, or 4D for this function. Got {ndim}D tensor.")

    # 确定整数范围
    if quant_mode == 'sym':
        int_min_q = -(2**(bitwidth - 1))
        int_max_q = (2**(bitwidth - 1)) - 1
    elif quant_mode == 'asym':
        int_min_q = 0
        int_max_q = (2**bitwidth) - 1

    # 确定 Per-Channel 量化维度和 Reduction 维度
    actual_channel_dim = None
    _per_channel_effective = per_channel # Use an internal flag for effective per_channel state
    _is_seq_x_effective = is_seq_x

    if ndim == 1:
        # 1D 张量 (Bias, Norm Weight) 不支持 Per-Channel
        if per_channel:
            if DEBUG_QUANT:
                print(f"Warning: 1D tensor (shape {input_tensor.shape}) does not support per-channel quantization based on shape. Forcing per-tensor.")
        _per_channel_effective = False

    if _per_channel_effective:
        if channel_dim is not None:
            # 使用用户指定的维度
            actual_channel_dim = channel_dim
        else:
            # 根据阶数和是否为权重推断维度
            if ndim == 2:
                # Linear Weight [O, I] (dim 0) vs Linear Activation [B, I] (dim 1)
                actual_channel_dim = 0 if is_weight else 1
                if DEBUG_QUANT:
                    print(f"Info: Inferring channel_dim={actual_channel_dim} for {ndim}D tensor based on is_weight={is_weight}.")
            elif ndim == 3:
                # Typical for Transformer/RNN Activation [B, S, E] (dim 2)
                # NOTE: is_weight==False means its per-token currently
                actual_channel_dim = 1
                if DEBUG_QUANT:
                    print(f"Info: Inferring channel_dim={actual_channel_dim} for {ndim}D tensor.")
                if is_weight:
                    actual_channel_dim = 2
                    if DEBUG_QUANT:
                        print("Info: Assuming 3D tensor is activation-like for per-channel inference. If it's a 3D weight, manually set channel_dim.")
            elif ndim == 4:
                if _is_seq_x_effective and not is_weight:
                    # Transformer Attention Score [B, H, L, L]
                    actual_channel_dim = 2
                    if DEBUG_QUANT:
                        print(f"Info: Inferring channel_dim={actual_channel_dim} for {ndim}D tensor based on is_weight={is_weight}. It's Transformer Attention Score")
                else:
                    # Conv Weight [O, I, K, K] (dim 0) vs Conv Activation [B, C, H, W] (dim 1)
                    actual_channel_dim = 0 if is_weight else 1
                    if DEBUG_QUANT:
                        print(f"Info: Inferring channel_dim={actual_channel_dim} for {ndim}D tensor based on is_weight={is_weight} (assuming channels-first for activations). For channels-last activation ([B, H, W, C]), manually set channel_dim=3.")

        # 验证确定的维度是否有效
        if actual_channel_dim < 0 or actual_channel_dim >= ndim:
             raise ValueError(f"Invalid channel_dim {actual_channel_dim} determined for a {ndim}D tensor (is_weight={is_weight}).")

        # 对齐 max_min：先把 channel 维移到最后，再展平到 2D，最后按列统计。
        permute_list = list(range(ndim))
        try:
            permute_list.remove(actual_channel_dim)
        except:
            raise ValueError(f"Invalid channel_dim {actual_channel_dim} determined for a {ndim}D tensor (is_weight={is_weight}).")
        permute_list.append(actual_channel_dim)

        input_tensor = input_tensor.permute(permute_list)
        input_tensor = input_tensor.reshape(-1, input_tensor.shape[-1])

        # 计算每个通道的 Mean 和 Std Dev
        # 使用 detach() 避免梯度计算的影响
        # unbiased=False 计算的是总体标准差
        mean_val = torch.mean(input_tensor.detach(), dim=0, keepdim=False)
        std_val = torch.std(input_tensor.detach(), dim=0, keepdim=False, unbiased=False)

    else: # Per-tensor quantization
        # 计算整个张量的 Mean 和 Std Dev (Scalar)
        # Keep per-tensor stats as 1D tensors for API consistency with max_min path.
        mean_val = torch.mean(input_tensor.detach()).unsqueeze(0)
        std_val = torch.std(input_tensor.detach(), unbiased=False).unsqueeze(0)
        reduction_dims = None # No reduction needed

    # 将 mean_val 和 std_val 转换为目标 dtype
    mean_val = mean_val.to(dtype)
    std_val = std_val.to(dtype)

    # 计算量化范围 [range_min_flt, range_max_flt]
    range_min_flt = mean_val - n_sigmas * std_val
    range_max_flt = mean_val + n_sigmas * std_val

    # 将 range_min_flt 和 range_max_flt 转换为目标 dtype (如果它们不是)
    range_min_flt = range_min_flt.to(dtype)
    range_max_flt = range_max_flt.to(dtype)


    # 计算 Scale 和 Zero Point
    scale = torch.tensor(1.0, dtype=dtype, device=input_tensor.device) # Default scale
    zero_point = None # Default zero point for symmetric

    # Handle edge case: std_val is 0 (tensor is constant)
    # In this case, range_min_flt == range_max_flt == mean_val
    is_constant = (std_val == 0) | (torch.isnan(std_val)) # Also check for NaN in std dev

    if quant_mode == 'sym':
        # 对称量化，范围通常是 [-Threshold, +Threshold]，Threshold = n_sigmas * std
        # 也可以直接从计算出的 [range_min_flt, range_max_flt] 中取最大绝对值作为 Threshold
        # threshold = torch.max(torch.abs(range_min_flt), torch.abs(range_max_flt)) # Option 1
        # Or simply use n_sigmas * std_val as threshold
        threshold = n_sigmas * std_val # Option 2 (more direct from mean/std concept)

        # Handle constant tensor case (std_val == 0)
        scale = torch.where(
            is_constant,
            torch.tensor(1e-8, dtype=dtype, device=input_tensor.device), # Use a small epsilon if range is 0
            threshold / (2**(bitwidth - 1) - 1) # Map threshold to symmetric int max (2^(B-1)-1)
        )
        zero_point = None # 对称量化通常不返回 zero_point

    elif quant_mode == 'asym':
        # 非对称量化，范围 [range_min_flt, range_max_flt] 映射到 [int_min_q, int_max_q]

        # Ensure range_min_flt <= range_max_flt
        range_min_flt = torch.min(range_min_flt, range_max_flt)
        range_max_flt = torch.max(range_min_flt, range_max_flt) # Recalculate max after potential swap

        range_flt = range_max_flt - range_min_flt

        # Handle constant tensor case (std_val == 0)
        scale = torch.where(
            is_constant,
            torch.tensor(1e-8, dtype=dtype, device=input_tensor.device), # Use a small epsilon if range is 0
            range_flt / (int_max_q - int_min_q)
        )

        # 计算 Zero Point (使用浮点数计算，结果四舍五入到最接近的整数)
        # We want range_min_flt to correspond to int_min_q
        # int_min_q = round(range_min_flt / Scale) + ZeroPoint # This is incorrect based on the formula
        # Correct formula: FP = (Q - ZeroPoint) * Scale => Q = FP / Scale + ZeroPoint
        # Map range_min_flt to int_min_q: int_min_q = range_min_flt / Scale + ZeroPoint
        # ZeroPoint = int_min_q - range_min_flt / Scale
        zero_point = int_min_q - range_min_flt / scale
        zero_point = torch.round(zero_point)

        # Handle constant tensor case (std_val == 0): Map the single value (mean_val)
        # In asymmetric, map mean_val to int_min_q or the center?
        # Let's be consistent with min/max edge case and map to int_min_q with scale 1.0
        # int_min_q = (mean_val - ZeroPoint) * 1.0 => ZeroPoint = mean_val - int_min_q
        zero_point = torch.where(
             is_constant,
             torch.round(mean_val - int_min_q), # Map mean_val to int_min_q
             zero_point
        )
        # Note: If range_flt == 0, the scale was set to 1e-8. The calculation `int_min_q - range_min_flt / scale`
        # will result in a very large number. The `torch.where` correctly overrides this with the `round(mean_val - int_min_q)` logic.

        # Clamp zero_point to be within the target integer range [int_min_q, int_max_q]
        zero_point = torch.clamp(zero_point, int_min_q, int_max_q)
        # zero_point is returned as float dtype

    # Keep output shape consistent: per-tensor returns length-1 1D tensors instead of 0D scalars.
    if scale is not None and scale.ndim == 0:
        scale = scale.unsqueeze(0)
    if zero_point is not None and zero_point.ndim == 0:
        zero_point = zero_point.unsqueeze(0)

    return scale, zero_point


def compute_quantize_params(
    stat_method: str,
    n_sigmas: float = 3.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Dispatch quantization statistics method for calibration.

    Supported methods:
    - "max_min": min/max based range
    - "mean_std": mean +/- n_sigmas * std based range
    """
    method = (stat_method or "max_min").lower()
    if method == "max_min":
        return max_min_quantize_params(**kwargs)
    if method == "mean_std":
        allowed = {
            "input_tensor",
            "bitwidth",
            "quant_mode",
            "per_channel",
            "is_weight",
            "is_seq_x",
            "channel_dim",
            "dtype",
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        return mean_std_quantize_params(n_sigmas=n_sigmas, **filtered_kwargs)
    raise ValueError(f"Unsupported calibration statistic method: {stat_method}")

def linear_quantize(input, scale, zero_point, is_weight):
    """
    Quantize single-precision input tensor to integers with the given scaling factor and zeropoint.
    Parameters:
    ----------
    input: single-precision input tensor to be quantized
    scale: scaling factor for quantization
    zero_point: shift for quantization
    """

    # reshape scale and zeropoint for convolutional/linear weights and activation
    # here scale and zero can be both per_tensor and per_channel, just depend the value of the dim who is set to -1 in the view function
    if is_weight:
        if len(input.shape) == 4:
            scale = scale.view(-1, 1, 1, 1)
            zero_point = zero_point.view(-1, 1, 1, 1)
            # reshape scale and zeropoint for linear weights
        elif len(input.shape) == 2:
            scale = scale.view(-1, 1)
            zero_point = zero_point.view(-1, 1)
        else: # len(input.shape) == 1:
            scale = scale.view(-1)
            zero_point = zero_point.view(-1)
    else:
        if len(input.shape) == 1:
            scale = scale.view(-1)
            zero_point = zero_point.view(-1)
        elif len(input.shape) == 2:
            scale = scale.view(1, -1)
            zero_point = zero_point.view(1, -1)
        # TODO: bmm?
        elif len(input.shape) == 3:
            scale = scale.view(1, 1, -1)
            zero_point = zero_point.view(1, 1, -1)
        elif len(input.shape) == 4:
            scale = scale.view(1, -1, 1, 1)
            zero_point = zero_point.view(1, -1, 1, 1)
        else:
            raise NotImplementedError

    # quantized = float / scale + zero_point
    return torch.round(1. / scale * input + zero_point)

def symmetric_linear_quantize(bits, input, scale, is_weight):
    zero = torch.tensor(0.)
    return asymmetric_linear_quantize(bits, input, scale, zero, is_weight)

def asymmetric_linear_quantize(bits, input, scale, zero, is_weight):
    # bits is just used to clamp the quantized params
    n = 2 ** (bits - 1) - 1
    quant_input = linear_quantize(input, scale, zero, is_weight)
    quant_input = torch.clamp(quant_input, -n-1, n)
    return quant_input

#----- Test -----#
def test_min_max_quantize_params():
    print("--- 示例使用 max_min_quantize_params (修订版) ---")

    # 示例 1: 1D 张量 (Bias 或 Norm Weight) - Per-tensor Symmetric INT8
    print("\n--- 1D Tensor (Bias/Norm Weight) - Per-tensor Symmetric INT8 ---")
    bias_1d = torch.randn(64)
    scale_sym_pt_1d, zero_sym_pt_1d = max_min_quantize_params(bias_1d, bitwidth=8, quant_mode='sym', per_channel=False, is_weight=True)
    scale_sym_pt_1d_pc_true, zero_sym_pt_1d_pc_true = max_min_quantize_params(bias_1d, bitwidth=8, quant_mode='sym', per_channel=True, is_weight=True) # per_channel will be ignored
    print(f"Input Shape: {bias_1d.shape}, Min: {bias_1d.min().item():.4f}, Max: {bias_1d.max().item():.4f}")
    print(f"Scale (per_channel=False): {scale_sym_pt_1d.item():.6f}")
    print(f"Zero Point (per_channel=False): {zero_sym_pt_1d}")
    print(f"Scale (per_channel=True - forced per-tensor): {scale_sym_pt_1d_pc_true.item():.6f}")


    # 示例 2: 2D 张量 - Per-channel Symmetric INT8 (Linear Weight)
    # is_weight=True -> inferred channel_dim = 0
    print("\n--- 2D Tensor - Per-channel Symmetric INT8 (Linear Weight, is_weight=True -> inferred dim=0) ---")
    weight_2d_linear = torch.randn(4, 16) # 4 out_features, 16 in_features
    scale_sym_pc_2d_w, zero_sym_pc_2d_w = max_min_quantize_params(weight_2d_linear, bitwidth=8, quant_mode='sym', per_channel=True, is_weight=True)
    print(f"Input Shape: {weight_2d_linear.shape}")
    print(f"Per-channel Scale (Shape {scale_sym_pc_2d_w.shape}):\n {scale_sym_pc_2d_w}")

    # 示例 3: 2D 张量 - Per-channel Asymmetric INT8 (Linear Activation)
    # is_weight=False -> inferred channel_dim = 1
    print("\n--- 2D Tensor - Per-channel Asymmetric INT8 (Linear Activation, is_weight=False -> inferred dim=1) ---")
    activation_2d_linear = torch.randn(8, 32) # Batch=8, Features=32
    scale_asym_pc_2d_a, zero_asym_pc_2d_a = max_min_quantize_params(activation_2d_linear, bitwidth=8, quant_mode='asym', per_channel=True, is_weight=False)
    print(f"Input Shape: {activation_2d_linear.shape}")
    print(f"Per-channel Scale (Shape {scale_asym_pc_2d_a.shape}):\n {scale_asym_pc_2d_a}")
    print(f"Per-channel Zero Point (Shape {zero_asym_pc_2d_a.shape}):\n {zero_asym_pc_2d_a}")


    # 示例 4: 3D 张量 - Per-channel Asymmetric INT8 (Transformer/RNN Activation)
    # is_weight=False (default) -> inferred channel_dim = 2
    print("\n--- 3D Tensor - Per-channel Asymmetric INT8 (Activation, inferred dim=2) ---")
    activation_3d = torch.randn(2, 10, 32) # Batch=2, Seq=10, Embedding=32
    scale_asym_pc_3d, zero_asym_pc_3d = max_min_quantize_params(activation_3d, bitwidth=8, quant_mode='asym', per_channel=True, is_weight=False)
    print(f"Input Shape: {activation_3d.shape}")
    print(f"Per-channel Scale (Shape {scale_asym_pc_3d.shape}):\n {scale_asym_pc_3d}")
    print(f"Per-channel Zero Point (Shape {zero_asym_pc_3d.shape}):\n {zero_asym_pc_3d}")


    # 示例 5: 4D 张量 - Per-channel Symmetric INT8 (Conv Weight)
    # is_weight=True -> inferred channel_dim = 0
    print("\n--- 4D Tensor - Per-channel Symmetric INT8 (Conv Weight, is_weight=True -> inferred dim=0) ---")
    weight_4d_conv = torch.randn(8, 3, 3, 3) # 8 out_channels, 3 in_channels, 3x3 kernel
    scale_sym_pc_4d_w, zero_sym_pc_4d_w = max_min_quantize_params(weight_4d_conv, bitwidth=8, quant_mode='sym', per_channel=True, is_weight=True)
    print(f"Input Shape: {weight_4d_conv.shape}")
    print(f"Per-channel Scale (Shape {scale_sym_pc_4d_w.shape}):\n {scale_sym_pc_4d_w}")


    # 示例 6: 4D 张量 - Per-channel Asymmetric INT8 (Conv Activation, channels-first)
    # is_weight=False -> inferred channel_dim = 1
    print("\n--- 4D Tensor - Per-channel Asymmetric INT8 (Conv Activation, channels-first, is_weight=False -> inferred dim=1) ---")
    activation_4d_cf = torch.randn(1, 64, 28, 28) # Batch=1, Channels=64, H=28, W=28
    scale_asym_pc_4d_a_cf, zero_asym_pc_4d_a_cf = max_min_quantize_params(activation_4d_cf, bitwidth=8, quant_mode='asym', per_channel=True, is_weight=False)
    print(f"Input Shape: {activation_4d_cf.shape}")
    print(f"Per-channel Scale (Shape {scale_asym_pc_4d_a_cf.shape}):\n {scale_asym_pc_4d_a_cf}")
    print(f"Per-channel Zero Point (Shape {zero_asym_pc_4d_a_cf.shape}):\n {zero_asym_pc_4d_a_cf}")

    # 示例 7: 4D 张量 - Per-channel Asymmetric INT8 (Conv Activation, channels-last)
    # is_weight=False, 但需要手动指定 channel_dim = 3
    print("\n--- 4D Tensor - Per-channel Asymmetric INT8 (Conv Activation, channels-last, is_weight=False, manual dim=3) ---")
    activation_4d_cl = torch.randn(1, 28, 28, 64) # Batch=1, H=28, W=28, Channels=64
    scale_asym_pc_4d_a_cl, zero_asym_pc_4d_a_cl = max_min_quantize_params(activation_4d_cl, bitwidth=8, quant_mode='asym', per_channel=True, is_weight=False, channel_dim=3)
    print(f"Input Shape: {activation_4d_cl.shape}")
    print(f"Per-channel Scale (Shape {scale_asym_pc_4d_a_cl.shape}):\n {scale_asym_pc_4d_a_cl}")
    print(f"Per-channel Zero Point (Shape {zero_asym_pc_4d_a_cl.shape}):\n {zero_asym_pc_4d_a_cl}")

    # 示例 8: 2D 张量 - Per-tensor Asymmetric INT8 (激活的 Per-tensor)
    print("\n--- 2D Tensor - Per-tensor Asymmetric INT8 (Activation) ---")
    activation_2d_pt = torch.randn(8, 32)
    scale_asym_pt_2d_a, zero_asym_pt_2d_a = max_min_quantize_params(activation_2d_pt, bitwidth=8, quant_mode='asym', per_channel=False, is_weight=False)
    print(f"Input Shape: {activation_2d_pt.shape}, Min: {activation_2d_pt.min().item():.4f}, Max: {activation_2d_pt.max().item():.4f}")
    print(f"Scale: {scale_asym_pt_2d_a.item():.6f}")
    print(f"Zero Point: {zero_asym_pt_2d_a.item():.4f}")

def test_mean_std_quantize_params():
    print("--- 示例使用 mean_std_quantize_params ---")

    # 示例 1: 2D 张量 - Per-tensor Symmetric INT8 (激活的常见情况)
    print("\n--- 2D Tensor - Per-tensor Symmetric INT8 (Activation) ---")
    tensor_2d_act = torch.randn(4, 16)
    scale_sym_pt_2d, zero_sym_pt_2d = mean_std_quantize_params(tensor_2d_act, bitwidth=8, quant_mode='sym', per_channel=False, is_weight=False, n_sigmas=3.0)
    print(f"Input Shape: {tensor_2d_act.shape}")
    print(f"Input Mean: {tensor_2d_act.mean().item():.4f}, Input Std: {tensor_2d_act.std(unbiased=False).item():.4f}")
    print(f"Calculated Range: [{tensor_2d_act.mean().item() - 3*tensor_2d_act.std(unbiased=False).item():.4f}, {tensor_2d_act.mean().item() + 3*tensor_2d_act.std(unbiased=False).item():.4f}]")
    print(f"Scale: {scale_sym_pt_2d.item():.6f}")
    print(f"Zero Point: {zero_sym_pt_2d}") # Should be None


    # 示例 2: 2D 张量 - Per-channel Symmetric INT8 (Linear Weight)
    # is_weight=True -> inferred channel_dim = 0
    print("\n--- 2D Tensor - Per-channel Symmetric INT8 (Linear Weight, is_weight=True -> inferred dim=0) ---")
    weight_2d_linear = torch.randn(4, 16) # 4 out_features, 16 in_features
    scale_sym_pc_2d_w, zero_sym_pc_2d_w = mean_std_quantize_params(weight_2d_linear, bitwidth=8, quant_mode='sym', per_channel=True, is_weight=True, n_sigmas=3.0)
    print(f"Input Shape: {weight_2d_linear.shape}")
    # print(f"Per-channel Mean (dim=0): {torch.mean(weight_2d_linear, dim=1)}") # 可以手动验证per-channel mean/std
    # print(f"Per-channel Std (dim=0): {torch.std(weight_2d_linear, dim=1, unbiased=False)}")
    print(f"Per-channel Scale (Shape {scale_sym_pc_2d_w.shape}):\n {scale_sym_pc_2d_w}")


    # 示例 3: 4D 张量 - Per-channel Asymmetric INT8 (Conv Activation, channels-first)
    # is_weight=False -> inferred channel_dim = 1
    print("\n--- 4D Tensor - Per-channel Asymmetric INT8 (Conv Activation, channels-first, is_weight=False -> inferred dim=1, n_sigmas=4) ---")
    activation_4d_cf = torch.randn(1, 64, 28, 28) # Batch=1, Channels=64, H=28, W=28
    scale_asym_pc_4d_a_cf, zero_asym_pc_4d_a_cf = mean_std_quantize_params(activation_4d_cf, bitwidth=8, quant_mode='asym', per_channel=True, is_weight=False, n_sigmas=4.0) # Use 4 sigmas
    print(f"Input Shape: {activation_4d_cf.shape}")
    print(f"Per-channel Scale (Shape {scale_asym_pc_4d_a_cf.shape}):\n {scale_asym_pc_4d_a_cf[::8]}") # Print subset for brevity
    print(f"Per-channel Zero Point (Shape {zero_asym_pc_4d_a_cf.shape}):\n {zero_asym_pc_4d_a_cf[::8]}") # Print subset


    # 示例 4: 全零张量 (Symmetric Per-channel)
    print("\n--- All Zeros Tensor (Per-channel Symmetric, inferred dim=0) ---")
    tensor_zeros_pc = torch.zeros(2, 5)
    scale_zeros_sym_pc, zero_zeros_sym_pc = mean_std_quantize_params(tensor_zeros_pc, bitwidth=8, quant_mode='sym', per_channel=True, is_weight=True, n_sigmas=3.0) # Inferred dim=0
    print(f"Input Shape: {tensor_zeros_pc.shape}")
    print(f"Per-channel Scale (Shape {scale_zeros_sym_pc.shape}):\n {scale_zeros_sym_pc}") # Should be 1e-8
    print(f"Per-channel Zero Point: {zero_zeros_sym_pc}") # Should be None


    # 示例 5: 常量非零张量 (Asymmetric Per-tensor)
    print("\n--- Constant Non-Zero Tensor (Per-tensor Asymmetric) ---")
    tensor_const = torch.full((3, 3), 5.0)
    scale_const_asym_pt, zero_const_asym_pt = mean_std_quantize_params(tensor_const, bitwidth=8, quant_mode='asym', per_channel=False, is_weight=False, n_sigmas=3.0)
    print(f"Input Value: {tensor_const.mean().item()}") # Mean is 5.0, Std is 0.0
    print(f"Scale: {scale_const_asym_pt.item():.4f}") # Should be 1.0
    print(f"Zero Point: {zero_const_asym_pt.item():.4f}") # Should be round(5.0 - 0) = 5 for 0-255 range, or similar based on mean
    # Verification: If Scale=1.0, ZeroPoint=round(mean_val - int_min_q) = round(5.0 - 0) = 5.0.
    # Quantized value Q = round(FP / Scale) + ZeroPoint = round(5.0 / 1.0) + 5.0 = 5.0 + 5.0 = 10.0.
    # Dequantized value FP = (Q - ZeroPoint) * Scale = (10.0 - 5.0) * 1.0 = 5.0. Correct.


    # 示例 6: 1D 张量 (Bias 或 Norm Weight) - Per-tensor Asymmetric INT8
    print("\n--- 1D Tensor (Bias/Norm Weight) - Per-tensor Asymmetric INT8 ---")
    norm_weight_1d = torch.randn(128) + 2.0 # Shifted mean
    scale_asym_pt_1d, zero_asym_pt_1d = mean_std_quantize_params(norm_weight_1d, bitwidth=8, quant_mode='asym', per_channel=False, is_weight=True, n_sigmas=3.0)
    print(f"Input Shape: {norm_weight_1d.shape}")
    print(f"Input Mean: {norm_weight_1d.mean().item():.4f}, Input Std: {norm_weight_1d.std(unbiased=False).item():.4f}")
    print(f"Scale: {scale_asym_pt_1d.item():.6f}")
    print(f"Zero Point: {zero_asym_pt_1d.item():.4f}")

if __name__ == '__main__':
    test_min_max_quantize_params()
    # test_mean_std_quantize_params()