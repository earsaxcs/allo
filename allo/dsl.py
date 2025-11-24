# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=redefined-builtin, unused-argument

import numpy as np


def grid(*args, name=None):
    return np.ndindex(*args)


def reduction(*args, name=None):
    return np.ndindex(*args)


def matmul(lhs, rhs, name=None):
    return np.matmul(lhs, rhs)


def bmm(lhs, rhs, name=None):
    return np.einsum("ijk,ikn->ijn", lhs, rhs)


def add(lhs, rhs, name=None):
    return lhs + rhs


def sub(lhs, rhs, name=None):
    return lhs - rhs


def div(lhs, rhs, name=None):
    return lhs / rhs


def copy(x, name=None):
    return np.copy(x)


def transpose(x, axes, name=None):
    return np.transpose(x, axes)

def floor(x, name=None):
    # Note this returns a float not int, but it's what we want for fake quant
    return np.floor(x)

def exp(x, name=None):
    return np.exp(x)


def log(x, name=None):
    return np.log(x)


def log2(x, name=None):
    return np.log2(x)


def log10(x, name=None):
    return np.log10(x)


def abs(x, name=None):
    return np.abs(x)


def softmax(x, name=None):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def sqrt(x, name=None):
    return np.sqrt(x)


def sin(x, name=None):
    return np.sin(x)


def cos(x, name=None):
    return np.cos(x)


def tan(x, name=None):
    return np.tan(x)


def tanh(x, name=None):
    return np.tanh(x)


def power(x, y, name=None):
    return np.power(x, y)


def relu(x, name=None):
    return np.maximum(x, 0)

def clamp(x, min=-2147483648, max=2147483647, name=None):
    return np.clip(x, min, max)

def clamp_max(x, max=2147483647, name=None):
    return np.clip(x, -2147483648, max)

def mean(x, axis=None, keepdim=False, name=None):
    return np.mean(x, axis=axis, keepdims=keepdim)

def sum(x, axis=None, keepdim=False, name=None):
    return np.sum(x, axis=axis, keepdims=keepdim)

def var(x, axis=None, keepdim=False, name=None):
    return np.var(x, axis=axis, keepdims=keepdim)

def max(x, axis=None, keepdim=False, name=None):    
    return np.max(x, axis=axis, keepdims=keepdim)

def min(x, axis=None, keepdim=False, name=None):    
    return np.min(x, axis=axis, keepdims=keepdim)

def conv2d(inp, filter, _stride, bias=None, name=None):
    view_shape = (
        tuple(inp.shape[:2]) # B, IC
        + tuple(np.subtract(inp.shape[2:], filter.shape[2:]) // _stride + 1) # (H - KH) / S + 1, (W - KW) / S + 1
        + filter.shape[2:] # KH, KW
    )
    strides = inp.strides[:2] + tuple([x * int(_stride[idx]) for idx, x in enumerate(inp.strides[2:])]) + inp.strides[2:]
    sub_matrices = np.lib.stride_tricks.as_strided(inp, view_shape, strides)
    if bias is None: 
        return np.einsum("fcij,nchwij->nfhw", filter, sub_matrices)
    return np.einsum("fcij,nchwij->nfhw", filter, sub_matrices) + np.broadcast_to(bias, shape=sub_matrices.shape[:1] + sub_matrices.shape[2:4] + bias.shape).transpose(0, 3, 1, 2)


def maxpool(inp, filter, name=None):
    view_shape = (
        tuple(inp.shape[:2])
        + tuple(np.subtract(inp.shape[2:], filter.shape) + 1)
        + filter.shape
    )
    strides = inp.strides[:2] + inp.strides[2:] + inp.strides[2:]
    sub_matrices = np.lib.stride_tricks.as_strided(inp, view_shape, strides)
    return np.max(sub_matrices, axis=(4, 5))


def sumpool(inp, filter, name=None):
    view_shape = (
        tuple(inp.shape[:2])
        + tuple(np.subtract(inp.shape[2:], filter.shape) + 1)
        + filter.shape
    )
    strides = inp.strides[:2] + inp.strides[2:] + inp.strides[2:]
    sub_matrices = np.lib.stride_tricks.as_strided(inp, view_shape, strides)
    return np.sum(sub_matrices, axis=(4, 5))


def linear(X, A, bias=None, name=None):
    if bias is None:
        return matmul(X, A.T)
    return matmul(X, A.T) + bias


def view(x, shape, name=None):
    return np.reshape(x, shape)

def expand(x, shape, name=None):
    return np.broadcast_to(x, shape)

def layernorm(x, gamma, beta, eps: float = 1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x = gamma * (x - mean) / np.sqrt(variance + eps) + beta
    return x


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def ones(shape, dtype=None):
    return np.ones(shape, dtype=dtype)


def zeros(shape, dtype=None):
    return np.zeros(shape, dtype=dtype)


def tril(x):
    return np.tril(x)


def concat(x, y, axis=0):
    return np.concatenate((x, y), axis=axis)

# quant op
# TODO: here just remain as placeholder

def qlinear(X, W, fscl_sign, fscl_coe, fscl_rshift, iscl_sign, iscl_coe, iscl_rshift, 
            oscl_sign, oscl_coe, oscl_rshift, wscl_sign, wscl_coe, wscl_rshift, 
            bscl_sign=None, bscl_coe=None, bscl_rshift=None, izr=None, ozr=None, bias=None, name=None):
    if bias is None:
        return matmul(X, W.T)
    return matmul(X, W.T) + bias

def qconv2d(inp, filter, _stride, fscl_sign, fscl_coe, fscl_rshift, iscl_sign, iscl_coe, iscl_rshift,
            oscl_sign, oscl_coe, oscl_rshift, wscl_sign, wscl_coe, wscl_rshift,
            bscl_sign=None, bscl_coe=None, bscl_rshift=None, izr=None, ozr=None, bias=None, name=None):
    # TODO: implement quantization logic with scales and zero points
    view_shape = (
        tuple(inp.shape[:2]) # B, IC
        + tuple(np.subtract(inp.shape[2:], filter.shape[2:]) // _stride + 1) # (H - KH) / S + 1, (W - KW) / S + 1
        + filter.shape[2:] # KH, KW
    )
    strides = inp.strides[:2] + tuple([x * int(_stride[idx]) for idx, x in enumerate(inp.strides[2:])]) + inp.strides[2:]
    sub_matrices = np.lib.stride_tricks.as_strided(inp, view_shape, strides)
    if bias is None: 
        return np.einsum("fcij,nchwij->nfhw", filter, sub_matrices)
    return np.einsum("fcij,nchwij->nfhw", filter, sub_matrices) + np.broadcast_to(bias, shape=sub_matrices.shape[:1] + sub_matrices.shape[2:4] + bias.shape).transpose(0, 3, 1, 2)

def int_gelu(x, input_scale_sign, input_scale_coe, input_scale_rshift,
          gelu_scale_sign, gelu_scale_coe, gelu_scale_rshift,
          fused_scale_sign, fused_scale_coe, fused_scale_rshift,
          output_scale_sign, output_scale_coe, output_scale_rshift, 
          input_zero=None, output_zero=None, name=None):
    """IntGELU 的 DSL 包装器 - 接受定点化的 scale 参数
    Args:
        x: 输入张量
        input_scale: 输入量化scale (sign, coe, rshift)
        gelu_scale: GELU内部scale (sign, coe, rshift)
        fused_scale: 融合scale = gelu_scale / output_scale (sign, coe, rshift)
        output_scale: 输出量化scale (sign, coe, rshift)
        input_zero: 输入零点 (可选)
        output_zero: 输出零点 (可选)
    """
    pass

def int_softmax(x, input_scale_sign, input_scale_coe, input_scale_rshift,
             softmax_scale_sign, softmax_scale_coe, softmax_scale_rshift,
             output_scale_sign, output_scale_coe, output_scale_rshift,
             fused_scale_sign=None, fused_scale_coe=None, fused_scale_rshift=None,
             input_zero=None, output_zero=None, name=None):
    """IntSoftmax 的 DSL 包装器 - 接受定点化的 scale 参数"""
    pass

def int_layernorm(x, bias_int, 
               input_scale_sign, input_scale_coe, input_scale_rshift,
               layernorm_scale_sign, layernorm_scale_coe, layernorm_scale_rshift,
               bias_scale_sign, bias_scale_coe, bias_scale_rshift,
               fused_scale_sign, fused_scale_coe, fused_scale_rshift,
               output_scale_sign, output_scale_coe, output_scale_rshift,
               input_zero=None, output_zero=None, eps: float = 1e-5):
    """IntLayerNorm 的 DSL 包装器 - 接受定点化的 scale 参数"""
    pass

def qmatmul(lhs, rhs, 
            x_scale_sign, x_scale_coe, x_scale_rshift,
            y_scale_sign, y_scale_coe, y_scale_rshift,
            o_scale_sign, o_scale_coe, o_scale_rshift,
            x_zero=None, y_zero=None, o_zero=None, name=None):
    """QMatMul 的 DSL 包装器 - 接受定点化的 scale 参数"""
    pass

def qmatmul_isqrtd(lhs, rhs, 
            x_scale_sign, x_scale_coe, x_scale_rshift,
            y_scale_sign, y_scale_coe, y_scale_rshift,
            o_scale_sign, o_scale_coe, o_scale_rshift,
            x_zero=None, y_zero=None, o_zero=None, name=None):
    """QMatMul 的 DSL 包装器 - 接受定点化的 scale 参数"""
    pass
def qadd(lhs, rhs, 
         x_scale_sign, x_scale_coe, x_scale_rshift,
         y_scale_sign, y_scale_coe, y_scale_rshift,
         o_scale_sign, o_scale_coe, o_scale_rshift,
         x_zero=None, y_zero=None, o_zero=None, name=None):
    """QAdd 的 DSL 包装器 - 接受定点化的 scale 参数"""
    pass