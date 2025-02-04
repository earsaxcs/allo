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
