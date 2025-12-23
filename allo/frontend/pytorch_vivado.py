# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=too-many-public-methods

import inspect

try:
    import torch
    from torch.fx.graph_module import GraphModule
    from torch.fx.passes.shape_prop import ShapeProp
    from .tracer import AlloTracer
except ImportError:
    pass
from .. import dsl 
from ..ir import types
from ..customize import customize
from .pytorch import TorchBuilder, _process_quantized_params


def from_pytorch_vivado(
    model,
    example_inputs,
    leaf_modules=None,
    verbose=False,
    enable_tensor=False,
    target="llvm",
    mode="csim",
    project="top.prj",
):
    sig = inspect.signature(model.forward)
    input_names = [
        p.name for i, p in enumerate(sig.parameters.values()) if i < len(example_inputs)
    ]
    concrete_args = {
        p.name: p.default for p in sig.parameters.values() if p.name not in input_names
    }
    args = []
    args += example_inputs
    for item in concrete_args.values():
        args.append(item)

    tracer = AlloTracer(model, concrete_args=concrete_args, leaf_modules=leaf_modules)
    graph = tracer.trace()
    name = (
        model.__class__.__name__
        if isinstance(model, torch.nn.Module)
        else model.__name__
    )
    gm = GraphModule(tracer.root, graph, name)
    # gm.print_readable()
    ShapeProp(gm).propagate(*args)
    if verbose:
        print(gm.graph)
    global_vars = {}
    for pymod in (types,):
        global_vars.update({item[0]: item[1] for item in inspect.getmembers(pymod)})
    global_vars.update({"dsl": dsl})
    for name, param in gm.named_parameters():
        new_name = "g_" + name.replace(".", "_")
        global_vars.update({new_name: param.detach().numpy()})
    # for name, buffer in gm.named_buffers():
    #     new_name = "g_" + name.replace(".", "_")
    #     global_vars.update({new_name: buffer.detach().numpy()})

    # ========== 关键插入点：处理量化参数 ==========
    # 在这里调用量化参数处理函数，将浮点权重/偏置替换为整数版本
    # 并注入所有scale/zero常量
    _process_quantized_params(gm, global_vars)

    builder = TorchBuilder(gm, example_inputs, leaf_modules)
    code = builder.build()
    print(code)
    s = customize(
        code, verbose=verbose, global_vars=global_vars, enable_tensor=enable_tensor
    )
    # print(s.module)
    mod = s.build(target='vivado')
    # print(mod)
    if verbose:
        print(s.module)
    return mod
