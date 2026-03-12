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
    quant_config=None,
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
    # Pass SCALE_FIXED_BITS to builder for quantization
    global_vars.update({"__allo_quant_fixed_bits__": quant_config.scale_fixed_bits})
    # PATH
    global_vars.update({"__allo_target_path__": project})

    # Assume batch size always comes from the left-most dimension of example_inputs[0]
    # and propagate it through the compilation pipeline to MLIR as a module attribute.
    batch = 1
    if example_inputs is not None and len(example_inputs) > 0:
        first_inp = example_inputs[0]
        if hasattr(first_inp, "shape") and len(first_inp.shape) > 0:
            batch = int(first_inp.shape[0])
    global_vars.update({"__allo_batch__": batch})

    # Propagate hidden dimension (e.g., transformer embedding dim) through the
    # compilation pipeline to MLIR as a module attribute (allo.hidden_dim).
    # Prefer reading it from LayerNorm/IntLayerNorm modules; fall back to (B,L,D)
    # style example input shapes when applicable.
    hidden_dim = None
    modules_dict = dict(gm.named_modules())

    # 1) IntLayerNorm commonly exposes `hidden_dim` explicitly.
    for _, m in modules_dict.items():
        if hasattr(m, "hidden_dim") and m.hidden_dim is not None:
            try:
                hidden_dim = int(m.hidden_dim)
                if hidden_dim > 0:
                    break
            except Exception:
                pass

    # 2) Fallback: nn.LayerNorm (and some wrappers) expose normalized_shape.
    if hidden_dim is None:
        for _, m in modules_dict.items():
            if hasattr(m, "normalized_shape"):
                ns = getattr(m, "normalized_shape")
                if isinstance(ns, (tuple, list)) and len(ns) == 1:
                    try:
                        hidden_dim = int(ns[0])
                        if hidden_dim > 0:
                            break
                    except Exception:
                        pass

    # 3) Last resort: if example input looks like (B, L, D) or (L, D).
    if hidden_dim is None and example_inputs is not None and len(example_inputs) > 0:
        first_inp = example_inputs[0]
        if hasattr(first_inp, "shape") and first_inp.shape is not None:
            shp = tuple(first_inp.shape)
            if len(shp) in (2, 3):
                try:
                    hidden_dim = int(shp[-1])
                except Exception:
                    hidden_dim = None

    if hidden_dim is not None:
        global_vars.update({"__allo_hidden_dim__": int(hidden_dim)})

    for name, param in gm.named_parameters():
        new_name = "g_" + name.replace(".", "_")
        global_vars.update({new_name: param.detach().numpy()})
    
    # for name, buffer in gm.named_buffers():
    #     new_name = "g_" + name.replace(".", "_")
    #     global_vars.update({new_name: buffer.detach().numpy()})

    # ========== 关键插入点：处理量化参数 ==========
    # 在这里调用量化参数处理函数，将浮点权重/偏置替换为整数版本
    # 并注入所有scale/zero常量
    _process_quantized_params(gm, global_vars, quant_config)

    builder = TorchBuilder(gm, example_inputs, leaf_modules, quant_act_bits=8, quant_weight_bits=8, quant_bias_bits=8, enable_quant=True, quant_config=quant_config)
    code = builder.build()
    print(code)
    s = customize(
        code, verbose=verbose, global_vars=global_vars, enable_tensor=enable_tensor
    )
    # print(s.module)
    mod = s.build(target='vivado', mode=mode, project=project)
    # print(mod)
    if verbose:
        print(s.module)
    return mod
