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
    def _infer_batch_from_example_inputs(inputs):
        # Prefer the left-most dimension of the first tensor-like input.
        if inputs is None or len(inputs) == 0:
            return 1

        def _walk_first_tensor_like(obj):
            if hasattr(obj, "shape"):
                return obj
            if isinstance(obj, (list, tuple)):
                for item in obj:
                    found = _walk_first_tensor_like(item)
                    if found is not None:
                        return found
            return None

        first = _walk_first_tensor_like(inputs[0])
        if first is None or not hasattr(first, "shape"):
            return 1

        try:
            if len(first.shape) > 0:
                batch_val = int(first.shape[0])
                return batch_val if batch_val > 0 else 1
        except Exception:
            pass
        return 1

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

    # Propagate batch size through the compilation pipeline to MLIR as a
    # module attribute (allo.batch).
    batch = _infer_batch_from_example_inputs(example_inputs)
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

    # Propagate sequence length through the compilation pipeline to MLIR as
    # a module attribute (allo.seqlen).
    # Prefer reading explicit attributes from modules; then infer from shapes.
    seqlen = None

    # 1) Explicit sequence-length attributes on modules (if present).
    for _, m in modules_dict.items():
        if hasattr(m, "seq_len") and m.seq_len is not None:
            try:
                seqlen = int(m.seq_len)
                if seqlen > 0:
                    break
            except Exception:
                pass
        if hasattr(m, "seqlen") and m.seqlen is not None:
            try:
                seqlen = int(m.seqlen)
                if seqlen > 0:
                    break
            except Exception:
                pass

    # 2) Fallback from example input shapes.
    #    For (B, L, D), use L; for (L, D), use L.
    if seqlen is None and example_inputs is not None and len(example_inputs) > 0:
        first_inp = example_inputs[0]
        if hasattr(first_inp, "shape") and first_inp.shape is not None:
            shp = tuple(first_inp.shape)
            try:
                if len(shp) >= 3:
                    seqlen = int(shp[-2])
                elif len(shp) == 2:
                    seqlen = int(shp[0])
            except Exception:
                seqlen = None

    # NOTE TODO: seqlen automatically infer
    seqlen = 197 # Currently hard-coded to 197 for ViT.

    if seqlen is not None and seqlen > 0:
        global_vars.update({"__allo_seqlen__": int(seqlen)})

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
