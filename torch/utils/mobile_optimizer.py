"""
This module contains utility method for mobile model optimization and lint.
"""

import torch
from enum import Enum
from torch._C import MobileOptimizerType
from typing import Set, List, AnyStr

class LintCode(Enum):
    BUNDLED_INPUT = 1
    REQUIRES_GRAD = 2
    DROPOUT = 3
    BATCHNORM = 4

def optimize_for_mobile(
        script_module,
        optimization_blocklist: Set[MobileOptimizerType] = None,
        preserved_methods: List[AnyStr] = None,
        backend: str = 'CPU'):
    """
    Args:
        script_module: An instance of torch script module with type of ScriptModule.
        optimization_blocklist: A set with type of MobileOptimizerType. When set is not passed,
            optimization method will run all the optimizer pass; otherwise, optimizer
            method will run the optimization pass that is not included inside optimization_blocklist.
        perserved_methods: A list of methods that needed to be preserved when freeze_module pass is invoked
        backend: Device type to use for running the result model ('CPU'(default), 'Vulkan' or 'Metal').
    Returns:
        A new optimized torch script module
    """
    if not isinstance(script_module, torch.jit.ScriptModule):
        raise TypeError(
            'Got {}, but ScriptModule is expected.'.format(type(script_module)))

    if optimization_blocklist is None:
        optimization_blocklist = set()

    if preserved_methods is None:
        preserved_methods = []

    backend = backend.lower()
    if backend == 'cpu':
        optimized_cpp_module = torch._C._jit_pass_optimize_for_mobile(script_module._c, optimization_blocklist, preserved_methods)
    elif backend == 'vulkan':
        optimized_cpp_module = torch._C._jit_pass_vulkan_optimize_for_mobile(script_module._c, preserved_methods)
    elif backend == 'metal':
        optimized_cpp_module = torch._C._jit_pass_metal_optimize_for_mobile(script_module._c, preserved_methods)
    else:
        raise TypeError("Unknown backend, must be one of 'CPU', 'Vulkan' or 'Metal'")

    return torch.jit._recursive.wrap_cpp_module(optimized_cpp_module)


def generate_mobile_module_lints(script_module: torch.jit.ScriptModule):
    """
    Args:
        script_module: An instance of torch script module with type of ScriptModule

    Returns:
        lint_map: A list of dictionary that contains modules lints
    """
    if not isinstance(script_module, torch.jit.ScriptModule):
        raise TypeError(
            'Got {}, but ScriptModule is expected.'.format(type(script_module)))

    lint_list = []

    if not hasattr(script_module, "_generate_bundled_inputs"):
        lint_list.append({"name": LintCode.BUNDLED_INPUT.name, "message": "No bundled input, please add bundled inputs before "
                          "saving the module using torch.utils.bundled_inputs.augment_model_with_bundled_inputs."})

    for name, param in script_module.named_parameters():
        if param.requires_grad:
            lint_list.append({"name": LintCode.REQUIRES_GRAD.name, "message": "Param {} requires grad, "
                             "please set torch.no_grad() to reduce memory usage and improve computation speed during "
                              "inference phase.".format(name)})

    op_names = torch.jit.export_opnames(script_module)
    for op_name in op_names:
        if "dropout" in op_name:
            lint_list.append({"name": LintCode.DROPOUT.name, "message": "Operator {} exists, remember to call eval() before "
                              "saving the module.and call torch.utils.mobile_optimizer.optimize_for_mobile to drop dropout "
                              "operator.".format(op_name)})
        if "batch_norm" in op_name:
            lint_list.append({"name": LintCode.BATCHNORM.name, "message": "Operator {} exists, remember to call eval() before "
                              "saving the module and call torch.utils.mobile_optimizer.optimize_for_mobile to drop batch_norm "
                              "operator.".format(op_name)})

    return lint_list
