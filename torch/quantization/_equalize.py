import torch
import copy
from typing import Dict, Any

_supported_types = {torch.nn.Conv2d, torch.nn.Linear}

def max_over_ndim(input, axis_list, keepdim=False):
    ''' Applies 'torch.max' over the given axises
    '''
    axis_list.sort(reverse=True)
    for axis in axis_list:
        input, _ = input.max(axis, keepdim)
    return input

def min_over_ndim(input, axis_list, keepdim=False):
    ''' Applies 'torch.min' over the given axises
    '''
    axis_list.sort(reverse=True)
    for axis in axis_list:
        input, _ = input.min(axis, keepdim)
    return input

def channel_range(input, axis=0):
    ''' finds the range of weights associated with a specific channel
    '''
    size_of_tensor_dim = input.ndim
    axis_list = list(range(size_of_tensor_dim))
    axis_list.remove(axis)

    mins = min_over_ndim(input, axis_list)
    maxs = max_over_ndim(input, axis_list)

    assert mins.size(0) == input.size(axis), "Dimensions of resultant channel range does not match size of requested axis"
    return maxs - mins

def cross_layer_equalization(module1, module2, output_axis=0, input_axis=1):
    ''' Given two adjacent tensors', the weights are scaled such that
    the ranges of the first tensors' output channel are equal to the
    ranges of the second tensors' input channel
    '''
    if type(module1) not in _supported_types or type(module2) not in _supported_types:
        raise ValueError("module type not supported:", type(module1), " ", type(module2))

    if module1.weight.size(output_axis) != module2.weight.size(input_axis):
        raise TypeError("Number of output channels of first arg do not match \
        number input channels of second arg")

    weight1 = module1.weight
    weight2 = module2.weight
    bias = module1.bias

    weight1_range = channel_range(weight1, output_axis)
    weight2_range = channel_range(weight2, input_axis)

    # producing scaling factors to applied
    weight2_range += 1e-9
    scaling_factors = torch.sqrt(weight1_range / weight2_range)
    inverse_scaling_factors = torch.reciprocal(scaling_factors)

    bias = bias * inverse_scaling_factors

    # formatting the scaling (1D) tensors to be applied on the given argument tensors
    # pads axis to (1D) tensors to then be broadcasted
    size1 = [1] * weight1.ndim
    size1[output_axis] = weight1.size(output_axis)
    size2 = [1] * weight2.ndim
    size2[input_axis] = weight2.size(input_axis)

    scaling_factors = torch.reshape(scaling_factors, size2)
    inverse_scaling_factors = torch.reshape(inverse_scaling_factors, size1)

    weight1 = weight1 * inverse_scaling_factors
    weight2 = weight2 * scaling_factors

    module1.weight = torch.nn.Parameter(weight1)
    module1.bias = torch.nn.Parameter(bias)
    module2.weight = torch.nn.Parameter(weight2)

def equalize(model, paired_modules_list, threshold=1e-4, inplace=True):
    ''' Given a list of adjacent modules within a model, equalization will
    be applied between each pair, this will repeated until convergence is achieved

    Keeps a copy of the changing modules from the previous iteration, if the copies
    are not that different than the current modules (determined by converged_test),
    then the modules have converged enough that further equalizing is not necessary

    Implementation of this referced section 4.1 of this paper https://arxiv.org/pdf/1906.04721.pdf

    Args:
        model: a model (nn.module) that equalization is to be applied on
        paired_modules_list: a list of lists where each sublist is a pair of two
            submodules found in the model, for each pair the two submodules generally
            have to be adjacent in the model to get expected/reasonable results
        threshold: a number used by the converged function to determine what degree
            similarity between models is necessary for them to be called equivalent
        inplace: determines if function is inplace or not
    '''
    if not inplace:
        model = copy.deepcopy(model)

    name_to_module : Dict[str, torch.nn.Module] = {}
    previous_name_to_module: Dict[str, Any] = {}
    name_set = {name for pair in paired_modules_list for name in pair}

    for name, module in model.named_modules():
        if name in name_set:
            name_to_module[name] = module
            previous_name_to_module[name] = None
    while not converged(name_to_module, previous_name_to_module, threshold):
        for pair in paired_modules_list:
            previous_name_to_module[pair[0]] = copy.deepcopy(name_to_module[pair[0]])
            previous_name_to_module[pair[1]] = copy.deepcopy(name_to_module[pair[1]])

            cross_layer_equalization(name_to_module[pair[0]], name_to_module[pair[1]])

    return model

def converged(curr_modules, prev_modules, threshold=1e-4):
    ''' Tests for the summed norm of the differences between each set of modules
    being less than the given threshold

    Takes two dictionaries mapping names to modules, the set of names for each dictionary
    should be the same, looping over the set of names, for each name take the differnce
    between the associated modules in each dictionary

    '''
    if curr_modules.keys() != prev_modules.keys():
        raise ValueError("The keys to the given mappings must have the same set of names of modules")

    summed_norms = torch.tensor(0.)
    if None in prev_modules.values():
        return False
    for name in curr_modules.keys():
        difference = curr_modules[name].weight.sub(prev_modules[name].weight)
        summed_norms += torch.norm(difference)
    return bool(summed_norms < threshold)
