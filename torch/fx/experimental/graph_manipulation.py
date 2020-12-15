import json
from typing import Dict, List, NamedTuple, Any

import torch
from torch.fx.experimental.shape_prop import ShapeProp
from torch.fx.experimental.param_fetch import lift_lowering_attrs_to_nodes
from torch.fx.graph import Graph, get_qualified_name
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, Target, map_arg


def replace_target_nodes_with(
    fx_module: GraphModule,
    old_op: str,
    old_target: Target,
    new_op: str,
    new_target: Target,
):
    """Modifies all nodes in fx_module.graph.nodes which match the specified op code and target,
    and updates them to match the new op code and target"""
    new_graph = Graph()
    val_map: Dict[Node, Node] = {}
    for node in fx_module.graph.nodes:
        if node.op == old_op and node.target == old_target:
            args = map_arg(node.args, lambda n: val_map[n])
            kwargs = map_arg(node.kwargs, lambda n: val_map[n])
            assert isinstance(args, tuple)
            assert isinstance(kwargs, dict)
            val_map[node] = new_graph.create_node(
                new_op, new_target, args, kwargs, node.name
            )
        else:
            val_map[node] = new_graph.node_copy(node, lambda n: val_map[n])
    fx_module.graph = new_graph


class size_bytes(NamedTuple):
    output_size: int
    total_size: int


def get_size_of_all_nodes(fx_module: GraphModule, args: List[torch.Tensor]) -> None:
    """Given a fx graph module, update each node with its total size (weights + bias + output)
    and its output_size(output). For a non-module node, the total size is the output size.
    return total size"""
    # Mark shape and dtype for each node (node.shape and node.dtype)
    ShapeProp(fx_module).propagate(*args)
    # Calculate the total size of the whole fx graph
    total_size_of_graph = 0.0
    for node in fx_module.graph.nodes:
        if node.op == "output":
            break
        node.size_bytes = get_size_of_node(fx_module, node)
    return


def get_size_of_node(fx_module: GraphModule, node: Node) -> size_bytes:
    """Given a node with node.dtype and node.shape, return its total size and its output size.
    total_size = weights + bias + output_size
    """
    # Total num of elements
    total_num_of_elems = 0
    # For a module, conside all parameters
    if node.op == "call_module":
        submodule_dict = dict(fx_module.named_modules())
        submodule = submodule_dict[node.target]
        parameters = submodule.named_parameters()
        # Parameters are named tuples
        for name, p in parameters:
            total_num_of_elems += p.numel()
    # Don't forget the output size
    # node.shape is the shape of this node's output
    shape = getattr(node, "shape", None)
    if shape:
        output_elem = shape.numel()
    else:
        raise RuntimeError("Node has no shape attr")
    total_num_of_elems += output_elem
    size_per_elem_bytes = 0
    dtype = getattr(node, "dtype", None)
    if dtype:
        size_per_elem_bytes = torch.tensor([], dtype=dtype).element_size()
    else:
        raise RuntimeError("Node has no dtype attr")
    total_size = size_per_elem_bytes * total_num_of_elems
    output_size = size_per_elem_bytes * output_elem
    return size_bytes(output_size, total_size)


def serialize_shape(shape: torch.Size) -> str:
    return str(list(shape))


def serialize_tensor_quantization(tensor: torch.Tensor) -> Dict[str, Any]:
    scheme: Dict[str, Any] = {}
    if tensor.is_quantized:
        scheme["q_scheme"] = str(tensor.qscheme())
        if tensor.qscheme() in {torch.per_tensor_affine, torch.per_tensor_symmetric}:
            scheme["q_scale"] = tensor.q_scale()
            scheme["q_zero_pont"] = tensor.q_zero_point()
        if tensor.qscheme() in {
            torch.per_channel_affine,
            torch.per_channel_affine_float_qparams,
            torch.per_channel_symmetric,
        }:
            scheme["q_per_channel_scales"] = tensor.q_per_channel_scales().tolist()
            scheme[
                "q_per_channel_zero_points"
            ] = tensor.q_per_channel_zero_points().tolist()
            scheme["q_per_channel_axis"] = tensor.q_per_channel_axis()

    return scheme


def serialize_weight(tensor: torch.Tensor) -> Dict:
    weight: Dict[str, Any] = {}
    weight["dtype"] = str(tensor.dtype)
    weight["is_quantized"] = tensor.is_quantized
    if tensor.is_quantized:
        weight["quantized_type"] = serialize_tensor_quantization(tensor)
    weight["shape"] = serialize_shape(tensor.shape)
    return weight


def serialize_leaf_module(
    node: Node, weights_metadata: Dict, weights: Dict, name_prefix: str
) -> Dict:
    parameters: Dict[str, Any] = {}

    for p_name, p_value in node.attrs_for_lowering.items():  # type: ignore
        if isinstance(p_value, torch.Tensor):
            weights_metadata[f"{name_prefix}.{p_name}"] = serialize_weight(p_value)
            weights[f"{name_prefix}.{p_name}"] = p_value
        else:
            parameters[p_name] = str(p_value)

    return parameters


def serialize_module(fx_module: GraphModule, weights: Dict, name_prefix="") -> Dict:
    """Recursively Serializes a graph module (fx_module) to a dictionary which is later exported to JSON.
    It also adds all weights the provided weights dictionary by qualified_name.
    Dictionary Schema:
    MODULE
    {
        modules: {module_name: MODULE],
        nodes: [NODE],
        weights {qualified_name: WEIGHT},
    }
    NODE
    {
        shape: [],
        dtype: dtype,
        target: target,
        op_code: op_code,
        name: name,
        args: [],
        kwargs: {}
    }
    WEIGHT
    {
        dtype: dtype,
        is_quantized: bool,
        shape: [],
        quantization_info: QUANTIZATION
    }
    QUANTIZATION
    {
        qscheme: qscheme,
        q_scale: float,
        q_zero_point: float,
        q_per_channel_scales, [],
        q_per_channel_zero_points: [],
        q_per_channel_axis, int
    }
    """
    serialized_dict: Dict[str, Any] = {}
    serialized_dict["modules"] = {}
    serialized_dict["weights"] = {}
    serialized_dict["nodes"] = []
    parameters = fx_module.named_parameters()
    prefix = f"{name_prefix}." if name_prefix else ""
    submodules = dict(fx_module.named_modules())
    for name, p in parameters:
        if isinstance(p, torch.Tensor):
            weight = serialize_weight(p)
            serialized_dict["weights"][prefix + name] = weight
            weights[prefix + name] = p
    lift_lowering_attrs_to_nodes(fx_module)
    for node in fx_module.graph.nodes:
        node_rep: Dict[str, Any] = {}
        # Get shape/type info, currently not needed for call_module.
        if node.op != "call_module" or not isinstance(
            submodules[node.target], GraphModule
        ):
            shape = getattr(node, "shape", None)
            if shape:
                node_rep["shape"] = serialize_shape(shape)
            else:
                raise RuntimeError(
                    "Node has no shape attr, this is likely because shape propagation has not been run on this Graph."
                )
            dtype = getattr(node, "dtype", None)
            if dtype:
                node_rep["dtype"] = str(dtype)
            else:
                raise RuntimeError(
                    "Node has no dtype attr, this is likely because shape propagation has not been run on this Graph."
                )

        # Recurse down into any submodules we are calling.
        if node.op == "call_module":
            if isinstance(submodules[node.target], GraphModule):
                serialized_module = serialize_module(
                    getattr(fx_module, node.target), weights, node.target
                )
                serialized_dict["modules"][node.target] = serialized_module
            else:
                node_rep["parameters"] = serialize_leaf_module(
                    node,
                    serialized_dict["weights"],
                    weights,
                    prefix + node.target,
                )

        if node.op == "call_function":
            node_rep["target"] = get_qualified_name(node.target)
        else:
            node_rep["target"] = str(node.target)

        # Make sure we capture all constants.
        if node.op == "get_attr":
            target = getattr(fx_module, node.target)
            qualname = prefix + node.target
            if isinstance(target, torch.Tensor) and qualname not in weights:
                weight = serialize_weight(target)
                serialized_dict["weights"][prefix + node.target] = weight
                weights[prefix + node.target] = target

        node_rep["op_code"] = node.op
        node_rep["name"] = node.name
        node_rep["args"] = map_arg(
            node.args, lambda arg: {"is_node": True, "name": str(arg)}
        )
        node_rep["kwargs"] = map_arg(
            node.kwargs, lambda arg: {"is_node": True, "name": str(arg)}
        )
        serialized_dict["nodes"] += [node_rep]

    return serialized_dict


class AcceleratedGraphModule:
    def __init__(self, fx_module: GraphModule):
        """Creates the needed data structures to pass to the glow runtime"""
        self.weights: Dict[str, Any] = {}
        self.serialized_graph = serialize_module(fx_module, self.weights)
        self.serialized_graph_json = json.dumps(self.serialized_graph, indent=4)
