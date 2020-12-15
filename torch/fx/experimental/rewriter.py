import ast
import inspect
import textwrap
import copy
from types import FunctionType
from typing import cast, Union, Callable
from torch.fx.symbolic_trace import Tracer
from torch.fx.graph import Graph
from torch.jit.frontend import normalize_source_lines
import torch

class AST_Rewriter(ast.NodeTransformer):
    """
    Take a FunctionType object representing a `forward` method, then
    perform an AST rewrite to swap out nodes that are not symbolically
    traceable with a callsite to the FX alternative.

    To support swapping out an AST node, define a new `visit` method on
    that node. For more details, see:
    https://docs.python.org/3/library/ast.html#ast.NodeTransformer
    """

    def rewrite(self, fn: FunctionType):

        # Normalize the source lines
        sourcelines, _ = inspect.getsourcelines(fn)
        sourcelines = normalize_source_lines(sourcelines)
        source = ''.join(sourcelines)
        normalized_str = textwrap.dedent(source)

        # Rewrite the original AST
        source_ast = ast.parse(normalized_str)
        dest_ast = ast.fix_missing_locations(self.visit(source_ast))

        # Pull out the compiled fucntion from the newly-created Module
        code = compile(dest_ast, "", "exec")
        globals_dict = copy.copy(fn.__globals__)
        keys_before = set(globals_dict.keys())
        exec(code, globals_dict)
        new_keys = list(set(globals_dict.keys()) - keys_before)
        assert len(new_keys) == 1
        fn_compiled = globals_dict[new_keys[0]]

        # Return the correct FunctionType object
        return fn_compiled

    def visit_Assert(self, node):
        """
        Swap out the Assert node (Python's `assert`) with a callsite to the
        symbolically-traceable torch._assert function
        """
        # Create the Call node
        n = ast.parse('torch._assert()', mode='eval')
        assert isinstance(n, ast.Expression)
        call_node = n.body
        assert isinstance(call_node, ast.Call)
        msg = node.msg if node.msg else ast.Constant(value="", kind=None)
        call_node.args = [node.test, msg]

        # Ensure that the new node conforms to the Python AST grammar
        expr_wrapper = ast.Expr(value=call_node)

        # Return the new Call node to signify that we want to use it as
        # a replacement for the original _assert node
        return ast.copy_location(expr_wrapper, node)


class RewritingTracer(Tracer):
    def trace(self, root: Union[torch.nn.Module, Callable]) -> Graph:
        return super().trace(_rewrite(root))


def _rewrite(fn : Union[torch.nn.Module, Callable]) -> Union[torch.nn.Module, Callable]:
    if isinstance(fn, torch.nn.Module):
        # Rewrite this module's forward() and all of its recursive children's
        # forward. Return the new rewritten module hierarchy.
        def rewrite_module(m : torch.nn.Module):
            class RewrittenModule(torch.nn.Module):
                def __init__(self, orig):
                    super().__init__()
                    self.__dict__ = copy.copy(orig.__dict__)
            RewrittenModule.forward = AST_Rewriter().rewrite(cast(FunctionType, m.forward))
            new_m = RewrittenModule(m)
            for name, child in new_m.named_children():
                new_m[name] = rewrite_module(child)    # type: ignore
            return new_m
        return rewrite_module(fn)
    else:
        # Rewrite this single free function
        return AST_Rewriter().rewrite(cast(FunctionType, fn))
