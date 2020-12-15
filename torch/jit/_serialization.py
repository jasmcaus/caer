"""Serialization

This module contains functionality for serializing TorchScript modules, notably:
    * torch.jit.save
    * torch.jit.load

This is not intended to be imported directly; please use the exposed
functionalities in `torch.jit`.
"""
import os
import pathlib

import torch
from torch._six import string_classes
from torch.jit._recursive import wrap_cpp_module
from torch.serialization import validate_cuda_device


def save(m, f, _extra_files=None):
    r"""
    Save an offline version of this module for use in a separate process. The
    saved module serializes all of the methods, submodules, parameters, and
    attributes of this module. It can be loaded into the C++ API using
    ``torch::jit::load(filename)`` or into the Python API with
    :func:`torch.jit.load <torch.jit.load>`.

    To be able to save a module, it must not make any calls to native Python
    functions.  This means that all submodules must be subclasses of
    :class:`ScriptModule` as well.

    .. DANGER::
        All modules, no matter their device, are always loaded onto the CPU
        during loading.  This is different from :func:`torch.load`'s semantics
        and may change in the future.

    Arguments:
        m: A :class:`ScriptModule` to save.
        f: A file-like object (has to implement write and flush) or a string
           containing a file name.
        _extra_files: Map from filename to contents which will be stored as part of `f`.

    .. note::
        torch.jit.save attempts to preserve the behavior of some operators
        across versions. For example, dividing two integer tensors in
        PyTorch 1.5 performed floor division, and if the module
        containing that code is saved in PyTorch 1.5 and loaded in PyTorch 1.6
        its division behavior will be preserved. The same module saved in
        PyTorch 1.6 will fail to load in PyTorch 1.5, however, since the
        behavior of division changed in 1.6, and 1.5 does not know how to
        replicate the 1.6 behavior.

    Example:

    .. testcode::

        import torch
        import io

        class MyModule(torch.nn.Module):
            def forward(self, x):
                return x + 10

        m = torch.jit.script(MyModule())

        # Save to file
        torch.jit.save(m, 'scriptmodule.pt')
        # This line is equivalent to the previous
        m.save("scriptmodule.pt")

        # Save to io.BytesIO buffer
        buffer = io.BytesIO()
        torch.jit.save(m, buffer)

        # Save with extra files
        extra_files = {'foo.txt': b'bar'}
        torch.jit.save(m, 'scriptmodule.pt', _extra_files=extra_files)
    """
    if _extra_files is None:
        _extra_files = {}
    if isinstance(f, str) or isinstance(f, pathlib.Path):
        m.save(f, _extra_files=_extra_files)
    else:
        ret = m.save_to_buffer(_extra_files=_extra_files)
        f.write(ret)


def load(f, map_location=None, _extra_files=None):
    r"""
    Load a :class:`ScriptModule` or :class:`ScriptFunction` previously
    saved with :func:`torch.jit.save <torch.jit.save>`

    All previously saved modules, no matter their device, are first loaded onto CPU,
    and then are moved to the devices they were saved from. If this fails (e.g.
    because the run time system doesn't have certain devices), an exception is
    raised.

    Arguments:
        f: a file-like object (has to implement read, readline, tell, and seek),
            or a string containing a file name
        map_location (string or torch.device): A simplified version of
            ``map_location`` in `torch.jit.save` used to dynamically remap
            storages to an alternative set of devices.
        _extra_files (dictionary of filename to content): The extra
            filenames given in the map would be loaded and their content
            would be stored in the provided map.

    Returns:
        A :class:`ScriptModule` object.

    Example:

    .. testcode::

        import torch
        import io

        torch.jit.load('scriptmodule.pt')

        # Load ScriptModule from io.BytesIO object
        with open('scriptmodule.pt', 'rb') as f:
            buffer = io.BytesIO(f.read())

        # Load all tensors to the original device
        torch.jit.load(buffer)

        # Load all tensors onto CPU, using a device
        buffer.seek(0)
        torch.jit.load(buffer, map_location=torch.device('cpu'))

        # Load all tensors onto CPU, using a string
        buffer.seek(0)
        torch.jit.load(buffer, map_location='cpu')

        # Load with extra files.
        extra_files = {'foo.txt': ''}  # values will be replaced with data
        torch.jit.load('scriptmodule.pt', _extra_files=extra_files)
        print(extra_files['foo.txt'])

    .. testoutput::
        :hide:

        ...

    .. testcleanup::

        import os
        os.remove("scriptmodule.pt")
    """
    if isinstance(f, string_classes):
        if not os.path.exists(f):  # type: ignore
            raise ValueError("The provided filename {} does not exist".format(f))  # type: ignore
        if os.path.isdir(f):
            raise ValueError("The provided filename {} is a directory".format(f))  # type: ignore

    map_location = validate_map_location(map_location)
    if _extra_files is None:
        _extra_files = {}

    cu = torch._C.CompilationUnit()
    if isinstance(f, str) or isinstance(f, pathlib.Path):
        cpp_module = torch._C.import_ir_module(cu, str(f), map_location, _extra_files)
    else:
        cpp_module = torch._C.import_ir_module_from_buffer(
            cu, f.read(), map_location, _extra_files
        )

    # TODO: Pretty sure this approach loses ConstSequential status and such
    return wrap_cpp_module(cpp_module)


def validate_map_location(map_location=None):
    if isinstance(map_location, str):
        map_location = torch.device(map_location)
    elif not (map_location is None or isinstance(map_location, torch.device)):
        raise ValueError(
            "map_location should be either None, string or torch.device, "
            "but got type: " + str(type(map_location))
        )

    if str(map_location).startswith("cuda"):
        validate_cuda_device(map_location)

    return map_location
