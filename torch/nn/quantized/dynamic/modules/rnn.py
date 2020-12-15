import numbers
import warnings

import torch
import torch.nn as nn
from torch import Tensor  # noqa: F401
from torch._jit_internal import Tuple, Optional, List, Union, Dict  # noqa: F401
from torch.nn.utils.rnn import PackedSequence
from torch.nn.quantized.modules.utils import _quantize_weight

def apply_permutation(tensor: Tensor, permutation: Tensor, dim: int = 1) -> Tensor:
    return tensor.index_select(dim, permutation)

class PackedParameter(torch.nn.Module):
    def __init__(self, param):
        super(PackedParameter, self).__init__()
        self.param = param

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super(PackedParameter, self)._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'param'] = self.param

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self.param = state_dict[prefix + 'param']
        super(PackedParameter, self)._load_from_state_dict(state_dict, prefix, local_metadata, False,
                                                           missing_keys, unexpected_keys, error_msgs)

class RNNBase(torch.nn.Module):

    _FLOAT_MODULE = nn.RNNBase

    _version = 2

    def __init__(self, mode, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0., bidirectional=False, dtype=torch.qint8):
        super(RNNBase, self).__init__()

        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        self.dtype = dtype
        self.version = 2
        self.training = False
        num_directions = 2 if bidirectional else 1

        # "type: ignore" is required since ints and Numbers are not fully comparable
        # https://github.com/python/mypy/issues/8566
        if not isinstance(dropout, numbers.Number) \
                or not 0 <= dropout <= 1 or isinstance(dropout, bool):  # type: ignore
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")
        if dropout > 0 and num_layers == 1:  # type: ignore
            warnings.warn("dropout option adds dropout after all but last "
                          "recurrent layer, so non-zero dropout expects "
                          "num_layers greater than 1, but got dropout={} and "
                          "num_layers={}".format(dropout, num_layers))

        if mode == 'LSTM':
            gate_size = 4 * hidden_size
        else:
            raise ValueError("Unrecognized RNN mode: " + mode)

        _all_weight_values = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions

                w_ih = torch.randn(gate_size, layer_input_size).to(torch.float)
                w_hh = torch.randn(gate_size, hidden_size).to(torch.float)
                b_ih = torch.randn(gate_size).to(torch.float)
                b_hh = torch.randn(gate_size).to(torch.float)
                if dtype == torch.qint8:
                    w_ih = torch.quantize_per_tensor(w_ih, scale=0.1, zero_point=0, dtype=torch.qint8)
                    w_hh = torch.quantize_per_tensor(w_hh, scale=0.1, zero_point=0, dtype=torch.qint8)
                    packed_ih = \
                        torch.ops.quantized.linear_prepack(w_ih, b_ih)
                    packed_hh = \
                        torch.ops.quantized.linear_prepack(w_hh, b_hh)
                    if self.version is None or self.version < 2:
                        cell_params = torch.ops.quantized.make_quantized_cell_params_dynamic(
                            packed_ih, packed_hh, b_ih, b_hh)
                    else:
                        cell_params = torch.ops.quantized.make_quantized_cell_params_dynamic(
                            packed_ih, packed_hh, b_ih, b_hh, True)

                else:

                    packed_ih = torch.ops.quantized.linear_prepack_fp16(w_ih, b_ih)
                    packed_hh = torch.ops.quantized.linear_prepack_fp16(w_hh, b_hh)
                    cell_params = torch.ops.quantized.make_quantized_cell_params_fp16(
                        packed_ih, packed_hh)

                _all_weight_values.append(PackedParameter(cell_params))
        self._all_weight_values = torch.nn.ModuleList(_all_weight_values)

    def _get_name(self):
        return 'DynamicQuantizedRNN'

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)

    def __repr__(self):
        # We don't want to show `ModuleList` children, hence custom
        # `__repr__`. This is the same as nn.Module.__repr__, except the check
        # for the `PackedParameter` and `nn.ModuleList`.
        # You should still override `extra_repr` to add more info.
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            if isinstance(module, (PackedParameter, nn.ModuleList)):
                continue
            mod_str = repr(module)
            mod_str = nn.modules.module._addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str

    def check_input(self, input: Tensor, batch_sizes: Optional[Tensor]) -> None:
        expected_input_dim = 2 if batch_sizes is not None else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))

    def get_expected_hidden_size(self, input: Tensor, batch_sizes: Optional[Tensor]) -> Tuple[int, int, int]:
        if batch_sizes is not None:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)
        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.hidden_size)
        return expected_hidden_size

    def check_hidden_size(
        self, hx: Tensor, expected_hidden_size: Tuple[int, int, int],
        msg: str = 'Expected hidden size {}, got {}'
    ) -> None:
        if hx.size() != expected_hidden_size:
            raise RuntimeError(msg.format(
                expected_hidden_size, list(hx.size())))

    def check_forward_args(self, input: Tensor, hidden: Tensor, batch_sizes: Optional[Tensor]) -> None:
        self.check_input(input, batch_sizes)
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)
        self.check_hidden_size(hidden, expected_hidden_size,
                               msg='Expected hidden size {}, got {}')

    def permute_hidden(self, hx: Tensor, permutation: Optional[Tensor]) -> Tensor:
        if permutation is None:
            return hx
        return apply_permutation(hx, permutation)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        self.version = version
        super(RNNBase, self)._load_from_state_dict(state_dict, prefix, local_metadata, False,
                                                   missing_keys, unexpected_keys, error_msgs)

    @classmethod
    def from_float(cls, mod):
        assert type(mod) in set(
            [torch.nn.LSTM,
             torch.nn.GRU]
        ), 'nn.quantized.dynamic.RNNBase.from_float only works for nn.LSTM and nn.GRU'
        assert hasattr(
            mod,
            'qconfig'
        ), 'Input float module must have qconfig defined'

        if mod.qconfig is not None and mod.qconfig.weight is not None:
            weight_observer_method = mod.qconfig.weight
        else:
            # We have the circular import issues if we import the qconfig in the beginning of this file:
            # https://github.com/pytorch/pytorch/pull/24231. The current workaround is to postpone the
            # import until we need it.
            from torch.quantization.qconfig import default_dynamic_qconfig
            weight_observer_method = default_dynamic_qconfig.weight

        dtype = weight_observer_method().dtype
        supported_scalar_types = [torch.qint8, torch.float16]
        if dtype not in supported_scalar_types:
            raise RuntimeError('Unsupported dtype for dynamic RNN quantization: {}'.format(dtype))

        if mod.mode == 'LSTM':
            qRNNBase = LSTM(mod.input_size, mod.hidden_size, mod.num_layers,
                            mod.bias, mod.batch_first, mod.dropout, mod.bidirectional, dtype)
        else:
            raise NotImplementedError('Only LSTM is supported for QuantizedRNN for now')

        num_directions = 2 if mod.bidirectional else 1

        assert mod.bias

        _all_weight_values = []
        for layer in range(qRNNBase.num_layers):
            for direction in range(num_directions):
                layer_input_size = qRNNBase.input_size if layer == 0 else qRNNBase.hidden_size * num_directions

                suffix = '_reverse' if direction == 1 else ''

                def retrieve_weight_bias(ihhh):
                    weight_name = 'weight_{}_l{}{}'.format(ihhh, layer, suffix)
                    bias_name = 'bias_{}_l{}{}'.format(ihhh, layer, suffix)
                    weight = getattr(mod, weight_name)
                    bias = getattr(mod, bias_name)
                    return weight, bias

                weight_ih, bias_ih = retrieve_weight_bias('ih')
                weight_hh, bias_hh = retrieve_weight_bias('hh')

                if dtype == torch.qint8:
                    def quantize_and_pack(w, b):
                        weight_observer = weight_observer_method()
                        weight_observer(w)
                        qweight = _quantize_weight(w.float(), weight_observer)
                        packed_weight = \
                            torch.ops.quantized.linear_prepack(qweight, b)
                        return packed_weight
                    packed_ih = quantize_and_pack(weight_ih, bias_ih)
                    packed_hh = quantize_and_pack(weight_hh, bias_hh)
                    if qRNNBase.version is None or qRNNBase.version < 2:
                        cell_params = torch.ops.quantized.make_quantized_cell_params_dynamic(
                            packed_ih, packed_hh, bias_ih, bias_hh)
                    else:
                        cell_params = torch.ops.quantized.make_quantized_cell_params_dynamic(
                            packed_ih, packed_hh, bias_ih, bias_hh, True)

                elif dtype == torch.float16:
                    packed_ih = torch.ops.quantized.linear_prepack_fp16(
                        weight_ih.float(), bias_ih)
                    packed_hh = torch.ops.quantized.linear_prepack_fp16(
                        weight_hh.float(), bias_hh)

                    cell_params = torch.ops.quantized.make_quantized_cell_params_fp16(
                        packed_ih, packed_hh)
                else:
                    raise RuntimeError('Unsupported dtype specified for dynamic quantized LSTM!')

                _all_weight_values.append(PackedParameter(cell_params))
        qRNNBase._all_weight_values = torch.nn.ModuleList(_all_weight_values)

        return qRNNBase


    def _weight_bias(self):
        # Returns a dict of weights and biases
        weight_bias_dict: Dict[str, Dict] = {'weight' : {}, 'bias' : {}}
        count = 0
        num_directions = 2 if self.bidirectional else 1
        for layer in range(self.num_layers):
            for direction in range(num_directions):
                suffix = '_reverse' if direction == 1 else ''
                key_name1 = 'weight_ih_l{layer_idx}{suffix}'.format(layer_idx=layer, suffix=suffix)
                key_name2 = 'weight_hh_l{layer_idx}{suffix}'.format(layer_idx=layer, suffix=suffix)
                # packed weights are part of torchbind class, CellParamsSerializationType
                # Within the packed weight class, the weight and bias are accessible as Tensors
                packed_weight_bias = self._all_weight_values[count].param.__getstate__()[0][4]
                weight_bias_dict['weight'][key_name1] = packed_weight_bias[0].__getstate__()[0][0]
                weight_bias_dict['weight'][key_name2] = packed_weight_bias[1].__getstate__()[0][0]
                key_name1 = 'bias_ih_l{layer_idx}{suffix}'.format(layer_idx=layer, suffix=suffix)
                key_name2 = 'bias_hh_l{layer_idx}{suffix}'.format(layer_idx=layer, suffix=suffix)
                weight_bias_dict['bias'][key_name1] = packed_weight_bias[0].__getstate__()[0][1]
                weight_bias_dict['bias'][key_name2] = packed_weight_bias[1].__getstate__()[0][1]
                count = count + 1
        return weight_bias_dict

    def get_weight(self):
        return self._weight_bias()['weight']

    def get_bias(self):
        return self._weight_bias()['bias']

class LSTM(RNNBase):
    r"""
    A dynamic quantized LSTM module with floating point tensor as inputs and outputs.
    We adopt the same interface as `torch.nn.LSTM`, please see
    https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM for documentation.

    Examples::

        >>> rnn = nn.LSTM(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> c0 = torch.randn(2, 3, 20)
        >>> output, (hn, cn) = rnn(input, (h0, c0))
    """
    _FLOAT_MODULE = nn.LSTM

    __overloads__ = {'forward': ['forward_packed', 'forward_tensor']}

    def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__('LSTM', *args, **kwargs)

    def _get_name(self):
        return 'DynamicQuantizedLSTM'

    def forward_impl(
        self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]],
        batch_sizes: Optional[Tensor], max_batch_size: int,
        sorted_indices: Optional[Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            zeros = torch.zeros(self.num_layers * num_directions,
                                max_batch_size, self.hidden_size,
                                dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, batch_sizes)

        _all_params = ([m.param for m in self._all_weight_values])
        if batch_sizes is None:
            result = torch.quantized_lstm(input, hx, _all_params, self.bias, self.num_layers,
                                          float(self.dropout), self.training, self.bidirectional,
                                          self.batch_first, dtype=self.dtype, use_dynamic=True)
        else:
            result = torch.quantized_lstm(input, batch_sizes, hx, _all_params, self.bias,
                                          self.num_layers, float(self.dropout), self.training,
                                          self.bidirectional, dtype=self.dtype, use_dynamic=True)
        output = result[0]
        hidden = result[1:]

        return output, hidden

    @torch.jit.export
    def forward_tensor(
        self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        batch_sizes = None
        max_batch_size = input.size(0) if self.batch_first else input.size(1)
        sorted_indices = None
        unsorted_indices = None

        output, hidden = self.forward_impl(
            input, hx, batch_sizes, max_batch_size, sorted_indices)

        return output, self.permute_hidden(hidden, unsorted_indices)

    @torch.jit.export
    def forward_packed(
        self, input: PackedSequence, hx: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[PackedSequence, Tuple[Tensor, Tensor]]:  # noqa
        input_, batch_sizes, sorted_indices, unsorted_indices = input
        max_batch_size = batch_sizes[0]
        max_batch_size = int(max_batch_size)

        output_, hidden = self.forward_impl(
            input_, hx, batch_sizes, max_batch_size, sorted_indices)

        output = PackedSequence(output_, batch_sizes,
                                sorted_indices, unsorted_indices)
        return output, self.permute_hidden(hidden, unsorted_indices)

    # "type: ignore" is required due to issue #43072
    def permute_hidden(  # type: ignore
        self, hx: Tuple[Tensor, Tensor], permutation: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        if permutation is None:
            return hx
        return apply_permutation(hx[0], permutation), apply_permutation(hx[1], permutation)

    # "type: ignore" is required due to issue #43072
    def check_forward_args(  # type: ignore
        self, input: Tensor, hidden: Tuple[Tensor, Tensor], batch_sizes: Optional[Tensor]
    ) -> None:
        self.check_input(input, batch_sizes)
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)

        self.check_hidden_size(hidden[0], expected_hidden_size,
                               'Expected hidden[0] size {}, got {}')
        self.check_hidden_size(hidden[1], expected_hidden_size,
                               'Expected hidden[1] size {}, got {}')

    @torch.jit.ignore
    def forward(self, input, hx=None):
        if isinstance(input, PackedSequence):
            return self.forward_packed(input, hx)
        else:
            return self.forward_tensor(input, hx)

    @classmethod
    def from_float(cls, mod):
        return super(LSTM, cls).from_float(mod)



class RNNCellBase(torch.nn.Module):
    # _FLOAT_MODULE = nn.CellRNNBase
    __constants__ = ['input_size', 'hidden_size', 'bias']

    def __init__(self, input_size, hidden_size, bias=True, num_chunks=4, dtype=torch.qint8):
        super(RNNCellBase, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        if bias:
            self.bias_ih = torch.randn(num_chunks * hidden_size).to(dtype=torch.float)
            self.bias_hh = torch.randn(num_chunks * hidden_size).to(dtype=torch.float)
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        weight_ih = torch.randn(num_chunks * hidden_size, input_size).to(torch.float)
        weight_hh = torch.randn(num_chunks * hidden_size, hidden_size).to(torch.float)
        if dtype == torch.qint8:
            weight_ih = torch.quantize_per_tensor(weight_ih, scale=1, zero_point=0, dtype=torch.qint8)
            weight_hh = torch.quantize_per_tensor(weight_hh, scale=1, zero_point=0, dtype=torch.qint8)

        if dtype == torch.qint8:
            # for each layer, for each direction we need to quantize and pack
            # weights and pack parameters in this order:
            #
            #   w_ih, w_hh
            packed_weight_ih = \
                torch.ops.quantized.linear_prepack(weight_ih, self.bias_ih)
            packed_weight_hh = \
                torch.ops.quantized.linear_prepack(weight_hh, self.bias_hh)
        else:
            # for each layer, for each direction we need to quantize and pack
            # weights and pack parameters in this order:
            #
            #   packed_ih, packed_hh, b_ih, b_hh
            packed_weight_ih = torch.ops.quantized.linear_prepack_fp16(
                weight_ih, self.bias_ih)
            packed_weight_hh = torch.ops.quantized.linear_prepack_fp16(
                weight_hh, self.bias_hh)

        self._packed_weight_ih = packed_weight_ih
        self._packed_weight_hh = packed_weight_hh

    def _get_name(self):
        return 'DynamicQuantizedRNNBase'

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    def check_forward_input(self, input):
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def check_forward_hidden(self, input: Tensor, hx: Tensor, hidden_label: str = '') -> None:
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size))

    @classmethod
    def from_float(cls, mod):
        assert type(mod) in set([torch.nn.LSTMCell,
                                 torch.nn.GRUCell,
                                 torch.nn.RNNCell]), 'nn.quantized.dynamic.RNNCellBase.from_float \
                                 only works for nn.LSTMCell, nn.GRUCell and nn.RNNCell'
        assert hasattr(
            mod, 'qconfig'), 'Input float module must have qconfig defined'

        if mod.qconfig is not None and mod.qconfig.weight is not None:
            weight_observer_method = mod.qconfig.weight
        else:
            # We have the circular import issues if we import the qconfig in the beginning of this file:
            # https://github.com/pytorch/pytorch/pull/24231. The current workaround is to postpone the
            # import until we need it.
            from torch.quantization.qconfig import default_dynamic_qconfig
            weight_observer_method = default_dynamic_qconfig.weight

        dtype = weight_observer_method().dtype
        supported_scalar_types = [torch.qint8, torch.float16]
        if dtype not in supported_scalar_types:
            raise RuntimeError('Unsupported dtype for dynamic RNN quantization: {}'.format(dtype))

        qRNNCellBase: Union[LSTMCell, GRUCell, RNNCell]

        if type(mod) == torch.nn.LSTMCell:
            qRNNCellBase = LSTMCell(mod.input_size, mod.hidden_size, bias=mod.bias, dtype=dtype)
        elif type(mod) == torch.nn.GRUCell:
            qRNNCellBase = GRUCell(mod.input_size, mod.hidden_size, bias=mod.bias, dtype=dtype)
        elif type(mod) == torch.nn.RNNCell:
            qRNNCellBase = RNNCell(mod.input_size, mod.hidden_size, bias=mod.bias, nonlinearity=mod.nonlinearity, dtype=dtype)
        else:
            raise NotImplementedError('Only LSTMCell, GRUCell and RNNCell \
            are supported for QuantizedRNN for now')


        assert mod.bias

        def process_weights(weight, bias, dtype):

            if dtype == torch.qint8:
                # for each layer, for each direction we need to quantize and pack
                # weights and pack parameters in this order:
                #
                #   w_ih, w_hh
                weight_observer = weight_observer_method()
                weight_observer(weight)
                qweight = _quantize_weight(weight.float(), weight_observer)
                packed_weight = \
                    torch.ops.quantized.linear_prepack(qweight, bias)

                return packed_weight
            else:
                # for each layer, for each direction we need to quantize and pack
                # weights and pack parameters in this order:
                #
                #   packed_ih, packed_hh, b_ih, b_hh
                packed_weight = torch.ops.quantized.linear_prepack_fp16(
                    weight.float(), bias)

                return packed_weight

        qRNNCellBase._packed_weight_ih = process_weights(mod.weight_ih, mod.bias_ih, dtype)
        qRNNCellBase._packed_weight_hh = process_weights(mod.weight_hh, mod.bias_hh, dtype)
        return qRNNCellBase

    def _weight_bias(self):
        # Returns a dict of weights and biases
        weight_bias_dict: Dict[str, Dict] = {'weight' : {}, 'bias' : {}}
        w1, b1 = self._packed_weight_ih.__getstate__()[0]
        w2, b2 = self._packed_weight_hh.__getstate__()[0]
        weight_bias_dict['weight']['weight_ih'] = w1
        weight_bias_dict['weight']['weight_hh'] = w2
        weight_bias_dict['bias']['bias_ih'] = b1
        weight_bias_dict['bias']['bias_hh'] = b2
        return weight_bias_dict

    def get_weight(self):
        return self._weight_bias()['weight']

    def get_bias(self):
        return self._weight_bias()['bias']

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super(RNNCellBase, self)._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + '_packed_weight_ih'] = self._packed_weight_ih
        destination[prefix + '_packed_weight_hh'] = self._packed_weight_hh


    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self._packed_weight_ih = state_dict.pop(prefix + '_packed_weight_ih')
        self._packed_weight_hh = state_dict.pop(prefix + '_packed_weight_hh')
        super(RNNCellBase, self)._load_from_state_dict(state_dict, prefix, local_metadata, False,
                                                       missing_keys, unexpected_keys, error_msgs)

class RNNCell(RNNCellBase):
    r"""An Elman RNN cell with tanh or ReLU non-linearity.
    A dynamic quantized RNNCell module with floating point tensor as inputs and outputs.
    Weights are quantized to 8 bits. We adopt the same interface as `torch.nn.RNNCell`,
    please see https://pytorch.org/docs/stable/nn.html#torch.nn.RNNCell for documentation.

    Examples::

        >>> rnn = nn.RNNCell(10, 20)
        >>> input = torch.randn(6, 3, 10)
        >>> hx = torch.randn(3, 20)
        >>> output = []
        >>> for i in range(6):
                hx = rnn(input[i], hx)
                output.append(hx)
    """
    __constants__ = ['input_size', 'hidden_size', 'bias', 'nonlinearity']

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh", dtype=torch.qint8):
        super(RNNCell, self).__init__(input_size, hidden_size, bias, num_chunks=1, dtype=dtype)
        self.nonlinearity = nonlinearity

    def _get_name(self):
        return 'DynamicQuantizedRNNCell'

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        self.check_forward_input(input)
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        self.check_forward_hidden(input, hx, '')
        if self.nonlinearity == "tanh":
            ret = torch.ops.quantized.quantized_rnn_tanh_cell_dynamic(
                input, hx,
                self._packed_weight_ih, self._packed_weight_hh,
                self.bias_ih, self.bias_hh)
        elif self.nonlinearity == "relu":
            ret = torch.ops.quantized.quantized_rnn_relu_cell_dynamic(
                input, hx,
                self._packed_weight_ih, self._packed_weight_hh,
                self.bias_ih, self.bias_hh)
        else:
            ret = input  # TODO: remove when jit supports exception flow
            raise RuntimeError(
                "Unknown nonlinearity: {}".format(self.nonlinearity))
        return ret

    @classmethod
    def from_float(cls, mod):
        return super(RNNCell, cls).from_float(mod)


class LSTMCell(RNNCellBase):
    r"""A long short-term memory (LSTM) cell.

    A dynamic quantized LSTMCell module with floating point tensor as inputs and outputs.
    Weights are quantized to 8 bits. We adopt the same interface as `torch.nn.LSTMCell`,
    please see https://pytorch.org/docs/stable/nn.html#torch.nn.LSTMCell for documentation.

    Examples::

        >>> rnn = nn.LSTMCell(10, 20)
        >>> input = torch.randn(6, 3, 10)
        >>> hx = torch.randn(3, 20)
        >>> cx = torch.randn(3, 20)
        >>> output = []
        >>> for i in range(6):
                hx, cx = rnn(input[i], (hx, cx))
                output.append(hx)
    """

    def __init__(self, *args, **kwargs):
        super(LSTMCell, self).__init__(*args, num_chunks=4, **kwargs)  # type: ignore

    def _get_name(self):
        return 'DynamicQuantizedLSTMCell'

    def forward(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        self.check_forward_input(input)
        if hx is None:
            zeros = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        self.check_forward_hidden(input, hx[0], '[0]')
        self.check_forward_hidden(input, hx[1], '[1]')
        return torch.ops.quantized.quantized_lstm_cell_dynamic(
            input, hx,
            self._packed_weight_ih, self._packed_weight_hh,
            self.bias_ih, self.bias_hh)

    @classmethod
    def from_float(cls, mod):
        return super(LSTMCell, cls).from_float(mod)

class GRUCell(RNNCellBase):
    r"""A gated recurrent unit (GRU) cell

    A dynamic quantized GRUCell module with floating point tensor as inputs and outputs.
    Weights are quantized to 8 bits. We adopt the same interface as `torch.nn.GRUCell`,
    please see https://pytorch.org/docs/stable/nn.html#torch.nn.GRUCell for documentation.

    Examples::

        >>> rnn = nn.GRUCell(10, 20)
        >>> input = torch.randn(6, 3, 10)
        >>> hx = torch.randn(3, 20)
        >>> output = []
        >>> for i in range(6):
                hx = rnn(input[i], hx)
                output.append(hx)
    """

    def __init__(self, input_size, hidden_size, bias=True, dtype=torch.qint8):
        super(GRUCell, self).__init__(input_size, hidden_size, bias, num_chunks=3, dtype=dtype)

    def _get_name(self):
        return 'DynamicQuantizedGRUCell'

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        self.check_forward_input(input)
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        self.check_forward_hidden(input, hx, '')
        return torch.ops.quantized.quantized_gru_cell_dynamic(
            input, hx,
            self._packed_weight_ih, self._packed_weight_hh,
            self.bias_ih, self.bias_hh,
        )

    @classmethod
    def from_float(cls, mod):
        return super(GRUCell, cls).from_float(mod)
