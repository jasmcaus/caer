# Pulled from Scikit-Learn's official Github Repo (18 Sep 2020) to speed up 'caer' package import speeds (since this was the only method referenced from sklearn)

from itertools import chain, compress
from math import ceil, floor
import numpy as np
from ._sklearn_utils import _num_samples, issparse

try:
    from pkg_resources import parse_version  # type: ignore
except ImportError:
    # setuptools not installed
    from distutils.version import LooseVersion
    parse_version = LooseVersion  # type: ignore

np_version = parse_version(np.__version__)

def train_test_split(*arrays,
                     test_size=None,
                     train_size=None
                     ):
    """Split arrays or matrices into random train and test subsets
    
    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
        Allowed inputs are lists, numpy arrays, scipy-sparse
        matrices or pandas dataframes.
    test_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.25.
    train_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.
    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs.
        .. versionadded:: 1.6.6
    """
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError('At least one array required as input')

    arrays = indexable(*arrays)

    n_samples = _num_samples(arrays[0])
    n_train, n_test = _validate_shuffle_split(n_samples, test_size, train_size,
                                              default_test_size=0.25)

    train = np.arange(n_train)
    test = np.arange(n_train, n_train + n_test)

    return list(chain.from_iterable((_safe_indexing(a, train),
                                     _safe_indexing(a, test)) for a in arrays))


def _make_indexable(iterable):
    """Ensure iterable supports indexing or convert to an indexable variant.
    Convert sparse matrices to csr and other non-indexable iterable to arrays.
    Let `None` and indexable objects (e.g. pandas dataframes) pass unchanged.
    Parameters
    ----------
    iterable : {list, dataframe, ndarray, sparse matrix} or None
        Object to be converted to an indexable iterable.
    """
    if issparse(iterable):
        return iterable.tocsr()
    elif hasattr(iterable, "__getitem__") or hasattr(iterable, "iloc"):
        return iterable
    elif iterable is None:
        return iterable
    return np.array(iterable)

def check_consistent_length(*arrays):
    """Check that all arrays have consistent first dimensions.
    Checks whether all objects in arrays have the same shape or length.
    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """

    lengths = [_num_samples(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError("Found input variables with inconsistent numbers of"
                         " samples: %r" % [int(l) for l in lengths])

def indexable(*iterables):
    """Make arrays indexable for cross-validation.
    Checks consistent length, passes through None, and ensures that everything
    can be indexed by converting sparse matrices to csr and converting
    non-interable objects to arrays.
    Parameters
    ----------
    *iterables : {lists, dataframes, ndarrays, sparse matrices}
        List of objects to ensure sliceability.
    """
    result = [_make_indexable(X) for X in iterables]
    check_consistent_length(*result)
    return result

def _determine_key_type(key, accept_slice=True):
    """Determine the data type of key.
    Parameters
    ----------
    key : scalar, slice or array-like
        The key from which we want to infer the data type.
    accept_slice : bool, default=True
        Whether or not to raise an error if the key is a slice.
    Returns
    -------
    dtype : {'int', 'str', 'bool', None}
        Returns the data type of key.
    """
    err_msg = ("No valid specification of the columns. Only a scalar, list or "
               "slice of all integers or all strings, or boolean mask is "
               "allowed")

    dtype_to_str = {int: 'int', str: 'str', bool: 'bool', np.bool_: 'bool'}
    array_dtype_to_str = {'i': 'int', 'u': 'int', 'b': 'bool', 'O': 'str',
                          'U': 'str', 'S': 'str'}

    if key is None:
        return None
    if isinstance(key, tuple(dtype_to_str.keys())):
        try:
            return dtype_to_str[type(key)]
        except KeyError:
            raise ValueError(err_msg)
    if isinstance(key, slice):
        if not accept_slice:
            raise TypeError(
                'Only array-like or scalar are supported. '
                'A Python slice was given.'
            )
        if key.start is None and key.stop is None:
            return None
        key_start_type = _determine_key_type(key.start)
        key_stop_type = _determine_key_type(key.stop)
        if key_start_type is not None and key_stop_type is not None:
            if key_start_type != key_stop_type:
                raise ValueError(err_msg)
        if key_start_type is not None:
            return key_start_type
        return key_stop_type
    if isinstance(key, (list, tuple)):
        unique_key = set(key)
        key_type = {_determine_key_type(elt) for elt in unique_key}
        if not key_type:
            return None
        if len(key_type) != 1:
            raise ValueError(err_msg)
        return key_type.pop()
    if hasattr(key, 'dtype'):
        try:
            return array_dtype_to_str[key.dtype.kind]
        except KeyError:
            raise ValueError(err_msg)
    raise ValueError(err_msg)

def _array_indexing(array, key, key_dtype, axis):
    """Index an array or scipy.sparse consistently across NumPy version."""
    if np_version < parse_version('1.12') or issparse(array):
        # Remove the check for NumPy when using >= 1.12
        # check if we have an boolean array-likes to make the proper indexing
        if key_dtype == 'bool':
            key = np.asarray(key)
    if isinstance(key, tuple):
        key = list(key)
    return array[key] if axis == 0 else array[:, key]


def _pandas_indexing(X, key, key_dtype, axis):
    """Index a pandas dataframe or a series."""
    if hasattr(key, 'shape'):
        # Work-around for indexing with read-only key in pandas
        # solved in pandas 0.25
        key = np.asarray(key)
        key = key if key.flags.writeable else key.copy()
    elif isinstance(key, tuple):
        key = list(key)
    # check whether we should index with loc or iloc
    indexer = X.iloc if key_dtype == 'int' else X.loc
    return indexer[:, key] if axis else indexer[key]


def _list_indexing(X, key, key_dtype):
    """Index a Python list."""
    if np.isscalar(key) or isinstance(key, slice):
        # key is a slice or a scalar
        return X[key]
    if key_dtype == 'bool':
        # key is a boolean array-like
        return list(compress(X, key))
    # key is a integer array-like of key
    return [X[idx] for idx in key]


def _safe_indexing(X, indices, *, axis=0):
    """Return rows, items or columns of X using indices.
    .. warning::
        This utility is documented, but **private**. This means that
        backward compatibility might be broken without any deprecation
        cycle.
    Parameters
    ----------
    X : array-like, sparse-matrix, list, pandas.DataFrame, pandas.Series
        Data from which to sample rows, items or columns. `list` are only
        supported when `axis=0`.
    indices : bool, int, str, slice, array-like
        - If `axis=0`, boolean and integer array-like, integer slice,
          and scalar integer are supported.
        - If `axis=1`:
            - to select a single column, `indices` can be of `int` type for
              all `X` types and `str` only for dataframe. The selected subset
              will be 1D, unless `X` is a sparse matrix in which case it will
              be 2D.
            - to select multiples columns, `indices` can be one of the
              following: `list`, `array`, `slice`. The type used in
              these containers can be one of the following: `int`, 'bool' and
              `str`. However, `str` is only supported when `X` is a dataframe.
              The selected subset will be 2D.
    axis : int, default=0
        The axis along which `X` will be subsampled. `axis=0` will select
        rows while `axis=1` will select columns.
    Returns
    -------
    subset
        Subset of X on axis 0 or 1.
    Notes
    -----
    CSR, CSC, and LIL sparse matrices are supported. COO sparse matrices are
    not supported.
    """
    if indices is None:
        return X

    if axis not in (0, 1):
        raise ValueError(
            "'axis' should be either 0 (to index rows) or 1 (to index "
            " column). Got {} instead.".format(axis)
        )

    indices_dtype = _determine_key_type(indices)

    if axis == 0 and indices_dtype == 'str':
        raise ValueError(
            "String indexing is not supported with 'axis=0'"
        )

    if axis == 1 and X.ndim != 2:
        raise ValueError(
            "'X' should be a 2D NumPy array, 2D sparse matrix or pandas "
            "dataframe when indexing the columns (i.e. 'axis=1'). "
            "Got {} instead with {} dimension(s).".format(type(X), X.ndim)
        )

    if axis == 1 and indices_dtype == 'str' and not hasattr(X, 'loc'):
        raise ValueError(
            "Specifying the columns using strings is only supported for "
            "pandas DataFrames"
        )

    if hasattr(X, "iloc"):
        return _pandas_indexing(X, indices, indices_dtype, axis=axis)
    elif hasattr(X, "shape"):
        return _array_indexing(X, indices, indices_dtype, axis=axis)
    else:
        return _list_indexing(X, indices, indices_dtype)

def _validate_shuffle_split(n_samples, test_size, train_size,
                            default_test_size=None):
    """
    Validation helper to check if the test/test sizes are meaningful wrt to the
    size of the data (n_samples)
    """
    if test_size is None and train_size is None:
        test_size = default_test_size

    test_size_type = np.asarray(test_size).dtype.kind
    train_size_type = np.asarray(train_size).dtype.kind

    if (test_size_type == 'i' and (test_size >= n_samples or test_size <= 0)
       or test_size_type == 'f' and (test_size <= 0 or test_size >= 1)):
        raise ValueError('test_size={0} should be either positive and smaller'
                         ' than the number of samples {1} or a float in the '
                         '(0, 1) range'.format(test_size, n_samples))

    if (train_size_type == 'i' and (train_size >= n_samples or train_size <= 0)
       or train_size_type == 'f' and (train_size <= 0 or train_size >= 1)):
        raise ValueError('train_size={0} should be either positive and smaller'
                         ' than the number of samples {1} or a float in the '
                         '(0, 1) range'.format(train_size, n_samples))

    if train_size is not None and train_size_type not in ('i', 'f'):
        raise ValueError("Invalid value for train_size: {}".format(train_size))
    if test_size is not None and test_size_type not in ('i', 'f'):
        raise ValueError("Invalid value for test_size: {}".format(test_size))

    if (train_size_type == 'f' and test_size_type == 'f' and
            train_size + test_size > 1):
        raise ValueError(
            'The sum of test_size and train_size = {}, should be in the (0, 1)'
            ' range. Reduce test_size and/or train_size.'
            .format(train_size + test_size))

    if test_size_type == 'f':
        n_test = ceil(test_size * n_samples)
    elif test_size_type == 'i':
        n_test = float(test_size)

    if train_size_type == 'f':
        n_train = floor(train_size * n_samples)
    elif train_size_type == 'i':
        n_train = float(train_size)

    if train_size is None:
        n_train = n_samples - n_test
    elif test_size is None:
        n_test = n_samples - n_train

    if n_train + n_test > n_samples:
        raise ValueError('The sum of train_size and test_size = %d, '
                         'should be smaller than the number of '
                         'samples %d. Reduce test_size and/or '
                         'train_size.' % (n_train + n_test, n_samples))

    n_train, n_test = int(n_train), int(n_test)

    if n_train == 0:
        raise ValueError(
            'With n_samples={}, test_size={} and train_size={}, the '
            'resulting train set will be empty. Adjust any of the '
            'aforementioned parameters.'.format(n_samples, test_size,
                                                train_size)
        )

    return n_train, n_test