
#pylint: disable=bare-except,protected-access,c-extension-no-member

import numbers
import numpy as np

# Sklearn-specific imports
from ._spmatrix import spmatrix
from inspect import signature, Parameter
from functools import wraps
import warnings


# ------------------------ SKLEARN SPECIFIC METHODS ------------------------- 
# -------------------- REFERENCED FROM ITS OFFICIAL REPO ---------------------

def _deprecate_positional_args(f):
    """Decorator for methods that issues warnings for positional arguments.
    Using the keyword-only argument syntax in pep 3102, arguments after the
    * will issue a warning when passed as a positional argument.
    Parameters
    ----------
    f : callable
        Function to check arguments on.
    """
    sig = signature(f)
    kwonly_args = []
    all_args = []

    for name, param in sig.parameters.items():
        if param.kind == Parameter.POSITIONAL_OR_KEYWORD:
            all_args.append(name)
        elif param.kind == Parameter.KEYWORD_ONLY:
            kwonly_args.append(name)

    @wraps(f)
    def inner_f(*args, **kwargs):
        extra_args = len(args) - len(all_args)
        if extra_args <= 0:
            return f(*args, **kwargs)

        # extra_args > 0
        args_msg = ['{}={}'.format(name, arg)
                    for name, arg in zip(kwonly_args[:extra_args],
                                         args[-extra_args:])]
        warnings.warn("Pass {} as keyword args. From version 0.25 "
                      "passing these as positional arguments will "
                      "result in an error".format(", ".join(args_msg)),
                      FutureWarning)
        kwargs.update(zip(sig.parameters, args))
        return f(**kwargs)
    return inner_f

@_deprecate_positional_args
def check_array(array, *, 
                dtype="numeric", order=None, copy=False, force_all_finite=True,
                ensure_2d=True, allow_nd=False, ensure_min_samples=1,
                ensure_min_features=1, estimator=None):

    """Input validation on an array, list, sparse matrix or similar.
    By default, the input is checked to be a non-empty 2D array containing
    only finite values. If the dtype of the array is object, attempt
    converting to float, raising on failure.
    Parameters
    ----------
    array : object
        Input object to check / convert.
    accept_sparse : str, bool or list/tuple of str, default=False
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.
    accept_large_sparse : bool, default=True
        If a CSR, CSC, COO or BSR sparse matrix is supplied and accepted by
        accept_sparse, accept_large_sparse=False will cause it to be accepted
        only if its indices are stored with a 32-bit dtype.
        .. versionadded:: 0.20
    dtype : 'numeric', type, list of type or None, default='numeric'
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.
    order : {'F', 'C'} or None, default=None
        Whether an array will be forced to be fortran or c-style.
        When order is None (default), then if copy=False, nothing is ensured
        about the memory layout of the output array; otherwise (copy=True)
        the memory layout of the returned array is kept as close as possible
        to the original array.
    copy : bool, default=False
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.
    force_all_finite : bool or 'allow-nan', default=True
        Whether to raise an error on np.inf, np.nan, pd.NA in array. The
        possibilities are:
        - True: Force all values of array to be finite.
        - False: accepts np.inf, np.nan, pd.NA in array.
        - 'allow-nan': accepts only np.nan and pd.NA values in array. Values
          cannot be infinite.
        .. versionadded:: 0.20
           ``force_all_finite`` accepts the string ``'allow-nan'``.
        .. versionchanged:: 0.23
           Accepts `pd.NA` and converts it into `np.nan`
    ensure_2d : bool, default=True
        Whether to raise a value error if array is not 2D.
    allow_nd : bool, default=False
        Whether to allow array.ndim > 2.
    ensure_min_samples : int, default=1
        Make sure that the array has a minimum number of samples in its first
        axis (rows for a 2D array). Setting to 0 disables this check.
    ensure_min_features : int, default=1
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when the input data has effectively 2
        dimensions or is originally 1D and ``ensure_2d`` is True. Setting to 0
        disables this check.
    estimator : str or estimator instance, default=None
        If passed, include the name of the estimator in warning messages.
    Returns
    -------
    array_converted : object
        The converted and validated array.
    """
    # store reference to original array to check if copy is needed when
    # function returns
    array_orig = array

    # store whether originally we wanted numeric dtype
    dtype_numeric = isinstance(dtype, str) and dtype == "numeric"

    dtype_orig = getattr(array, "dtype", None)
    if not hasattr(dtype_orig, 'kind'):
        # not a data type (e.g. a column named dtype in a pandas DataFrame)
        dtype_orig = None

    # check if the object contains several dtypes (typically a pandas
    # DataFrame), and store them. If not, store None.
    # dtypes_orig = None
    # has_pd_integer_array = False
    # if hasattr(array, "dtypes") and hasattr(array.dtypes, '__array__'):
    #     # throw warning if columns are sparse. If all columns are sparse, then
    #     # array.sparse exists and sparsity will be perserved (later).
    #     with suppress(ImportError):
    #         from pandas.api.types import is_sparse
    #         if (not hasattr(array, 'sparse') and
    #                 array.dtypes.apply(is_sparse).any()):
    #             warnings.warn(
    #                 "pandas.DataFrame with sparse columns found."
    #                 "It will be converted to a dense numpy array."
    #             )

    #     dtypes_orig = list(array.dtypes)
    #     # pandas boolean dtype __array__ interface coerces bools to objects
    #     for i, dtype_iter in enumerate(dtypes_orig):
    #         if dtype_iter.kind == 'b':
    #             dtypes_orig[i] = np.dtype(object)
    #         elif dtype_iter.name.startswith(("Int", "UInt")):
    #             # name looks like an Integer Extension Array, now check for
    #             # the dtype
    #             with suppress(ImportError):
    #                 from pandas import (Int8Dtype, Int16Dtype,
    #                                     Int32Dtype, Int64Dtype,
    #                                     UInt8Dtype, UInt16Dtype,
    #                                     UInt32Dtype, UInt64Dtype)
    #                 if isinstance(dtype_iter, (Int8Dtype, Int16Dtype,
    #                                            Int32Dtype, Int64Dtype,
    #                                            UInt8Dtype, UInt16Dtype,
    #                                            UInt32Dtype, UInt64Dtype)):
    #                     has_pd_integer_array = True

    #     if all(isinstance(dtype, np.dtype) for dtype in dtypes_orig):
    #         dtype_orig = np.result_type(*dtypes_orig)

    if dtype_numeric:
        if dtype_orig is not None and dtype_orig.kind == "O":
            # if input is object, convert to float.
            dtype = np.float64
        else:
            dtype = None

    if isinstance(dtype, (list, tuple)):
        if dtype_orig is not None and dtype_orig in dtype:
            # no dtype conversion required
            dtype = None
        else:
            # dtype conversion required. Let's select the first element of the
            # list of accepted types.
            dtype = dtype[0]

    # if has_pd_integer_array:
    #     # If there are any pandas integer extension arrays,
    #     array = array.astype(dtype)

    if force_all_finite not in (True, False, 'allow-nan'):
        raise ValueError('force_all_finite should be a bool or "allow-nan"'
                         '. Got {!r} instead'.format(force_all_finite))

    if estimator is not None:
        if isinstance(estimator, str):
            estimator_name = estimator
        else:
            estimator_name = estimator.__class__.__name__
    else:
        estimator_name = "Estimator"
    context = " by %s" % estimator_name if estimator is not None else ""

    # When all dataframe columns are sparse, convert to a sparse array
    if hasattr(array, 'sparse') and array.ndim > 1:
        # DataFrame.sparse only supports `to_coo`
        array = array.sparse.to_coo()
        if array.dtype == np.dtype('object'):
            unique_dtypes = set(
                [dt.subtype.name for dt in array_orig.dtypes]
            )
            if len(unique_dtypes) > 1:
                raise ValueError(
                    "Pandas DataFrame with mixed sparse extension arrays "
                    "generated a sparse matrix with object dtype which "
                    "can not be converted to a scipy sparse matrix."
                    "Sparse extension arrays should all have the same "
                    "numeric type.")

    # if issparse(array):
    #     _ensure_no_complex_data(array)
    #     array = _ensure_sparse_format(array, accept_sparse=accept_sparse,
    #                                   dtype=dtype, copy=copy,
    #                                   force_all_finite=force_all_finite,
    #                                   accept_large_sparse=accept_large_sparse)
    else:
        # If np.array(..) gives ComplexWarning, then we convert the warning
        # to an error. This is needed because specifying a non complex
        # dtype to the function converts complex to real dtype,
        # thereby passing the test made in the lines following the scope
        # of warnings context manager.
        # with warnings.catch_warnings():
        #     try:
        #         warnings.simplefilter('error', ComplexWarning)
        #         if dtype is not None and np.dtype(dtype).kind in 'iu':
        #             # Conversion float -> int should not contain NaN or
        #             # inf (numpy#14412). We cannot use casting='safe' because
        #             # then conversion float -> int would be disallowed.
        #             array = np.asarray(array, order=order)
        #             if array.dtype.kind == 'f':
        #                 _assert_all_finite(array, allow_nan=False,
        #                                    msg_dtype=dtype)
        #             array = array.astype(dtype, casting="unsafe", copy=False)
        #         else:
        #             array = np.asarray(array, order=order, dtype=dtype)
        #     except ComplexWarning as complex_warning:
        #         raise ValueError("Complex data not supported\n"
        #                          "{}\n".format(array)) from complex_warning

        # It is possible that the np.array(..) gave no warning. This happens
        # when no dtype conversion happened, for example dtype = None. The
        # result is that np.array(..) produces an array of complex dtype
        # and we need to catch and raise exception for such cases.
        # _ensure_no_complex_data(array)

        if ensure_2d:
            # If input is scalar raise error
            if array.ndim == 0:
                raise ValueError(
                    "Expected 2D array, got scalar array instead:\narray={}.\n"
                    "Reshape your data either using array.reshape(-1, 1) if "
                    "your data has a single feature or array.reshape(1, -1) "
                    "if it contains a single sample.".format(array))
            # If input is 1D raise error
            if array.ndim == 1:
                raise ValueError(
                    "Expected 2D array, got 1D array instead:\narray={}.\n"
                    "Reshape your data either using array.reshape(-1, 1) if "
                    "your data has a single feature or array.reshape(1, -1) "
                    "if it contains a single sample.".format(array))

        # in the future np.flexible dtypes will be handled like object dtypes
        if dtype_numeric and np.issubdtype(array.dtype, np.flexible):
            warnings.warn(
                "Beginning in version 0.22, arrays of bytes/strings will be "
                "converted to decimal numbers if dtype='numeric'. "
                "It is recommended that you convert the array to "
                "a float dtype before using it in scikit-learn, "
                "for example by using "
                "your_array = your_array.astype(np.float64).",
                FutureWarning, stacklevel=2)

        # make sure we actually converted to numeric:
        if dtype_numeric and array.dtype.kind == "O":
            array = array.astype(np.float64)
        if not allow_nd and array.ndim >= 3:
            raise ValueError("Found array with dim %d. %s expected <= 2."
                             % (array.ndim, estimator_name))

        if force_all_finite:
            _assert_all_finite(array,
                               allow_nan=force_all_finite == 'allow-nan')

    if ensure_min_samples > 0:
        n_samples = _num_samples(array)
        if n_samples < ensure_min_samples:
            raise ValueError("Found array with %d sample(s) (shape=%s) while a"
                             " minimum of %d is required%s."
                             % (n_samples, array.shape, ensure_min_samples,
                                context))

    if ensure_min_features > 0 and array.ndim == 2:
        n_features = array.shape[1]
        if n_features < ensure_min_features:
            raise ValueError("Found array with %d feature(s) (shape=%s) while"
                             " a minimum of %d is required%s."
                             % (n_features, array.shape, ensure_min_features,
                                context))

    if copy and np.may_share_memory(array, array_orig):
        array = np.array(array, dtype=dtype, order=order)

    return array


# Use at least float64 for the accumulating functions to avoid precision issue
# see https://github.com/numpy/numpy/issues/9393. The float64 is also retained
# as it is in case the float overflows
def _safe_accumulator_op(op, x, *args, **kwargs):
    """
    This function provides numpy accumulator functions with a float64 dtype
    when used on a floating point input. This prevents accumulator overflow on
    smaller floating point dtypes.
    Parameters
    ----------
    op : function
        A numpy accumulator function such as np.mean or np.sum.
    x : ndarray
        A numpy array to apply the accumulator function.
    *args : positional arguments
        Positional arguments passed to the accumulator function after the
        input x.
    **kwargs : keyword arguments
        Keyword arguments passed to the accumulator function.
    Returns
    -------
    result
        The output of the accumulator function passed to this function.
    """
    if np.issubdtype(x.dtype, np.floating) and x.dtype.itemsize < 8:
        result = op(x, *args, **kwargs, dtype=np.float64)
    else:
        result = op(x, *args, **kwargs)
    return result

def _assert_all_finite(X, allow_nan=False, msg_dtype=None):
    """Like assert_all_finite, but only for ndarray."""

    X = np.asanyarray(X)
    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space np.isfinite to prevent
    # false positives from overflow in sum method. The sum is also calculated
    # safely to reduce dtype induced overflows.
    is_float = X.dtype.kind in 'fc'
    if is_float and (np.isfinite(_safe_accumulator_op(np.sum, X))):
        pass
    elif is_float:
        msg_err = "Input contains {} or a value too large for {!r}."
        if (allow_nan and np.isinf(X).any() or
                not allow_nan and not np.isfinite(X).all()):
            type_err = 'infinity' if allow_nan else 'NaN, infinity'
            raise ValueError(
                    msg_err.format
                    (type_err,
                     msg_dtype if msg_dtype is not None else X.dtype)
            )
    # for object dtype data, we only check for NaNs (GH-13254)
    elif X.dtype == np.dtype('object') and not allow_nan:
        if _object_dtype_isnan(X).any():
            raise ValueError("Input contains NaN")

def _object_dtype_isnan(X):
    return X != X

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

def _num_samples(x):
    """Return number of samples in array-like x."""
    message = 'Expected sequence or array-like, got %s' % type(x)
    if hasattr(x, 'fit') and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError(message)

    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError(message)

    if hasattr(x, 'shape') and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered"
                            " a valid collection." % x)
        # Check that shape is returning an integer or default to len
        # Dask dataframes may not return numeric shape[0] value
        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]

    try:
        return len(x)
    except TypeError as type_error:
        raise TypeError(message) from type_error

def issparse(x):
    """Is x of a sparse matrix type?
    Parameters
    ----------
    x
        object to check for being a sparse matrix
    Returns
    -------
    bool
        True if x is a sparse matrix, False otherwise
    """
    return isinstance(x, spmatrix)