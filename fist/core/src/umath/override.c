#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define NO_IMPORT_ARRAY

#include "npy_pycompat.h"
#include "numpy/ufuncobject.h"
#include "npy_import.h"

#include "override.h"
#include "ufunc_override.h"

/*
 * For each positional argument and each argument in a possible "out"
 * keyword, look for overrides of the standard ufunc behaviour, i.e.,
 * non-default __array_ufunc__ methods.
 *
 * Returns the number of overrides, setting corresponding objects
 * in PyObject array ``with_override`` and the corresponding
 * __array_ufunc__ methods in ``methods`` (both using new references).
 *
 * Only the first override for a given class is returned.
 *
 * Returns -1 on failure.
 */
static int
get_array_ufunc_overrides(PyObject *args, PyObject *kwds,
                          PyObject **with_override, PyObject **methods)
{
    int i;
    int num_override_args = 0;
    int narg, nout = 0;
    PyObject *out_kwd_obj;
    PyObject **arg_objs, **out_objs;

    narg = PyTuple_Size(args);
    if (narg < 0) {
        return -1;
    }
    arg_objs = PySequence_Fast_ITEMS(args);

    nout = PyUFuncOverride_GetOutObjects(kwds, &out_kwd_obj, &out_objs);
    if (nout < 0) {
        return -1;
    }

    for (i = 0; i < narg + nout; ++i) {
        PyObject *obj;
        int j;
        int new_class = 1;

        if (i < narg) {
            obj = arg_objs[i];
        }
        else {
            obj = out_objs[i - narg];
        }
        /*
         * Have we seen this class before?  If so, ignore.
         */
        for (j = 0; j < num_override_args; j++) {
            new_class = (Py_TYPE(obj) != Py_TYPE(with_override[j]));
            if (!new_class) {
                break;
            }
        }
        if (new_class) {
            /*
             * Now see if the object provides an __array_ufunc__. However, we should
             * ignore the base ndarray.__ufunc__, so we skip any ndarray as well as
             * any ndarray subclass instances that did not override __array_ufunc__.
             */
            PyObject *method = PyUFuncOverride_GetNonDefaultArrayUfunc(obj);
            if (method == NULL) {
                continue;
            }
            if (method == Py_None) {
                PyErr_Format(PyExc_TypeError,
                             "operand '%.200s' does not support ufuncs "
                             "(__array_ufunc__=None)",
                             obj->ob_type->tp_name);
                Py_DECREF(method);
                goto fail;
            }
            Py_INCREF(obj);
            with_override[num_override_args] = obj;
            methods[num_override_args] = method;
            ++num_override_args;
        }
    }
    Py_DECREF(out_kwd_obj);
    return num_override_args;

fail:
    for (i = 0; i < num_override_args; i++) {
        Py_DECREF(with_override[i]);
        Py_DECREF(methods[i]);
    }
    Py_DECREF(out_kwd_obj);
    return -1;
}

/*
 * The following functions normalize ufunc arguments. The work done is similar
 * to what is done inside ufunc_object by get_ufunc_arguments for __call__ and
 * generalized ufuncs, and by PyUFunc_GenericReduction for the other methods.
 * It would be good to unify (see gh-8892).
 */

/*
 * ufunc() and ufunc.outer() accept 'sig' or 'signature';
 * normalize to 'signature'
 */
static int
normalize_signature_keyword(PyObject *normal_kwds)
{
    PyObject *obj = _PyDict_GetItemStringWithError(normal_kwds, "sig");
    if (obj == NULL && PyErr_Occurred()){
        return -1;
    }
    if (obj != NULL) {
        PyObject *sig = _PyDict_GetItemStringWithError(normal_kwds, "signature");
        if (sig == NULL && PyErr_Occurred()) {
            return -1;
        }
        if (sig) {
            PyErr_SetString(PyExc_TypeError,
                            "cannot specify both 'sig' and 'signature'");
            return -1;
        }
        /*
         * No INCREF or DECREF needed: got a borrowed reference above,
         * and, unlike e.g. PyList_SetItem, PyDict_SetItem INCREF's it.
         */
        PyDict_SetItemString(normal_kwds, "signature", obj);
        PyDict_DelItemString(normal_kwds, "sig");
    }
    return 0;
}

static int
normalize___call___args(PyUFuncObject *ufunc, PyObject *args,
                        PyObject **normal_args, PyObject **normal_kwds)
{
    /*
     * ufunc.__call__(*args, **kwds)
     */
    npy_intp i;
    int not_all_none;
    npy_intp nin = ufunc->nin;
    npy_intp nout = ufunc->nout;
    npy_intp nargs = PyTuple_GET_SIZE(args);
    npy_intp nkwds = PyDict_Size(*normal_kwds);
    PyObject *obj;

    if (nargs < nin) {
        PyErr_Format(PyExc_TypeError,
                     "ufunc() missing %"NPY_INTP_FMT" of %"NPY_INTP_FMT
                     "required positional argument(s)", nin - nargs, nin);
        return -1;
    }
    if (nargs > nin+nout) {
        PyErr_Format(PyExc_TypeError,
                     "ufunc() takes from %"NPY_INTP_FMT" to %"NPY_INTP_FMT
                     "arguments but %"NPY_INTP_FMT" were given",
                     nin, nin+nout, nargs);
        return -1;
    }

    *normal_args = PyTuple_GetSlice(args, 0, nin);
    if (*normal_args == NULL) {
        return -1;
    }

    /* If we have more args than nin, they must be the output variables.*/
    if (nargs > nin) {
        if (nkwds > 0) {
            PyObject *out_kwd = _PyDict_GetItemStringWithError(*normal_kwds, "out");
            if (out_kwd == NULL && PyErr_Occurred()) {
                return -1;
            }
            else if (out_kwd) {
                PyErr_Format(PyExc_TypeError,
                             "argument given by name ('out') and position "
                             "(%"NPY_INTP_FMT")", nin);
                return -1;
            }
        }
        for (i = nin; i < nargs; i++) {
            not_all_none = (PyTuple_GET_ITEM(args, i) != Py_None);
            if (not_all_none) {
                break;
            }
        }
        if (not_all_none) {
            if (nargs - nin == nout) {
                obj = PyTuple_GetSlice(args, nin, nargs);
            }
            else {
                PyObject *item;

                obj = PyTuple_New(nout);
                if (obj == NULL) {
                    return -1;
                }
                for (i = 0; i < nout; i++) {
                    if (i + nin < nargs) {
                        item = PyTuple_GET_ITEM(args, nin+i);
                    }
                    else {
                        item = Py_None;
                    }
                    Py_INCREF(item);
                    PyTuple_SET_ITEM(obj, i, item);
                }
            }
            PyDict_SetItemString(*normal_kwds, "out", obj);
            Py_DECREF(obj);
        }
    }
    /* gufuncs accept either 'axes' or 'axis', but not both */
    if (nkwds >= 2) {
        PyObject *axis_kwd = _PyDict_GetItemStringWithError(*normal_kwds, "axis");
        if (axis_kwd == NULL && PyErr_Occurred()) {
            return -1;
        }
        PyObject *axes_kwd = _PyDict_GetItemStringWithError(*normal_kwds, "axes");
        if (axes_kwd == NULL && PyErr_Occurred()) {
            return -1;
        }
        if (axis_kwd && axes_kwd) {
            PyErr_SetString(PyExc_TypeError,
                            "cannot specify both 'axis' and 'axes'");
            return -1;
        }
    }
    /* finally, ufuncs accept 'sig' or 'signature' normalize to 'signature' */
    return nkwds == 0 ? 0 : normalize_signature_keyword(*normal_kwds);
}

static int
normalize_reduce_args(PyUFuncObject *ufunc, PyObject *args,
                      PyObject **normal_args, PyObject **normal_kwds)
{
    /*
     * ufunc.reduce(a[, axis, dtype, out, keepdims])
     */
    npy_intp nargs = PyTuple_GET_SIZE(args);
    npy_intp i;
    PyObject *obj;
    static PyObject *NoValue = NULL;
    static char *kwlist[] = {"array", "axis", "dtype", "out", "keepdims",
                             "initial", "where"};

    npy_cache_import("numpy", "_NoValue", &NoValue);
    if (NoValue == NULL) return -1;

    if (nargs < 1 || nargs > 7) {
        PyErr_Format(PyExc_TypeError,
                     "ufunc.reduce() takes from 1 to 7 positional "
                     "arguments but %"NPY_INTP_FMT" were given", nargs);
        return -1;
    }
    *normal_args = PyTuple_GetSlice(args, 0, 1);
    if (*normal_args == NULL) {
        return -1;
    }

    for (i = 1; i < nargs; i++) {
        PyObject *kwd = _PyDict_GetItemStringWithError(*normal_kwds, kwlist[i]);
        if (kwd == NULL && PyErr_Occurred()) {
            return -1;
        }
        else if (kwd) {
            PyErr_Format(PyExc_TypeError,
                         "argument given by name ('%s') and position "
                         "(%"NPY_INTP_FMT")", kwlist[i], i);
            return -1;
        }
        obj = PyTuple_GET_ITEM(args, i);
        if (i == 3) {
            /* remove out=None */
            if (obj == Py_None) {
                continue;
            }
            obj = PyTuple_GetSlice(args, 3, 4);
        }
        /* Remove initial=np._NoValue */
        if (i == 5 && obj == NoValue) {
            continue;
        }
        PyDict_SetItemString(*normal_kwds, kwlist[i], obj);
        if (i == 3) {
            Py_DECREF(obj);
        }
    }
    return 0;
}

static int
normalize_accumulate_args(PyUFuncObject *ufunc, PyObject *args,
                          PyObject **normal_args, PyObject **normal_kwds)
{
    /*
     * ufunc.accumulate(a[, axis, dtype, out])
     */
    npy_intp nargs = PyTuple_GET_SIZE(args);
    npy_intp i;
    PyObject *obj;
    static char *kwlist[] = {"array", "axis", "dtype", "out", "keepdims"};

    if (nargs < 1 || nargs > 4) {
        PyErr_Format(PyExc_TypeError,
                     "ufunc.accumulate() takes from 1 to 4 positional "
                     "arguments but %"NPY_INTP_FMT" were given", nargs);
        return -1;
    }
    *normal_args = PyTuple_GetSlice(args, 0, 1);
    if (*normal_args == NULL) {
        return -1;
    }

    for (i = 1; i < nargs; i++) {
        PyObject *kwd = _PyDict_GetItemStringWithError(*normal_kwds, kwlist[i]);
        if (kwd == NULL && PyErr_Occurred()) {
            return -1;
        }
        else if (kwd) {
            PyErr_Format(PyExc_TypeError,
                         "argument given by name ('%s') and position "
                         "(%"NPY_INTP_FMT")", kwlist[i], i);
            return -1;
        }
        obj = PyTuple_GET_ITEM(args, i);
        if (i == 3) {
            /* remove out=None */
            if (obj == Py_None) {
                continue;
            }
            obj = PyTuple_GetSlice(args, 3, 4);
        }
        PyDict_SetItemString(*normal_kwds, kwlist[i], obj);
        if (i == 3) {
            Py_DECREF(obj);
        }
    }
    return 0;
}

static int
normalize_reduceat_args(PyUFuncObject *ufunc, PyObject *args,
                    PyObject **normal_args, PyObject **normal_kwds)
{
    /*
     * ufunc.reduceat(a, indices[, axis, dtype, out])
     * the number of arguments has been checked in PyUFunc_GenericReduction.
     */
    npy_intp i;
    npy_intp nargs = PyTuple_GET_SIZE(args);
    PyObject *obj;
    static char *kwlist[] = {"array", "indices", "axis", "dtype", "out"};

    if (nargs < 2 || nargs > 5) {
        PyErr_Format(PyExc_TypeError,
                     "ufunc.reduceat() takes from 2 to 4 positional "
                     "arguments but %"NPY_INTP_FMT" were given", nargs);
        return -1;
    }
    /* a and indices */
    *normal_args = PyTuple_GetSlice(args, 0, 2);
    if (*normal_args == NULL) {
        return -1;
    }

    for (i = 2; i < nargs; i++) {
        PyObject *kwd = _PyDict_GetItemStringWithError(*normal_kwds, kwlist[i]);
        if (kwd == NULL && PyErr_Occurred()) {
            return -1;
        }
        else if (kwd) {
            PyErr_Format(PyExc_TypeError,
                         "argument given by name ('%s') and position "
                         "(%"NPY_INTP_FMT")", kwlist[i], i);
            return -1;
        }
        obj = PyTuple_GET_ITEM(args, i);
        if (i == 4) {
            /* remove out=None */
            if (obj == Py_None) {
                continue;
            }
            obj = PyTuple_GetSlice(args, 4, 5);
        }
        PyDict_SetItemString(*normal_kwds, kwlist[i], obj);
        if (i == 4) {
            Py_DECREF(obj);
        }
    }
    return 0;
}

static int
normalize_outer_args(PyUFuncObject *ufunc, PyObject *args,
                     PyObject **normal_args, PyObject **normal_kwds)
{
    /*
     * ufunc.outer(*args, **kwds)
     * all positional arguments should be inputs.
     * for the keywords, we only need to check 'sig' vs 'signature'.
     */
    npy_intp nin = ufunc->nin;
    npy_intp nargs = PyTuple_GET_SIZE(args);

    if (nargs < nin) {
        PyErr_Format(PyExc_TypeError,
                     "ufunc.outer() missing %"NPY_INTP_FMT" of %"NPY_INTP_FMT
                     "required positional " "argument(s)", nin - nargs, nin);
        return -1;
    }
    if (nargs > nin) {
        PyErr_Format(PyExc_TypeError,
                     "ufunc.outer() takes %"NPY_INTP_FMT" arguments but"
                     "%"NPY_INTP_FMT" were given", nin, nargs);
        return -1;
    }

    *normal_args = PyTuple_GetSlice(args, 0, nin);
    if (*normal_args == NULL) {
        return -1;
    }
    /* ufuncs accept 'sig' or 'signature' normalize to 'signature' */
    return normalize_signature_keyword(*normal_kwds);
}

static int
normalize_at_args(PyUFuncObject *ufunc, PyObject *args,
                  PyObject **normal_args, PyObject **normal_kwds)
{
    /* ufunc.at(a, indices[, b]) */
    npy_intp nargs = PyTuple_GET_SIZE(args);

    if (nargs < 2 || nargs > 3) {
        PyErr_Format(PyExc_TypeError,
                     "ufunc.at() takes from 2 to 3 positional "
                     "arguments but %"NPY_INTP_FMT" were given", nargs);
        return -1;
    }
    *normal_args = PyTuple_GetSlice(args, 0, nargs);
    return (*normal_args == NULL);
}

/*
 * Check a set of args for the `__array_ufunc__` method.  If more than one of
 * the input arguments implements `__array_ufunc__`, they are tried in the
 * order: subclasses before superclasses, otherwise left to right. The first
 * (non-None) routine returning something other than `NotImplemented`
 * determines the result. If all of the `__array_ufunc__` operations return
 * `NotImplemented` (or are None), a `TypeError` is raised.
 *
 * Returns 0 on success and 1 on exception. On success, *result contains the
 * result of the operation, if any. If *result is NULL, there is no override.
 */
NPY_NO_EXPORT int
PyUFunc_CheckOverride(PyUFuncObject *ufunc, char *method,
                      PyObject *args, PyObject *kwds,
                      PyObject **result)
{
    int i;
    int j;
    int status;

    int num_override_args;
    PyObject *with_override[NPY_MAXARGS];
    PyObject *array_ufunc_methods[NPY_MAXARGS];

    PyObject *out;

    PyObject *method_name = NULL;
    PyObject *normal_args = NULL; /* normal_* holds normalized arguments. */
    PyObject *normal_kwds = NULL;

    PyObject *override_args = NULL;
    Py_ssize_t len;

    /*
     * Check inputs for overrides
     */
    num_override_args = get_array_ufunc_overrides(
        args, kwds, with_override, array_ufunc_methods);
    if (num_override_args == -1) {
        goto fail;
    }
    /* No overrides, bail out.*/
    if (num_override_args == 0) {
        *result = NULL;
        return 0;
    }

    /*
     * Normalize ufunc arguments.
     */

    /* Build new kwds */
    if (kwds && PyDict_CheckExact(kwds)) {

        /* ensure out is always a tuple */
        normal_kwds = PyDict_Copy(kwds);
        out = _PyDict_GetItemStringWithError(normal_kwds, "out");
        if (out == NULL && PyErr_Occurred()) {
            goto fail;
        }
        else if (out) {
            int nout = ufunc->nout;

            if (PyTuple_CheckExact(out)) {
                int all_none = 1;

                if (PyTuple_GET_SIZE(out) != nout) {
                    PyErr_Format(PyExc_ValueError,
                                 "The 'out' tuple must have exactly "
                                 "%d entries: one per ufunc output", nout);
                    goto fail;
                }
                for (i = 0; i < PyTuple_GET_SIZE(out); i++) {
                    all_none = (PyTuple_GET_ITEM(out, i) == Py_None);
                    if (!all_none) {
                        break;
                    }
                }
                if (all_none) {
                    PyDict_DelItemString(normal_kwds, "out");
                }
            }
            else {
                /* not a tuple */
                if (nout > 1) {
                    PyErr_SetString(PyExc_TypeError,
                                    "'out' must be a tuple of arguments");
                    goto fail;
                }
                if (out != Py_None) {
                    /* not already a tuple and not None */
                    PyObject *out_tuple = PyTuple_New(1);

                    if (out_tuple == NULL) {
                        goto fail;
                    }
                    /* out was borrowed ref; make it permanent */
                    Py_INCREF(out);
                    /* steals reference */
                    PyTuple_SET_ITEM(out_tuple, 0, out);
                    PyDict_SetItemString(normal_kwds, "out", out_tuple);
                    Py_DECREF(out_tuple);
                }
                else {
                    /* out=None; remove it */
                    PyDict_DelItemString(normal_kwds, "out");
                }
            }
        }
    }
    else {
        normal_kwds = PyDict_New();
    }
    if (normal_kwds == NULL) {
        goto fail;
    }

    /* decide what to do based on the method. */

    /* ufunc.__call__ */
    if (strcmp(method, "__call__") == 0) {
        status = normalize___call___args(ufunc, args, &normal_args,
                                         &normal_kwds);
    }
    /* ufunc.reduce */
    else if (strcmp(method, "reduce") == 0) {
        status = normalize_reduce_args(ufunc, args, &normal_args,
                                       &normal_kwds);
    }
    /* ufunc.accumulate */
    else if (strcmp(method, "accumulate") == 0) {
        status = normalize_accumulate_args(ufunc, args, &normal_args,
                                           &normal_kwds);
    }
    /* ufunc.reduceat */
    else if (strcmp(method, "reduceat") == 0) {
        status = normalize_reduceat_args(ufunc, args, &normal_args,
                                         &normal_kwds);
    }
    /* ufunc.outer */
    else if (strcmp(method, "outer") == 0) {
        status = normalize_outer_args(ufunc, args, &normal_args, &normal_kwds);
    }
    /* ufunc.at */
    else if (strcmp(method, "at") == 0) {
        status = normalize_at_args(ufunc, args, &normal_args, &normal_kwds);
    }
    /* unknown method */
    else {
        PyErr_Format(PyExc_TypeError,
                     "Internal Numpy error: unknown ufunc method '%s' in call "
                     "to PyUFunc_CheckOverride", method);
        status = -1;
    }
    if (status != 0) {
        goto fail;
    }

    method_name = PyUnicode_FromString(method);
    if (method_name == NULL) {
        goto fail;
    }

    len = PyTuple_GET_SIZE(normal_args);

    /* Call __array_ufunc__ functions in correct order */
    while (1) {
        PyObject *override_obj;
        PyObject *override_array_ufunc;

        override_obj = NULL;
        *result = NULL;

        /* Choose an overriding argument */
        for (i = 0; i < num_override_args; i++) {
            override_obj = with_override[i];
            if (override_obj == NULL) {
                continue;
            }

            /* Check for sub-types to the right of obj. */
            for (j = i + 1; j < num_override_args; j++) {
                PyObject *other_obj = with_override[j];
                if (other_obj != NULL &&
                    Py_TYPE(other_obj) != Py_TYPE(override_obj) &&
                    PyObject_IsInstance(other_obj,
                                        (PyObject *)Py_TYPE(override_obj))) {
                    override_obj = NULL;
                    break;
                }
            }

            /* override_obj had no subtypes to the right. */
            if (override_obj) {
                override_array_ufunc = array_ufunc_methods[i];
                /* We won't call this one again (references decref'd below) */
                with_override[i] = NULL;
                array_ufunc_methods[i] = NULL;
                break;
            }
        }
        /*
         * Set override arguments for each call since the tuple must
         * not be mutated after use in PyPy
         * We increase all references since SET_ITEM steals
         * them and they will be DECREF'd when the tuple is deleted.
         */
        override_args = PyTuple_New(len + 3);
        if (override_args == NULL) {
            goto fail;
        }
        Py_INCREF(ufunc);
        PyTuple_SET_ITEM(override_args, 1, (PyObject *)ufunc);
        Py_INCREF(method_name);
        PyTuple_SET_ITEM(override_args, 2, method_name);
        for (i = 0; i < len; i++) {
            PyObject *item = PyTuple_GET_ITEM(normal_args, i);

            Py_INCREF(item);
            PyTuple_SET_ITEM(override_args, i + 3, item);
        }

        /* Check if there is a method left to call */
        if (!override_obj) {
            /* No acceptable override found. */
            static PyObject *errmsg_formatter = NULL;
            PyObject *errmsg;

            npy_cache_import("numpy.core._internal",
                             "array_ufunc_errmsg_formatter",
                             &errmsg_formatter);

            if (errmsg_formatter != NULL) {
                /* All tuple items must be set before use */
                Py_INCREF(Py_None);
                PyTuple_SET_ITEM(override_args, 0, Py_None);
                errmsg = PyObject_Call(errmsg_formatter, override_args,
                                       normal_kwds);
                if (errmsg != NULL) {
                    PyErr_SetObject(PyExc_TypeError, errmsg);
                    Py_DECREF(errmsg);
                }
            }
            Py_DECREF(override_args);
            goto fail;
        }

        /*
         * Set the self argument of our unbound method.
         * This also steals the reference, so no need to DECREF after.
         */
        PyTuple_SET_ITEM(override_args, 0, override_obj);
        /* Call the method */
        *result = PyObject_Call(
            override_array_ufunc, override_args, normal_kwds);
        Py_DECREF(override_array_ufunc);
        Py_DECREF(override_args);
        if (*result == NULL) {
            /* Exception occurred */
            goto fail;
        }
        else if (*result == Py_NotImplemented) {
            /* Try the next one */
            Py_DECREF(*result);
            continue;
        }
        else {
            /* Good result. */
            break;
        }
    }
    status = 0;
    /* Override found, return it. */
    goto cleanup;
fail:
    status = -1;
cleanup:
    for (i = 0; i < num_override_args; i++) {
        Py_XDECREF(with_override[i]);
        Py_XDECREF(array_ufunc_methods[i]);
    }
    Py_XDECREF(normal_args);
    Py_XDECREF(method_name);
    Py_XDECREF(normal_kwds);
    return status;
}
