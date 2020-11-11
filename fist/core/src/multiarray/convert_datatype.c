#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "npy_config.h"

#include "npy_pycompat.h"
#include "numpy/npy_math.h"

#include "array_coercion.h"
#include "common.h"
#include "ctors.h"
#include "dtypemeta.h"
#include "scalartypes.h"
#include "mapping.h"

#include "convert_datatype.h"
#include "_datetime.h"
#include "datetime_strings.h"


/*
 * Required length of string when converting from unsigned integer type.
 * Array index is integer size in bytes.
 * - 3 chars needed for cast to max value of 255 or 127
 * - 5 chars needed for cast to max value of 65535 or 32767
 * - 10 chars needed for cast to max value of 4294967295 or 2147483647
 * - 20 chars needed for cast to max value of 18446744073709551615
 *   or 9223372036854775807
 */
NPY_NO_EXPORT npy_intp REQUIRED_STR_LEN[] = {0, 3, 5, 10, 10, 20, 20, 20, 20};

/*NUMPY_API
 * For backward compatibility
 *
 * Cast an array using typecode structure.
 * steals reference to dtype --- cannot be NULL
 *
 * This function always makes a copy of arr, even if the dtype
 * doesn't change.
 */
NPY_NO_EXPORT PyObject *
PyArray_CastToType(PyArrayObject *arr, PyArray_Descr *dtype, int is_f_order)
{
    PyObject *out;

    Py_SETREF(dtype, PyArray_AdaptDescriptorToArray(arr, (PyObject *)dtype));
    if (dtype == NULL) {
        return NULL;
    }

    out = PyArray_NewFromDescr(Py_TYPE(arr), dtype,
                               PyArray_NDIM(arr),
                               PyArray_DIMS(arr),
                               NULL, NULL,
                               is_f_order,
                               (PyObject *)arr);

    if (out == NULL) {
        return NULL;
    }

    if (PyArray_CopyInto((PyArrayObject *)out, arr) < 0) {
        Py_DECREF(out);
        return NULL;
    }

    return out;
}

/*NUMPY_API
 * Get a cast function to cast from the input descriptor to the
 * output type_number (must be a registered data-type).
 * Returns NULL if un-successful.
 */
NPY_NO_EXPORT PyArray_VectorUnaryFunc *
PyArray_GetCastFunc(PyArray_Descr *descr, int type_num)
{
    PyArray_VectorUnaryFunc *castfunc = NULL;

    if (type_num < NPY_NTYPES_ABI_COMPATIBLE) {
        castfunc = descr->f->cast[type_num];
    }
    else {
        PyObject *obj = descr->f->castdict;
        if (obj && PyDict_Check(obj)) {
            PyObject *key;
            PyObject *cobj;

            key = PyLong_FromLong(type_num);
            cobj = PyDict_GetItem(obj, key);
            Py_DECREF(key);
            if (cobj && PyCapsule_CheckExact(cobj)) {
                castfunc = PyCapsule_GetPointer(cobj, NULL);
                if (castfunc == NULL) {
                    return NULL;
                }
            }
        }
    }
    if (PyTypeNum_ISCOMPLEX(descr->type_num) &&
            !PyTypeNum_ISCOMPLEX(type_num) &&
            PyTypeNum_ISNUMBER(type_num) &&
            !PyTypeNum_ISBOOL(type_num)) {
        PyObject *cls = NULL, *obj = NULL;
        int ret;
        obj = PyImport_ImportModule("numpy.core");

        if (obj) {
            cls = PyObject_GetAttrString(obj, "ComplexWarning");
            Py_DECREF(obj);
        }
        ret = PyErr_WarnEx(cls,
                "Casting complex values to real discards "
                "the imaginary part", 1);
        Py_XDECREF(cls);
        if (ret < 0) {
            return NULL;
        }
    }
    if (castfunc) {
        return castfunc;
    }

    PyErr_SetString(PyExc_ValueError,
            "No cast function available.");
    return NULL;
}

/*
 * Legacy function to find the correct dtype when casting from any built-in
 * dtype to NPY_STRING, NPY_UNICODE, NPY_VOID, and NPY_DATETIME with generic
 * units.
 *
 * This function returns a dtype based on flex_dtype and the values in
 * data_dtype. It also calls Py_DECREF on the flex_dtype. If the
 * flex_dtype is not flexible, it returns it as-is.
 *
 * Usually, if data_obj is not an array, dtype should be the result
 * given by the PyArray_GetArrayParamsFromObject function.
 *
 * If *flex_dtype is NULL, returns immediately, without setting an
 * exception, leaving any previous error handling intact.
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_AdaptFlexibleDType(PyArray_Descr *data_dtype, PyArray_Descr *flex_dtype)
{
    PyArray_DatetimeMetaData *meta;
    PyArray_Descr *retval = NULL;
    int flex_type_num;

    if (flex_dtype == NULL) {
        return retval;
    }

    flex_type_num = flex_dtype->type_num;

    /* Flexible types with expandable size */
    if (PyDataType_ISUNSIZED(flex_dtype)) {
        /* First replace the flex_dtype */
        retval = PyArray_DescrNew(flex_dtype);
        Py_DECREF(flex_dtype);
        if (retval == NULL) {
            return retval;
        }

        if (data_dtype->type_num == flex_type_num ||
                                    flex_type_num == NPY_VOID) {
            (retval)->elsize = data_dtype->elsize;
        }
        else if (flex_type_num == NPY_STRING || flex_type_num == NPY_UNICODE) {
            npy_intp size = 8;

            /*
             * Get a string-size estimate of the input. These
             * are generallly the size needed, rounded up to
             * a multiple of eight.
             */
            switch (data_dtype->type_num) {
                case NPY_BOOL:
                case NPY_UBYTE:
                case NPY_BYTE:
                case NPY_USHORT:
                case NPY_SHORT:
                case NPY_UINT:
                case NPY_INT:
                case NPY_ULONG:
                case NPY_LONG:
                case NPY_ULONGLONG:
                case NPY_LONGLONG:
                    if (data_dtype->kind == 'b') {
                        /* 5 chars needed for cast to 'True' or 'False' */
                        size = 5;
                    }
                    else if (data_dtype->elsize > 8 ||
                             data_dtype->elsize < 0) {
                        /*
                         * Element size should never be greater than 8 or
                         * less than 0 for integer type, but just in case...
                         */
                        break;
                    }
                    else if (data_dtype->kind == 'u') {
                        size = REQUIRED_STR_LEN[data_dtype->elsize];
                    }
                    else if (data_dtype->kind == 'i') {
                        /* Add character for sign symbol */
                        size = REQUIRED_STR_LEN[data_dtype->elsize] + 1;
                    }
                    break;
                case NPY_HALF:
                case NPY_FLOAT:
                case NPY_DOUBLE:
                    size = 32;
                    break;
                case NPY_LONGDOUBLE:
                    size = 48;
                    break;
                case NPY_CFLOAT:
                case NPY_CDOUBLE:
                    size = 2 * 32;
                    break;
                case NPY_CLONGDOUBLE:
                    size = 2 * 48;
                    break;
                case NPY_OBJECT:
                    size = 64;
                    break;
                case NPY_STRING:
                case NPY_VOID:
                    size = data_dtype->elsize;
                    break;
                case NPY_UNICODE:
                    size = data_dtype->elsize / 4;
                    break;
                case NPY_DATETIME:
                    meta = get_datetime_metadata_from_dtype(data_dtype);
                    if (meta == NULL) {
                        Py_DECREF(retval);
                        return NULL;
                    }
                    size = get_datetime_iso_8601_strlen(0, meta->base);
                    break;
                case NPY_TIMEDELTA:
                    size = 21;
                    break;
            }

            if (flex_type_num == NPY_STRING) {
                retval->elsize = size;
            }
            else if (flex_type_num == NPY_UNICODE) {
                retval->elsize = size * 4;
            }
        }
        else {
            /*
             * We should never get here, but just in case someone adds
             * a new flex dtype...
             */
            PyErr_SetString(PyExc_TypeError,
                    "don't know how to adapt flex dtype");
            Py_DECREF(retval);
            return NULL;
        }
    }
    /* Flexible type with generic time unit that adapts */
    else if (flex_type_num == NPY_DATETIME ||
                flex_type_num == NPY_TIMEDELTA) {
        meta = get_datetime_metadata_from_dtype(flex_dtype);
        retval = flex_dtype;
        if (meta == NULL) {
            return NULL;
        }

        if (meta->base == NPY_FR_GENERIC) {
            if (data_dtype->type_num == NPY_DATETIME ||
                    data_dtype->type_num == NPY_TIMEDELTA) {
                meta = get_datetime_metadata_from_dtype(data_dtype);
                if (meta == NULL) {
                    return NULL;
                }

                retval = create_datetime_dtype(flex_type_num, meta);
                Py_DECREF(flex_dtype);
            }
        }
    }
    else {
        retval = flex_dtype;
    }
    return retval;
}

/*
 * Must be broadcastable.
 * This code is very similar to PyArray_CopyInto/PyArray_MoveInto
 * except casting is done --- NPY_BUFSIZE is used
 * as the size of the casting buffer.
 */

/*NUMPY_API
 * Cast to an already created array.
 */
NPY_NO_EXPORT int
PyArray_CastTo(PyArrayObject *out, PyArrayObject *mp)
{
    /* CopyInto handles the casting now */
    return PyArray_CopyInto(out, mp);
}

/*NUMPY_API
 * Cast to an already created array.  Arrays don't have to be "broadcastable"
 * Only requirement is they have the same number of elements.
 */
NPY_NO_EXPORT int
PyArray_CastAnyTo(PyArrayObject *out, PyArrayObject *mp)
{
    /* CopyAnyInto handles the casting now */
    return PyArray_CopyAnyInto(out, mp);
}

/*NUMPY_API
 *Check the type coercion rules.
 */
NPY_NO_EXPORT int
PyArray_CanCastSafely(int fromtype, int totype)
{
    PyArray_Descr *from;

    /* Fast table lookup for small type numbers */
    if ((unsigned int)fromtype < NPY_NTYPES &&
                                (unsigned int)totype < NPY_NTYPES) {
        return _npy_can_cast_safely_table[fromtype][totype];
    }

    /* Identity */
    if (fromtype == totype) {
        return 1;
    }

    from = PyArray_DescrFromType(fromtype);
    /*
     * cancastto is a NPY_NOTYPE terminated C-int-array of types that
     * the data-type can be cast to safely.
     */
    if (from->f->cancastto) {
        int *curtype = from->f->cancastto;

        while (*curtype != NPY_NOTYPE) {
            if (*curtype++ == totype) {
                return 1;
            }
        }
    }
    return 0;
}

/*NUMPY_API
 * leaves reference count alone --- cannot be NULL
 *
 * PyArray_CanCastTypeTo is equivalent to this, but adds a 'casting'
 * parameter.
 */
NPY_NO_EXPORT npy_bool
PyArray_CanCastTo(PyArray_Descr *from, PyArray_Descr *to)
{
    int from_type_num = from->type_num;
    int to_type_num = to->type_num;
    npy_bool ret;

    ret = (npy_bool) PyArray_CanCastSafely(from_type_num, to_type_num);
    if (ret) {
        /* Check String and Unicode more closely */
        if (from_type_num == NPY_STRING) {
            if (to_type_num == NPY_STRING) {
                ret = (from->elsize <= to->elsize);
            }
            else if (to_type_num == NPY_UNICODE) {
                ret = (from->elsize << 2 <= to->elsize);
            }
        }
        else if (from_type_num == NPY_UNICODE) {
            if (to_type_num == NPY_UNICODE) {
                ret = (from->elsize <= to->elsize);
            }
        }
        /*
         * For datetime/timedelta, only treat casts moving towards
         * more precision as safe.
         */
        else if (from_type_num == NPY_DATETIME && to_type_num == NPY_DATETIME) {
            PyArray_DatetimeMetaData *meta1, *meta2;
            meta1 = get_datetime_metadata_from_dtype(from);
            if (meta1 == NULL) {
                PyErr_Clear();
                return 0;
            }
            meta2 = get_datetime_metadata_from_dtype(to);
            if (meta2 == NULL) {
                PyErr_Clear();
                return 0;
            }

            return can_cast_datetime64_metadata(meta1, meta2,
                                                NPY_SAFE_CASTING);
        }
        else if (from_type_num == NPY_TIMEDELTA &&
                                    to_type_num == NPY_TIMEDELTA) {
            PyArray_DatetimeMetaData *meta1, *meta2;
            meta1 = get_datetime_metadata_from_dtype(from);
            if (meta1 == NULL) {
                PyErr_Clear();
                return 0;
            }
            meta2 = get_datetime_metadata_from_dtype(to);
            if (meta2 == NULL) {
                PyErr_Clear();
                return 0;
            }

            return can_cast_timedelta64_metadata(meta1, meta2,
                                                 NPY_SAFE_CASTING);
        }
        /*
         * If to_type_num is STRING or unicode
         * see if the length is long enough to hold the
         * stringified value of the object.
         */
        else if (to_type_num == NPY_STRING || to_type_num == NPY_UNICODE) {
            /*
             * Boolean value cast to string type is 5 characters max
             * for string 'False'.
             */
            int char_size = 1;
            if (to_type_num == NPY_UNICODE) {
                char_size = 4;
            }

            ret = 0;
            if (PyDataType_ISUNSIZED(to)) {
                ret = 1;
            }
            /*
             * Need at least 5 characters to convert from boolean
             * to 'True' or 'False'.
             */
            else if (from->kind == 'b' && to->elsize >= 5 * char_size) {
                ret = 1;
            }
            else if (from->kind == 'u') {
                /* Guard against unexpected integer size */
                if (from->elsize > 8 || from->elsize < 0) {
                    ret = 0;
                }
                else if (to->elsize >=
                        REQUIRED_STR_LEN[from->elsize] * char_size) {
                    ret = 1;
                }
            }
            else if (from->kind == 'i') {
                /* Guard against unexpected integer size */
                if (from->elsize > 8 || from->elsize < 0) {
                    ret = 0;
                }
                /* Extra character needed for sign */
                else if (to->elsize >=
                        (REQUIRED_STR_LEN[from->elsize] + 1) * char_size) {
                    ret = 1;
                }
            }
        }
    }
    return ret;
}

/* Provides an ordering for the dtype 'kind' character codes */
static int
dtype_kind_to_ordering(char kind)
{
    switch (kind) {
        /* Boolean kind */
        case 'b':
            return 0;
        /* Unsigned int kind */
        case 'u':
            return 1;
        /* Signed int kind */
        case 'i':
            return 2;
        /* Float kind */
        case 'f':
            return 4;
        /* Complex kind */
        case 'c':
            return 5;
        /* String kind */
        case 'S':
        case 'a':
            return 6;
        /* Unicode kind */
        case 'U':
            return 7;
        /* Void kind */
        case 'V':
            return 8;
        /* Object kind */
        case 'O':
            return 9;
        /*
         * Anything else, like datetime, is special cased to
         * not fit in this hierarchy
         */
        default:
            return -1;
    }
}

/* Converts a type number from unsigned to signed */
static int
type_num_unsigned_to_signed(int type_num)
{
    switch (type_num) {
        case NPY_UBYTE:
            return NPY_BYTE;
        case NPY_USHORT:
            return NPY_SHORT;
        case NPY_UINT:
            return NPY_INT;
        case NPY_ULONG:
            return NPY_LONG;
        case NPY_ULONGLONG:
            return NPY_LONGLONG;
        default:
            return type_num;
    }
}

/*
 * Compare two field dictionaries for castability.
 *
 * Return 1 if 'field1' can be cast to 'field2' according to the rule
 * 'casting', 0 if not.
 *
 * Castabiliy of field dictionaries is defined recursively: 'field1' and
 * 'field2' must have the same field names (possibly in different
 * orders), and the corresponding field types must be castable according
 * to the given casting rule.
 */
static int
can_cast_fields(PyObject *field1, PyObject *field2, NPY_CASTING casting)
{
    Py_ssize_t ppos;
    PyObject *key;
    PyObject *tuple1, *tuple2;

    if (field1 == field2) {
        return 1;
    }
    if (field1 == NULL || field2 == NULL) {
        return 0;
    }
    if (PyDict_Size(field1) != PyDict_Size(field2)) {
        return 0;
    }

    /* Iterate over all the fields and compare for castability */
    ppos = 0;
    while (PyDict_Next(field1, &ppos, &key, &tuple1)) {
        if ((tuple2 = PyDict_GetItem(field2, key)) == NULL) {
            return 0;
        }
        /* Compare the dtype of the field for castability */
        if (!PyArray_CanCastTypeTo(
                        (PyArray_Descr *)PyTuple_GET_ITEM(tuple1, 0),
                        (PyArray_Descr *)PyTuple_GET_ITEM(tuple2, 0),
                        casting)) {
            return 0;
        }
    }

    return 1;
}

/*NUMPY_API
 * Returns true if data of type 'from' may be cast to data of type
 * 'to' according to the rule 'casting'.
 */
NPY_NO_EXPORT npy_bool
PyArray_CanCastTypeTo(PyArray_Descr *from, PyArray_Descr *to,
                                                    NPY_CASTING casting)
{
    /*
     * Fast paths for equality and for basic types.
     */
    if (from == to ||
        ((NPY_LIKELY(PyDataType_ISNUMBER(from)) ||
          PyDataType_ISOBJECT(from)) &&
         NPY_LIKELY(from->type_num == to->type_num) &&
         NPY_LIKELY(from->byteorder == to->byteorder))) {
        return 1;
    }
    /*
     * Cases with subarrays and fields need special treatment.
     */
    if (PyDataType_HASFIELDS(from)) {
        /*
         * If from is a structured data type, then it can be cast to a simple
         * non-object one only for unsafe casting *and* if it has a single
         * field; recurse just in case the single field is itself structured.
         */
        if (!PyDataType_HASFIELDS(to) && !PyDataType_ISOBJECT(to)) {
            if (casting == NPY_UNSAFE_CASTING &&
                    PyDict_Size(from->fields) == 1) {
                Py_ssize_t ppos = 0;
                PyObject *tuple;
                PyArray_Descr *field;
                PyDict_Next(from->fields, &ppos, NULL, &tuple);
                field = (PyArray_Descr *)PyTuple_GET_ITEM(tuple, 0);
                /*
                 * For a subarray, we need to get the underlying type;
                 * since we already are casting unsafely, we can ignore
                 * the shape.
                 */
                if (PyDataType_HASSUBARRAY(field)) {
                    field = field->subarray->base;
                }
                return PyArray_CanCastTypeTo(field, to, casting);
            }
            else {
                return 0;
            }
        }
        /*
         * Casting from one structured data type to another depends on the fields;
         * we pass that case on to the EquivTypenums case below.
         *
         * TODO: move that part up here? Need to check whether equivalent type
         * numbers is an addition constraint that is needed.
         *
         * TODO/FIXME: For now, always allow structured to structured for unsafe
         * casting; this is not correct, but needed since the treatment in can_cast
         * below got out of sync with astype; see gh-13667.
         */
        if (casting == NPY_UNSAFE_CASTING) {
            return 1;
        }
    }
    else if (PyDataType_HASFIELDS(to)) {
        /*
         * If "from" is a simple data type and "to" has fields, then only
         * unsafe casting works (and that works always, even to multiple fields).
         */
        return casting == NPY_UNSAFE_CASTING;
    }
    /*
     * Everything else we consider castable for unsafe for now.
     * FIXME: ensure what we do here is consistent with "astype",
     * i.e., deal more correctly with subarrays and user-defined dtype.
     */
    else if (casting == NPY_UNSAFE_CASTING) {
        return 1;
    }
    /*
     * Equivalent simple types can be cast with any value of 'casting', but
     * we need to be careful about structured to structured.
     */
    if (PyArray_EquivTypenums(from->type_num, to->type_num)) {
        /* For complicated case, use EquivTypes (for now) */
        if (PyTypeNum_ISUSERDEF(from->type_num) ||
                        from->subarray != NULL) {
            int ret;

            /* Only NPY_NO_CASTING prevents byte order conversion */
            if ((casting != NPY_NO_CASTING) &&
                                (!PyArray_ISNBO(from->byteorder) ||
                                 !PyArray_ISNBO(to->byteorder))) {
                PyArray_Descr *nbo_from, *nbo_to;

                nbo_from = PyArray_DescrNewByteorder(from, NPY_NATIVE);
                nbo_to = PyArray_DescrNewByteorder(to, NPY_NATIVE);
                if (nbo_from == NULL || nbo_to == NULL) {
                    Py_XDECREF(nbo_from);
                    Py_XDECREF(nbo_to);
                    PyErr_Clear();
                    return 0;
                }
                ret = PyArray_EquivTypes(nbo_from, nbo_to);
                Py_DECREF(nbo_from);
                Py_DECREF(nbo_to);
            }
            else {
                ret = PyArray_EquivTypes(from, to);
            }
            return ret;
        }

        if (PyDataType_HASFIELDS(from)) {
            switch (casting) {
                case NPY_EQUIV_CASTING:
                case NPY_SAFE_CASTING:
                case NPY_SAME_KIND_CASTING:
                    /*
                     * `from' and `to' must have the same fields, and
                     * corresponding fields must be (recursively) castable.
                     */
                    return can_cast_fields(from->fields, to->fields, casting);

                case NPY_NO_CASTING:
                default:
                    return PyArray_EquivTypes(from, to);
            }
        }

        switch (from->type_num) {
            case NPY_DATETIME: {
                PyArray_DatetimeMetaData *meta1, *meta2;
                meta1 = get_datetime_metadata_from_dtype(from);
                if (meta1 == NULL) {
                    PyErr_Clear();
                    return 0;
                }
                meta2 = get_datetime_metadata_from_dtype(to);
                if (meta2 == NULL) {
                    PyErr_Clear();
                    return 0;
                }

                if (casting == NPY_NO_CASTING) {
                    return PyArray_ISNBO(from->byteorder) ==
                                        PyArray_ISNBO(to->byteorder) &&
                            can_cast_datetime64_metadata(meta1, meta2, casting);
                }
                else {
                    return can_cast_datetime64_metadata(meta1, meta2, casting);
                }
            }
            case NPY_TIMEDELTA: {
                PyArray_DatetimeMetaData *meta1, *meta2;
                meta1 = get_datetime_metadata_from_dtype(from);
                if (meta1 == NULL) {
                    PyErr_Clear();
                    return 0;
                }
                meta2 = get_datetime_metadata_from_dtype(to);
                if (meta2 == NULL) {
                    PyErr_Clear();
                    return 0;
                }

                if (casting == NPY_NO_CASTING) {
                    return PyArray_ISNBO(from->byteorder) ==
                                        PyArray_ISNBO(to->byteorder) &&
                        can_cast_timedelta64_metadata(meta1, meta2, casting);
                }
                else {
                    return can_cast_timedelta64_metadata(meta1, meta2, casting);
                }
            }
            default:
                switch (casting) {
                    case NPY_NO_CASTING:
                        return PyArray_EquivTypes(from, to);
                    case NPY_EQUIV_CASTING:
                        return (from->elsize == to->elsize);
                    case NPY_SAFE_CASTING:
                        return (from->elsize <= to->elsize);
                    default:
                        return 1;
                }
                break;
        }
    }
    /* If safe or same-kind casts are allowed */
    else if (casting == NPY_SAFE_CASTING || casting == NPY_SAME_KIND_CASTING) {
        if (PyArray_CanCastTo(from, to)) {
            return 1;
        }
        else if(casting == NPY_SAME_KIND_CASTING) {
            /*
             * Also allow casting from lower to higher kinds, according
             * to the ordering provided by dtype_kind_to_ordering.
             * Some kinds, like datetime, don't fit in the hierarchy,
             * and are special cased as -1.
             */
            int from_order, to_order;

            from_order = dtype_kind_to_ordering(from->kind);
            to_order = dtype_kind_to_ordering(to->kind);

            if (to->kind == 'm') {
                /* both types being timedelta is already handled before. */
                int integer_order = dtype_kind_to_ordering('i');
                return (from_order != -1) && (from_order <= integer_order);
            }

            return (from_order != -1) && (from_order <= to_order);
        }
        else {
            return 0;
        }
    }
    /* NPY_NO_CASTING or NPY_EQUIV_CASTING was specified */
    else {
        return 0;
    }
}

/* CanCastArrayTo needs this function */
static int min_scalar_type_num(char *valueptr, int type_num,
                                            int *is_small_unsigned);

NPY_NO_EXPORT npy_bool
can_cast_scalar_to(PyArray_Descr *scal_type, char *scal_data,
                    PyArray_Descr *to, NPY_CASTING casting)
{
    int swap;
    int is_small_unsigned = 0, type_num;
    npy_bool ret;
    PyArray_Descr *dtype;

    /* An aligned memory buffer large enough to hold any type */
    npy_longlong value[4];

    /*
     * If the two dtypes are actually references to the same object
     * or if casting type is forced unsafe then always OK.
     */
    if (scal_type == to || casting == NPY_UNSAFE_CASTING ) {
        return 1;
    }

    /*
     * If the scalar isn't a number, or the rule is stricter than
     * NPY_SAFE_CASTING, use the straight type-based rules
     */
    if (!PyTypeNum_ISNUMBER(scal_type->type_num) ||
                            casting < NPY_SAFE_CASTING) {
        return PyArray_CanCastTypeTo(scal_type, to, casting);
    }

    swap = !PyArray_ISNBO(scal_type->byteorder);
    scal_type->f->copyswap(&value, scal_data, swap, NULL);

    type_num = min_scalar_type_num((char *)&value, scal_type->type_num,
                                    &is_small_unsigned);

    /*
     * If we've got a small unsigned scalar, and the 'to' type
     * is not unsigned, then make it signed to allow the value
     * to be cast more appropriately.
     */
    if (is_small_unsigned && !(PyTypeNum_ISUNSIGNED(to->type_num))) {
        type_num = type_num_unsigned_to_signed(type_num);
    }

    dtype = PyArray_DescrFromType(type_num);
    if (dtype == NULL) {
        return 0;
    }
#if 0
    printf("min scalar cast ");
    PyObject_Print(dtype, stdout, 0);
    printf(" to ");
    PyObject_Print(to, stdout, 0);
    printf("\n");
#endif
    ret = PyArray_CanCastTypeTo(dtype, to, casting);
    Py_DECREF(dtype);
    return ret;
}

/*NUMPY_API
 * Returns 1 if the array object may be cast to the given data type using
 * the casting rule, 0 otherwise.  This differs from PyArray_CanCastTo in
 * that it handles scalar arrays (0 dimensions) specially, by checking
 * their value.
 */
NPY_NO_EXPORT npy_bool
PyArray_CanCastArrayTo(PyArrayObject *arr, PyArray_Descr *to,
                        NPY_CASTING casting)
{
    PyArray_Descr *from = PyArray_DESCR(arr);

    /* If it's a scalar, check the value */
    if (PyArray_NDIM(arr) == 0 && !PyArray_HASFIELDS(arr)) {
        return can_cast_scalar_to(from, PyArray_DATA(arr), to, casting);
    }

    /* Otherwise, use the standard rules */
    return PyArray_CanCastTypeTo(from, to, casting);
}


NPY_NO_EXPORT const char *
npy_casting_to_string(NPY_CASTING casting)
{
    switch (casting) {
        case NPY_NO_CASTING:
            return "'no'";
        case NPY_EQUIV_CASTING:
            return "'equiv'";
        case NPY_SAFE_CASTING:
            return "'safe'";
        case NPY_SAME_KIND_CASTING:
            return "'same_kind'";
        case NPY_UNSAFE_CASTING:
            return "'unsafe'";
        default:
            return "<unknown>";
    }
}


/**
 * Helper function to set a useful error when casting is not possible.
 *
 * @param src_dtype
 * @param dst_dtype
 * @param casting
 * @param scalar Whether this was a "scalar" cast (includes 0-D array with
 *               PyArray_CanCastArrayTo result).
 */
NPY_NO_EXPORT void
npy_set_invalid_cast_error(
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        NPY_CASTING casting, npy_bool scalar)
{
    char *msg;

    if (!scalar) {
        msg = "Cannot cast array data from %R to %R according to the rule %s";
    }
    else {
        msg = "Cannot cast scalar from %R to %R according to the rule %s";
    }
    PyErr_Format(PyExc_TypeError,
            msg, src_dtype, dst_dtype, npy_casting_to_string(casting));
}


/*NUMPY_API
 * See if array scalars can be cast.
 *
 * TODO: For NumPy 2.0, add a NPY_CASTING parameter.
 */
NPY_NO_EXPORT npy_bool
PyArray_CanCastScalar(PyTypeObject *from, PyTypeObject *to)
{
    int fromtype;
    int totype;

    fromtype = _typenum_fromtypeobj((PyObject *)from, 0);
    totype = _typenum_fromtypeobj((PyObject *)to, 0);
    if (fromtype == NPY_NOTYPE || totype == NPY_NOTYPE) {
        return NPY_FALSE;
    }
    return (npy_bool) PyArray_CanCastSafely(fromtype, totype);
}

/*
 * Internal promote types function which handles unsigned integers which
 * fit in same-sized signed integers specially.
 */
static PyArray_Descr *
promote_types(PyArray_Descr *type1, PyArray_Descr *type2,
                        int is_small_unsigned1, int is_small_unsigned2)
{
    if (is_small_unsigned1) {
        int type_num1 = type1->type_num;
        int type_num2 = type2->type_num;
        int ret_type_num;

        if (type_num2 < NPY_NTYPES && !(PyTypeNum_ISBOOL(type_num2) ||
                                        PyTypeNum_ISUNSIGNED(type_num2))) {
            /* Convert to the equivalent-sized signed integer */
            type_num1 = type_num_unsigned_to_signed(type_num1);

            ret_type_num = _npy_type_promotion_table[type_num1][type_num2];
            /* The table doesn't handle string/unicode/void, check the result */
            if (ret_type_num >= 0) {
                return PyArray_DescrFromType(ret_type_num);
            }
        }

        return PyArray_PromoteTypes(type1, type2);
    }
    else if (is_small_unsigned2) {
        int type_num1 = type1->type_num;
        int type_num2 = type2->type_num;
        int ret_type_num;

        if (type_num1 < NPY_NTYPES && !(PyTypeNum_ISBOOL(type_num1) ||
                                        PyTypeNum_ISUNSIGNED(type_num1))) {
            /* Convert to the equivalent-sized signed integer */
            type_num2 = type_num_unsigned_to_signed(type_num2);

            ret_type_num = _npy_type_promotion_table[type_num1][type_num2];
            /* The table doesn't handle string/unicode/void, check the result */
            if (ret_type_num >= 0) {
                return PyArray_DescrFromType(ret_type_num);
            }
        }

        return PyArray_PromoteTypes(type1, type2);
    }
    else {
        return PyArray_PromoteTypes(type1, type2);
    }

}

/*
 * Returns a new reference to type if it is already NBO, otherwise
 * returns a copy converted to NBO.
 */
NPY_NO_EXPORT PyArray_Descr *
ensure_dtype_nbo(PyArray_Descr *type)
{
    if (PyArray_ISNBO(type->byteorder)) {
        Py_INCREF(type);
        return type;
    }
    else {
        return PyArray_DescrNewByteorder(type, NPY_NATIVE);
    }
}


/**
 * This function should possibly become public API eventually.  At this
 * time it is implemented by falling back to `PyArray_AdaptFlexibleDType`.
 * We will use `CastingImpl[from, to].adjust_descriptors(...)` to implement
 * this logic.
 * Before that, the API needs to be reviewed though.
 *
 * WARNING: This function currently does not guarantee that `descr` can
 *          actually be cast to the given DType.
 *
 * @param descr The dtype instance to adapt "cast"
 * @param given_DType The DType class for which we wish to find an instance able
 *        to represent `descr`.
 * @returns Instance of `given_DType`. If `given_DType` is parametric the
 *          descr may be adapted to hold it.
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_CastDescrToDType(PyArray_Descr *descr, PyArray_DTypeMeta *given_DType)
{
    if (NPY_DTYPE(descr) == given_DType) {
        Py_INCREF(descr);
        return descr;
    }
    if (!given_DType->parametric) {
        /*
         * Don't actually do anything, the default is always the result
         * of any cast.
         */
        return given_DType->default_descr(given_DType);
    }
    if (PyObject_TypeCheck((PyObject *)descr, (PyTypeObject *)given_DType)) {
        Py_INCREF(descr);
        return descr;
    }

    if (!given_DType->legacy) {
        PyErr_SetString(PyExc_NotImplementedError,
                "Must use casting to find the correct DType for a parametric "
                "user DType. This is not yet implemented (this error should be "
                "unreachable).");
        return NULL;
    }

    PyArray_Descr *flex_dtype = PyArray_DescrNew(given_DType->singleton);
    return PyArray_AdaptFlexibleDType(descr, flex_dtype);
}


/**
 * This function defines the common DType operator.
 *
 * Note that the common DType will not be "object" (unless one of the dtypes
 * is object), even though object can technically represent all values
 * correctly.
 *
 * TODO: Before exposure, we should review the return value (e.g. no error
 *       when no common DType is found).
 *
 * @param dtype1 DType class to find the common type for.
 * @param dtype2 Second DType class.
 * @return The common DType or NULL with an error set
 */
NPY_NO_EXPORT PyArray_DTypeMeta *
PyArray_CommonDType(PyArray_DTypeMeta *dtype1, PyArray_DTypeMeta *dtype2)
{
    if (dtype1 == dtype2) {
        Py_INCREF(dtype1);
        return dtype1;
    }

    PyArray_DTypeMeta *common_dtype;

    common_dtype = dtype1->common_dtype(dtype1, dtype2);
    if (common_dtype == (PyArray_DTypeMeta *)Py_NotImplemented) {
        Py_DECREF(common_dtype);
        common_dtype = dtype2->common_dtype(dtype2, dtype1);
    }
    if (common_dtype == NULL) {
        return NULL;
    }
    if (common_dtype == (PyArray_DTypeMeta *)Py_NotImplemented) {
        Py_DECREF(Py_NotImplemented);
        PyErr_Format(PyExc_TypeError,
                "The DTypes %S and %S do not have a common DType. "
                "For example they cannot be stored in a single array unless "
                "the dtype is `object`.", dtype1, dtype2);
        return NULL;
    }
    return common_dtype;
}


/*NUMPY_API
 * Produces the smallest size and lowest kind type to which both
 * input types can be cast.
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_PromoteTypes(PyArray_Descr *type1, PyArray_Descr *type2)
{
    PyArray_DTypeMeta *common_dtype;
    PyArray_Descr *res;

    /* Fast path for identical inputs (NOTE: This path preserves metadata!) */
    if (type1 == type2 && PyArray_ISNBO(type1->byteorder)) {
        Py_INCREF(type1);
        return type1;
    }

    common_dtype = PyArray_CommonDType(NPY_DTYPE(type1), NPY_DTYPE(type2));
    if (common_dtype == NULL) {
        return NULL;
    }

    if (!common_dtype->parametric) {
        res = common_dtype->default_descr(common_dtype);
        Py_DECREF(common_dtype);
        return res;
    }

    /* Cast the input types to the common DType if necessary */
    type1 = PyArray_CastDescrToDType(type1, common_dtype);
    if (type1 == NULL) {
        Py_DECREF(common_dtype);
        return NULL;
    }
    type2 = PyArray_CastDescrToDType(type2, common_dtype);
    if (type2 == NULL) {
        Py_DECREF(type1);
        Py_DECREF(common_dtype);
        return NULL;
    }

    /*
     * And find the common instance of the two inputs
     * NOTE: Common instance preserves metadata (normally and of one input)
     */
    res = common_dtype->common_instance(type1, type2);
    Py_DECREF(type1);
    Py_DECREF(type2);
    Py_DECREF(common_dtype);
    return res;
}

/*
 * Produces the smallest size and lowest kind type to which all
 * input types can be cast.
 *
 * Equivalent to functools.reduce(PyArray_PromoteTypes, types)
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_PromoteTypeSequence(PyArray_Descr **types, npy_intp ntypes)
{
    npy_intp i;
    PyArray_Descr *ret = NULL;
    if (ntypes == 0) {
        PyErr_SetString(PyExc_TypeError, "at least one type needed to promote");
        return NULL;
    }
    ret = types[0];
    Py_INCREF(ret);
    for (i = 1; i < ntypes; ++i) {
        PyArray_Descr *tmp = PyArray_PromoteTypes(types[i], ret);
        Py_DECREF(ret);
        ret = tmp;
        if (ret == NULL) {
            return NULL;
        }
    }
    return ret;
}

/*
 * NOTE: While this is unlikely to be a performance problem, if
 *       it is it could be reverted to a simple positive/negative
 *       check as the previous system used.
 *
 * The is_small_unsigned output flag indicates whether it's an unsigned integer,
 * and would fit in a signed integer of the same bit size.
 */
static int min_scalar_type_num(char *valueptr, int type_num,
                                            int *is_small_unsigned)
{
    switch (type_num) {
        case NPY_BOOL: {
            return NPY_BOOL;
        }
        case NPY_UBYTE: {
            npy_ubyte value = *(npy_ubyte *)valueptr;
            if (value <= NPY_MAX_BYTE) {
                *is_small_unsigned = 1;
            }
            return NPY_UBYTE;
        }
        case NPY_BYTE: {
            npy_byte value = *(npy_byte *)valueptr;
            if (value >= 0) {
                *is_small_unsigned = 1;
                return NPY_UBYTE;
            }
            break;
        }
        case NPY_USHORT: {
            npy_ushort value = *(npy_ushort *)valueptr;
            if (value <= NPY_MAX_UBYTE) {
                if (value <= NPY_MAX_BYTE) {
                    *is_small_unsigned = 1;
                }
                return NPY_UBYTE;
            }

            if (value <= NPY_MAX_SHORT) {
                *is_small_unsigned = 1;
            }
            break;
        }
        case NPY_SHORT: {
            npy_short value = *(npy_short *)valueptr;
            if (value >= 0) {
                return min_scalar_type_num(valueptr, NPY_USHORT, is_small_unsigned);
            }
            else if (value >= NPY_MIN_BYTE) {
                return NPY_BYTE;
            }
            break;
        }
#if NPY_SIZEOF_LONG == NPY_SIZEOF_INT
        case NPY_ULONG:
#endif
        case NPY_UINT: {
            npy_uint value = *(npy_uint *)valueptr;
            if (value <= NPY_MAX_UBYTE) {
                if (value <= NPY_MAX_BYTE) {
                    *is_small_unsigned = 1;
                }
                return NPY_UBYTE;
            }
            else if (value <= NPY_MAX_USHORT) {
                if (value <= NPY_MAX_SHORT) {
                    *is_small_unsigned = 1;
                }
                return NPY_USHORT;
            }

            if (value <= NPY_MAX_INT) {
                *is_small_unsigned = 1;
            }
            break;
        }
#if NPY_SIZEOF_LONG == NPY_SIZEOF_INT
        case NPY_LONG:
#endif
        case NPY_INT: {
            npy_int value = *(npy_int *)valueptr;
            if (value >= 0) {
                return min_scalar_type_num(valueptr, NPY_UINT, is_small_unsigned);
            }
            else if (value >= NPY_MIN_BYTE) {
                return NPY_BYTE;
            }
            else if (value >= NPY_MIN_SHORT) {
                return NPY_SHORT;
            }
            break;
        }
#if NPY_SIZEOF_LONG != NPY_SIZEOF_INT && NPY_SIZEOF_LONG != NPY_SIZEOF_LONGLONG
        case NPY_ULONG: {
            npy_ulong value = *(npy_ulong *)valueptr;
            if (value <= NPY_MAX_UBYTE) {
                if (value <= NPY_MAX_BYTE) {
                    *is_small_unsigned = 1;
                }
                return NPY_UBYTE;
            }
            else if (value <= NPY_MAX_USHORT) {
                if (value <= NPY_MAX_SHORT) {
                    *is_small_unsigned = 1;
                }
                return NPY_USHORT;
            }
            else if (value <= NPY_MAX_UINT) {
                if (value <= NPY_MAX_INT) {
                    *is_small_unsigned = 1;
                }
                return NPY_UINT;
            }

            if (value <= NPY_MAX_LONG) {
                *is_small_unsigned = 1;
            }
            break;
        }
        case NPY_LONG: {
            npy_long value = *(npy_long *)valueptr;
            if (value >= 0) {
                return min_scalar_type_num(valueptr, NPY_ULONG, is_small_unsigned);
            }
            else if (value >= NPY_MIN_BYTE) {
                return NPY_BYTE;
            }
            else if (value >= NPY_MIN_SHORT) {
                return NPY_SHORT;
            }
            else if (value >= NPY_MIN_INT) {
                return NPY_INT;
            }
            break;
        }
#endif
#if NPY_SIZEOF_LONG == NPY_SIZEOF_LONGLONG
        case NPY_ULONG:
#endif
        case NPY_ULONGLONG: {
            npy_ulonglong value = *(npy_ulonglong *)valueptr;
            if (value <= NPY_MAX_UBYTE) {
                if (value <= NPY_MAX_BYTE) {
                    *is_small_unsigned = 1;
                }
                return NPY_UBYTE;
            }
            else if (value <= NPY_MAX_USHORT) {
                if (value <= NPY_MAX_SHORT) {
                    *is_small_unsigned = 1;
                }
                return NPY_USHORT;
            }
            else if (value <= NPY_MAX_UINT) {
                if (value <= NPY_MAX_INT) {
                    *is_small_unsigned = 1;
                }
                return NPY_UINT;
            }
#if NPY_SIZEOF_LONG != NPY_SIZEOF_INT && NPY_SIZEOF_LONG != NPY_SIZEOF_LONGLONG
            else if (value <= NPY_MAX_ULONG) {
                if (value <= NPY_MAX_LONG) {
                    *is_small_unsigned = 1;
                }
                return NPY_ULONG;
            }
#endif

            if (value <= NPY_MAX_LONGLONG) {
                *is_small_unsigned = 1;
            }
            break;
        }
#if NPY_SIZEOF_LONG == NPY_SIZEOF_LONGLONG
        case NPY_LONG:
#endif
        case NPY_LONGLONG: {
            npy_longlong value = *(npy_longlong *)valueptr;
            if (value >= 0) {
                return min_scalar_type_num(valueptr, NPY_ULONGLONG, is_small_unsigned);
            }
            else if (value >= NPY_MIN_BYTE) {
                return NPY_BYTE;
            }
            else if (value >= NPY_MIN_SHORT) {
                return NPY_SHORT;
            }
            else if (value >= NPY_MIN_INT) {
                return NPY_INT;
            }
#if NPY_SIZEOF_LONG != NPY_SIZEOF_INT && NPY_SIZEOF_LONG != NPY_SIZEOF_LONGLONG
            else if (value >= NPY_MIN_LONG) {
                return NPY_LONG;
            }
#endif
            break;
        }
        /*
         * Float types aren't allowed to be demoted to integer types,
         * but precision loss is allowed.
         */
        case NPY_HALF: {
            return NPY_HALF;
        }
        case NPY_FLOAT: {
            float value = *(float *)valueptr;
            if ((value > -65000 && value < 65000) || !npy_isfinite(value)) {
                return NPY_HALF;
            }
            break;
        }
        case NPY_DOUBLE: {
            double value = *(double *)valueptr;
            if ((value > -65000 && value < 65000) || !npy_isfinite(value)) {
                return NPY_HALF;
            }
            else if (value > -3.4e38 && value < 3.4e38) {
                return NPY_FLOAT;
            }
            break;
        }
        case NPY_LONGDOUBLE: {
            npy_longdouble value = *(npy_longdouble *)valueptr;
            if ((value > -65000 && value < 65000) || !npy_isfinite(value)) {
                return NPY_HALF;
            }
            else if (value > -3.4e38 && value < 3.4e38) {
                return NPY_FLOAT;
            }
            else if (value > -1.7e308 && value < 1.7e308) {
                return NPY_DOUBLE;
            }
            break;
        }
        /*
         * The code to demote complex to float is disabled for now,
         * as forcing complex by adding 0j is probably desirable.
         */
        case NPY_CFLOAT: {
            /*
            npy_cfloat value = *(npy_cfloat *)valueptr;
            if (value.imag == 0) {
                return min_scalar_type_num((char *)&value.real,
                                            NPY_FLOAT, is_small_unsigned);
            }
            */
            break;
        }
        case NPY_CDOUBLE: {
            npy_cdouble value = *(npy_cdouble *)valueptr;
            /*
            if (value.imag == 0) {
                return min_scalar_type_num((char *)&value.real,
                                            NPY_DOUBLE, is_small_unsigned);
            }
            */
            if (value.real > -3.4e38 && value.real < 3.4e38 &&
                     value.imag > -3.4e38 && value.imag < 3.4e38) {
                return NPY_CFLOAT;
            }
            break;
        }
        case NPY_CLONGDOUBLE: {
            npy_clongdouble value = *(npy_clongdouble *)valueptr;
            /*
            if (value.imag == 0) {
                return min_scalar_type_num((char *)&value.real,
                                            NPY_LONGDOUBLE, is_small_unsigned);
            }
            */
            if (value.real > -3.4e38 && value.real < 3.4e38 &&
                     value.imag > -3.4e38 && value.imag < 3.4e38) {
                return NPY_CFLOAT;
            }
            else if (value.real > -1.7e308 && value.real < 1.7e308 &&
                     value.imag > -1.7e308 && value.imag < 1.7e308) {
                return NPY_CDOUBLE;
            }
            break;
        }
    }

    return type_num;
}


NPY_NO_EXPORT PyArray_Descr *
PyArray_MinScalarType_internal(PyArrayObject *arr, int *is_small_unsigned)
{
    PyArray_Descr *dtype = PyArray_DESCR(arr);
    *is_small_unsigned = 0;
    /*
     * If the array isn't a numeric scalar, just return the array's dtype.
     */
    if (PyArray_NDIM(arr) > 0 || !PyTypeNum_ISNUMBER(dtype->type_num)) {
        Py_INCREF(dtype);
        return dtype;
    }
    else {
        char *data = PyArray_BYTES(arr);
        int swap = !PyArray_ISNBO(dtype->byteorder);
        /* An aligned memory buffer large enough to hold any type */
        npy_longlong value[4];
        dtype->f->copyswap(&value, data, swap, NULL);

        return PyArray_DescrFromType(
                        min_scalar_type_num((char *)&value,
                                dtype->type_num, is_small_unsigned));

    }
}

/*NUMPY_API
 * If arr is a scalar (has 0 dimensions) with a built-in number data type,
 * finds the smallest type size/kind which can still represent its data.
 * Otherwise, returns the array's data type.
 *
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_MinScalarType(PyArrayObject *arr)
{
    int is_small_unsigned;
    return PyArray_MinScalarType_internal(arr, &is_small_unsigned);
}

/*
 * Provides an ordering for the dtype 'kind' character codes, to help
 * determine when to use the min_scalar_type function. This groups
 * 'kind' into boolean, integer, floating point, and everything else.
 */
static int
dtype_kind_to_simplified_ordering(char kind)
{
    switch (kind) {
        /* Boolean kind */
        case 'b':
            return 0;
        /* Unsigned int kind */
        case 'u':
        /* Signed int kind */
        case 'i':
            return 1;
        /* Float kind */
        case 'f':
        /* Complex kind */
        case 'c':
            return 2;
        /* Anything else */
        default:
            return 3;
    }
}


/*
 * Determine if there is a mix of scalars and arrays/dtypes.
 * If this is the case, the scalars should be handled as the minimum type
 * capable of holding the value when the maximum "category" of the scalars
 * surpasses the maximum "category" of the arrays/dtypes.
 * If the scalars are of a lower or same category as the arrays, they may be
 * demoted to a lower type within their category (the lowest type they can
 * be cast to safely according to scalar casting rules).
 */
NPY_NO_EXPORT int
should_use_min_scalar(npy_intp narrs, PyArrayObject **arr,
                      npy_intp ndtypes, PyArray_Descr **dtypes)
{
    int use_min_scalar = 0;

    if (narrs > 0) {
        int all_scalars;
        int max_scalar_kind = -1;
        int max_array_kind = -1;

        all_scalars = (ndtypes > 0) ? 0 : 1;

        /* Compute the maximum "kinds" and whether everything is scalar */
        for (npy_intp i = 0; i < narrs; ++i) {
            if (PyArray_NDIM(arr[i]) == 0) {
                int kind = dtype_kind_to_simplified_ordering(
                                    PyArray_DESCR(arr[i])->kind);
                if (kind > max_scalar_kind) {
                    max_scalar_kind = kind;
                }
            }
            else {
                int kind = dtype_kind_to_simplified_ordering(
                                    PyArray_DESCR(arr[i])->kind);
                if (kind > max_array_kind) {
                    max_array_kind = kind;
                }
                all_scalars = 0;
            }
        }
        /*
         * If the max scalar kind is bigger than the max array kind,
         * finish computing the max array kind
         */
        for (npy_intp i = 0; i < ndtypes; ++i) {
            int kind = dtype_kind_to_simplified_ordering(dtypes[i]->kind);
            if (kind > max_array_kind) {
                max_array_kind = kind;
            }
        }

        /* Indicate whether to use the min_scalar_type function */
        if (!all_scalars && max_array_kind >= max_scalar_kind) {
            use_min_scalar = 1;
        }
    }
    return use_min_scalar;
}


/*NUMPY_API
 * Produces the result type of a bunch of inputs, using the UFunc
 * type promotion rules. Use this function when you have a set of
 * input arrays, and need to determine an output array dtype.
 *
 * If all the inputs are scalars (have 0 dimensions) or the maximum "kind"
 * of the scalars is greater than the maximum "kind" of the arrays, does
 * a regular type promotion.
 *
 * Otherwise, does a type promotion on the MinScalarType
 * of all the inputs.  Data types passed directly are treated as array
 * types.
 *
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_ResultType(npy_intp narrs, PyArrayObject **arr,
                    npy_intp ndtypes, PyArray_Descr **dtypes)
{
    npy_intp i;

    /* If there's just one type, pass it through */
    if (narrs + ndtypes == 1) {
        PyArray_Descr *ret = NULL;
        if (narrs == 1) {
            ret = PyArray_DESCR(arr[0]);
        }
        else {
            ret = dtypes[0];
        }
        Py_INCREF(ret);
        return ret;
    }

    int use_min_scalar = should_use_min_scalar(narrs, arr, ndtypes, dtypes);

    /* Loop through all the types, promoting them */
    if (!use_min_scalar) {
        PyArray_Descr *ret;

        /* Build a single array of all the dtypes */
        PyArray_Descr **all_dtypes = PyArray_malloc(
            sizeof(*all_dtypes) * (narrs + ndtypes));
        if (all_dtypes == NULL) {
            PyErr_NoMemory();
            return NULL;
        }
        for (i = 0; i < narrs; ++i) {
            all_dtypes[i] = PyArray_DESCR(arr[i]);
        }
        for (i = 0; i < ndtypes; ++i) {
            all_dtypes[narrs + i] = dtypes[i];
        }
        ret = PyArray_PromoteTypeSequence(all_dtypes, narrs + ndtypes);
        PyArray_free(all_dtypes);
        return ret;
    }
    else {
        int ret_is_small_unsigned = 0;
        PyArray_Descr *ret = NULL;

        for (i = 0; i < narrs; ++i) {
            int tmp_is_small_unsigned;
            PyArray_Descr *tmp = PyArray_MinScalarType_internal(
                arr[i], &tmp_is_small_unsigned);
            if (tmp == NULL) {
                Py_XDECREF(ret);
                return NULL;
            }
            /* Combine it with the existing type */
            if (ret == NULL) {
                ret = tmp;
                ret_is_small_unsigned = tmp_is_small_unsigned;
            }
            else {
                PyArray_Descr *tmpret = promote_types(
                    tmp, ret, tmp_is_small_unsigned, ret_is_small_unsigned);
                Py_DECREF(tmp);
                Py_DECREF(ret);
                ret = tmpret;
                if (ret == NULL) {
                    return NULL;
                }

                ret_is_small_unsigned = tmp_is_small_unsigned &&
                                        ret_is_small_unsigned;
            }
        }

        for (i = 0; i < ndtypes; ++i) {
            PyArray_Descr *tmp = dtypes[i];
            /* Combine it with the existing type */
            if (ret == NULL) {
                ret = tmp;
                Py_INCREF(ret);
            }
            else {
                PyArray_Descr *tmpret = promote_types(
                    tmp, ret, 0, ret_is_small_unsigned);
                Py_DECREF(ret);
                ret = tmpret;
                if (ret == NULL) {
                    return NULL;
                }
            }
        }
        /* None of the above loops ran */
        if (ret == NULL) {
            PyErr_SetString(PyExc_TypeError,
                    "no arrays or types available to calculate result type");
        }

        return ret;
    }
}

/*NUMPY_API
 * Is the typenum valid?
 */
NPY_NO_EXPORT int
PyArray_ValidType(int type)
{
    PyArray_Descr *descr;
    int res=NPY_TRUE;

    descr = PyArray_DescrFromType(type);
    if (descr == NULL) {
        res = NPY_FALSE;
    }
    Py_DECREF(descr);
    return res;
}

/* Backward compatibility only */
/* In both Zero and One

***You must free the memory once you are done with it
using PyDataMem_FREE(ptr) or you create a memory leak***

If arr is an Object array you are getting a
BORROWED reference to Zero or One.
Do not DECREF.
Please INCREF if you will be hanging on to it.

The memory for the ptr still must be freed in any case;
*/

static int
_check_object_rec(PyArray_Descr *descr)
{
    if (PyDataType_HASFIELDS(descr) && PyDataType_REFCHK(descr)) {
        PyErr_SetString(PyExc_TypeError, "Not supported for this data-type.");
        return -1;
    }
    return 0;
}

/*NUMPY_API
  Get pointer to zero of correct type for array.
*/
NPY_NO_EXPORT char *
PyArray_Zero(PyArrayObject *arr)
{
    char *zeroval;
    int ret, storeflags;
    static PyObject * zero_obj = NULL;

    if (_check_object_rec(PyArray_DESCR(arr)) < 0) {
        return NULL;
    }
    zeroval = PyDataMem_NEW(PyArray_DESCR(arr)->elsize);
    if (zeroval == NULL) {
        PyErr_SetNone(PyExc_MemoryError);
        return NULL;
    }

    if (zero_obj == NULL) {
        zero_obj = PyLong_FromLong((long) 0);
        if (zero_obj == NULL) {
            return NULL;
        }
    }
    if (PyArray_ISOBJECT(arr)) {
        /* XXX this is dangerous, the caller probably is not
           aware that zeroval is actually a static PyObject*
           In the best case they will only use it as-is, but
           if they simply memcpy it into a ndarray without using
           setitem(), refcount errors will occur
        */
        memcpy(zeroval, &zero_obj, sizeof(PyObject *));
        return zeroval;
    }
    storeflags = PyArray_FLAGS(arr);
    PyArray_ENABLEFLAGS(arr, NPY_ARRAY_BEHAVED);
    ret = PyArray_SETITEM(arr, zeroval, zero_obj);
    ((PyArrayObject_fields *)arr)->flags = storeflags;
    if (ret < 0) {
        PyDataMem_FREE(zeroval);
        return NULL;
    }
    return zeroval;
}

/*NUMPY_API
  Get pointer to one of correct type for array
*/
NPY_NO_EXPORT char *
PyArray_One(PyArrayObject *arr)
{
    char *oneval;
    int ret, storeflags;
    static PyObject * one_obj = NULL;

    if (_check_object_rec(PyArray_DESCR(arr)) < 0) {
        return NULL;
    }
    oneval = PyDataMem_NEW(PyArray_DESCR(arr)->elsize);
    if (oneval == NULL) {
        PyErr_SetNone(PyExc_MemoryError);
        return NULL;
    }

    if (one_obj == NULL) {
        one_obj = PyLong_FromLong((long) 1);
        if (one_obj == NULL) {
            return NULL;
        }
    }
    if (PyArray_ISOBJECT(arr)) {
        /* XXX this is dangerous, the caller probably is not
           aware that oneval is actually a static PyObject*
           In the best case they will only use it as-is, but
           if they simply memcpy it into a ndarray without using
           setitem(), refcount errors will occur
        */
        memcpy(oneval, &one_obj, sizeof(PyObject *));
        return oneval;
    }

    storeflags = PyArray_FLAGS(arr);
    PyArray_ENABLEFLAGS(arr, NPY_ARRAY_BEHAVED);
    ret = PyArray_SETITEM(arr, oneval, one_obj);
    ((PyArrayObject_fields *)arr)->flags = storeflags;
    if (ret < 0) {
        PyDataMem_FREE(oneval);
        return NULL;
    }
    return oneval;
}

/* End deprecated */

/*NUMPY_API
 * Return the typecode of the array a Python object would be converted to
 *
 * Returns the type number the result should have, or NPY_NOTYPE on error.
 */
NPY_NO_EXPORT int
PyArray_ObjectType(PyObject *op, int minimum_type)
{
    PyArray_Descr *dtype = NULL;
    int ret;

    if (minimum_type != NPY_NOTYPE && minimum_type >= 0) {
        dtype = PyArray_DescrFromType(minimum_type);
        if (dtype == NULL) {
            return NPY_NOTYPE;
        }
    }
    if (PyArray_DTypeFromObject(op, NPY_MAXDIMS, &dtype) < 0) {
        return NPY_NOTYPE;
    }

    if (dtype == NULL) {
        ret = NPY_DEFAULT_TYPE;
    }
    else if (!NPY_DTYPE(dtype)->legacy) {
        /*
         * TODO: If we keep all type number style API working, by defining
         *       type numbers always. We may be able to allow this again.
         */
        PyErr_Format(PyExc_TypeError,
                "This function currently only supports native NumPy dtypes "
                "and old-style user dtypes, but the dtype was %S.\n"
                "(The function may need to be updated to support arbitrary"
                "user dtypes.)",
                dtype);
        ret = NPY_NOTYPE;
    }
    else {
        ret = dtype->type_num;
    }

    Py_XDECREF(dtype);

    return ret;
}

/* Raises error when len(op) == 0 */

/*NUMPY_API
 *
 * This function is only used in one place within NumPy and should
 * generally be avoided. It is provided mainly for backward compatibility.
 *
 * The user of the function has to free the returned array.
 */
NPY_NO_EXPORT PyArrayObject **
PyArray_ConvertToCommonType(PyObject *op, int *retn)
{
    int i, n;
    PyArray_Descr *common_descr = NULL;
    PyArrayObject **mps = NULL;

    *retn = n = PySequence_Length(op);
    if (n == 0) {
        PyErr_SetString(PyExc_ValueError, "0-length sequence.");
    }
    if (PyErr_Occurred()) {
        *retn = 0;
        return NULL;
    }
    mps = (PyArrayObject **)PyDataMem_NEW(n*sizeof(PyArrayObject *));
    if (mps == NULL) {
        *retn = 0;
        return (void*)PyErr_NoMemory();
    }

    if (PyArray_Check(op)) {
        for (i = 0; i < n; i++) {
            mps[i] = (PyArrayObject *) array_item_asarray((PyArrayObject *)op, i);
        }
        if (!PyArray_ISCARRAY((PyArrayObject *)op)) {
            for (i = 0; i < n; i++) {
                PyObject *obj;
                obj = PyArray_NewCopy(mps[i], NPY_CORDER);
                Py_DECREF(mps[i]);
                mps[i] = (PyArrayObject *)obj;
            }
        }
        return mps;
    }

    for (i = 0; i < n; i++) {
        mps[i] = NULL;
    }

    for (i = 0; i < n; i++) {
        /* Convert everything to an array, this could be optimized away */
        PyObject *tmp = PySequence_GetItem(op, i);
        if (tmp == NULL) {
            goto fail;
        }

        mps[i] = (PyArrayObject *)PyArray_FROM_O(tmp);
        Py_DECREF(tmp);
        if (mps[i] == NULL) {
            goto fail;
        }
    }

    common_descr = PyArray_ResultType(n, mps, 0, NULL);
    if (common_descr == NULL) {
        goto fail;
    }

    /* Make sure all arrays are contiguous and have the correct dtype. */
    for (i = 0; i < n; i++) {
        int flags = NPY_ARRAY_CARRAY;
        PyArrayObject *tmp = mps[i];

        Py_INCREF(common_descr);
        mps[i] = (PyArrayObject *)PyArray_FromArray(tmp, common_descr, flags);
        Py_DECREF(tmp);
        if (mps[i] == NULL) {
            goto fail;
        }
    }
    Py_DECREF(common_descr);
    return mps;

 fail:
    Py_XDECREF(common_descr);
    *retn = 0;
    for (i = 0; i < n; i++) {
        Py_XDECREF(mps[i]);
    }
    PyDataMem_FREE(mps);
    return NULL;
}
