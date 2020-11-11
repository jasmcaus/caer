#ifndef _NPY_ARRAY_CONVERT_DATATYPE_H_
#define _NPY_ARRAY_CONVERT_DATATYPE_H_

NPY_NO_EXPORT PyArray_VectorUnaryFunc *
PyArray_GetCastFunc(PyArray_Descr *descr, int type_num);

NPY_NO_EXPORT int
PyArray_ObjectType(PyObject *op, int minimum_type);

NPY_NO_EXPORT PyArrayObject **
PyArray_ConvertToCommonType(PyObject *op, int *retn);

NPY_NO_EXPORT PyArray_DTypeMeta *
PyArray_CommonDType(PyArray_DTypeMeta *dtype1, PyArray_DTypeMeta *dtype2);

NPY_NO_EXPORT int
PyArray_ValidType(int type);

/* Like PyArray_CanCastArrayTo */
NPY_NO_EXPORT npy_bool
can_cast_scalar_to(PyArray_Descr *scal_type, char *scal_data,
                    PyArray_Descr *to, NPY_CASTING casting);

NPY_NO_EXPORT PyArray_Descr *
ensure_dtype_nbo(PyArray_Descr *type);

NPY_NO_EXPORT int
should_use_min_scalar(npy_intp narrs, PyArrayObject **arr,
                      npy_intp ndtypes, PyArray_Descr **dtypes);

NPY_NO_EXPORT const char *
npy_casting_to_string(NPY_CASTING casting);

NPY_NO_EXPORT void
npy_set_invalid_cast_error(
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        NPY_CASTING casting, npy_bool scalar);

/*
 * This function calls Py_DECREF on flex_dtype, and replaces it with
 * a new dtype that has been adapted based on the values in data_dtype
 * and data_obj. If the flex_dtype is not flexible, it returns it as-is.
 *
 * Usually, if data_obj is not an array, dtype should be the result
 * given by the PyArray_GetArrayParamsFromObject function.
 *
 * The data_obj may be NULL if just a dtype is known for the source.
 *
 * If *flex_dtype is NULL, returns immediately, without setting an
 * exception, leaving any previous error handling intact.
 *
 * The current flexible dtypes include NPY_STRING, NPY_UNICODE, NPY_VOID,
 * and NPY_DATETIME with generic units.
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_AdaptFlexibleDType(PyArray_Descr *data_dtype, PyArray_Descr *flex_dtype);

NPY_NO_EXPORT PyArray_Descr *
PyArray_CastDescrToDType(PyArray_Descr *descr, PyArray_DTypeMeta *given_DType);

#endif
