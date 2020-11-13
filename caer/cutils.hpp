/*
   _____           ______  _____ 
 / ____/    /\    |  ____ |  __ \
| |        /  \   | |__   | |__) | Caer - Modern Computer Vision
| |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
| |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
 \_____\/_/    \_ \______ |_|__\_

Licensed under the MIT License <http://opensource.org/licenses/MIT>
SPDX-License-Identifier: MIT
Copyright (c) 2020 The Caer Authors <http://github.com/jasmcaus>
*/


#include <Python.h>
#include <numpy/ndarrayobject.h>


// holdref is a RAII object for decreasing a reference at scope exit
struct holdref {
    holdref(PyObject* obj, bool incref=true)
        :obj(obj) {
        if (incref) { Py_XINCREF(obj); }
    }
    holdref(PyArrayObject* obj, bool incref=true)
        :obj((PyObject*)obj) {
        if (incref) { Py_XINCREF(obj); }
    }
    ~holdref() { Py_XDECREF(obj); }

private:
    PyObject* const obj;
};

// gil_release is a sort of reverse RAII object: it acquires the GIL on scope exit
struct gil_release {
    gil_release() {
        _save = PyEval_SaveThread();
        active_ = true;
    }
    ~gil_release() {
        if (active_) restore();
    }
    void restore() {
        PyEval_RestoreThread(_save);
        active_ = false;
    }
    PyThreadState *_save;
    bool active_;
};


// This encapsulates the arguments to PyErr_SetString
// The reason that it doesn't call PyErr_SetString directly is that we wish
// that this be throw-able in an environment where the thread might not own the
// GIL as long as it is caught when the GIL is held.
struct PythonException {
    PythonException(PyObject *type, const char *message)
        :type_(type)
        ,message_(message)
        { }

    PyObject* type() const { return type_; }
    const char* message() const { return message_; }

    PyObject* const type_;
    const char* const message_;
};



// DECLARE_MODULE is slightly ugly, but it encapsulates the differences in
// initializing a module between Python 2.x & Python 3.x

#if PY_MAJOR_VERSION < 3
#define DECLARE_MODULE(name) \
extern "C" \
void init##name () { \
    import_array(); \
    (void)Py_InitModule(#name, methods); \
}

#else

#define DECLARE_MODULE(name) \
namespace { \
    struct PyModuleDef moduledef = { \
        PyModuleDef_HEAD_INIT, \
        #name, \
        NULL, \
        -1, \
        methods, \
        NULL, \
        NULL, \
        NULL, \
        NULL \
    }; \
} \
PyMODINIT_FUNC \
PyInit_##name () { \
    import_array(); \
    return PyModule_Create(&moduledef); \
}


#endif