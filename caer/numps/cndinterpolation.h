

#ifndef NI_INTERPOLATION_H
#define NI_INTERPOLATION_H

int NI_SplineFilter1D(PyArrayObject*, int, int, NI_ExtendMode, PyArrayObject*);
int NI_GeometricTransform(PyArrayObject*, int (*)(npy_intp*, double*, int, int,
                                                    void*), void*, PyArrayObject*, PyArrayObject*,
                                                    PyArrayObject*, PyArrayObject*, int, int,
                                                    double);
int NI_ZoomShift(PyArrayObject*, PyArrayObject*, PyArrayObject*,
                                 PyArrayObject*, int, int, double);

#endif
