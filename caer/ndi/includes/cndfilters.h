

#ifndef NI_FILTERS_H
#define NI_FILTERS_H

int NI_Correlate1D(PyArrayObject*, PyArrayObject*, int, PyArrayObject*,
                   NI_ExtendMode, double, npy_intp);
int NI_Correlate(PyArrayObject*, PyArrayObject*, PyArrayObject*,
                 NI_ExtendMode, double, npy_intp*);
int NI_UniformFilter1D(PyArrayObject*, npy_intp, int, PyArrayObject*,
                       NI_ExtendMode, double, npy_intp);
int NI_MinOrMaxFilter1D(PyArrayObject*, npy_intp, int, PyArrayObject*,
                        NI_ExtendMode, double, npy_intp, int);
int NI_MinOrMaxFilter(PyArrayObject*, PyArrayObject*, PyArrayObject*,
                      PyArrayObject*, NI_ExtendMode, double, npy_intp*,
                                            int);
int NI_RankFilter(PyArrayObject*, int, PyArrayObject*, PyArrayObject*,
                                    NI_ExtendMode, double, npy_intp*);
int NI_GenericFilter1D(PyArrayObject*, int (*)(double*, npy_intp,
                       double*, npy_intp, void*), void*, npy_intp, int,
                       PyArrayObject*, NI_ExtendMode, double, npy_intp);
int NI_GenericFilter(PyArrayObject*, int (*)(double*, npy_intp, double*,
                                         void*), void*, PyArrayObject*, PyArrayObject*,
                     NI_ExtendMode, double, npy_intp*);
#endif
