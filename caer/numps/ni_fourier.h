

#ifndef NI_FOURIER_H
#define NI_FOURIER_H

int NI_FourierFilter(PyArrayObject*, PyArrayObject*, npy_intp, int,
                                         PyArrayObject*, int);
int NI_FourierShift(PyArrayObject*, PyArrayObject*, npy_intp, int,
                                        PyArrayObject*);

#endif
