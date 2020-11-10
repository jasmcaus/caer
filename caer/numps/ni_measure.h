

#ifndef NI_MEASURE_H
#define NI_MEASURE_H

int NI_FindObjects(PyArrayObject*, npy_intp, npy_intp*);

int NI_WatershedIFT(PyArrayObject*, PyArrayObject*, PyArrayObject*,
                    PyArrayObject*);

#endif
