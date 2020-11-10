

#ifndef NI_MORPHOLOGY_H
#define NI_MORPHOLOGY_H

int NI_BinaryErosion(PyArrayObject*, PyArrayObject*, PyArrayObject*,
         PyArrayObject*, int, npy_intp*, int, int, int*, NI_CoordinateList**);
int NI_BinaryErosion2(PyArrayObject*, PyArrayObject*, PyArrayObject*,
                      int, npy_intp*, int, NI_CoordinateList**);
int NI_DistanceTransformBruteForce(PyArrayObject*, int, PyArrayObject*,
                                                                     PyArrayObject*, PyArrayObject*);
int NI_DistanceTransformOnePass(PyArrayObject*, PyArrayObject *,
                                                                PyArrayObject*);
int NI_EuclideanFeatureTransform(PyArrayObject*, PyArrayObject*,
                                                                 PyArrayObject*);

#endif
