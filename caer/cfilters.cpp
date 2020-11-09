// Copyright (C) 2003-2005 Peter J. Verveer
// Copyright (C) 2010-2013 Luis Pedro Coelho

#define NO_IMPORT_ARRAY

#include <cassert>
#include <memory>

#include "filters.h"
#include "utils.hpp"

// Calculate the offsets to the filter points, for all border regions and the interior of the array:
int init_filter_offsets(PyArrayObject *array, bool *footprint,
         const npy_intp * const fshape, npy_intp* origins,
         const ExtendMode mode, std::vector<npy_intp>& offsets,
         std::vector<npy_intp>* coordinate_offsets) {

    npy_intp coordinates[NPY_MAXDIMS], position[NPY_MAXDIMS];
    npy_intp forigins[NPY_MAXDIMS];
    const int rank = PyArray_NDIM(array);
    const npy_intp* const ashape = PyArray_DIMS(array);

    npy_intp astrides[NPY_MAXDIMS];
    for (int d = 0; d != rank; ++d) 
        astrides[d] = PyArray_STRIDE(array, d)/PyArray_ITEMSIZE(array);

    // calculate how many sets of offsets must be stored:
    npy_intp offsets_size = 1;
    for(int ii = 0; ii < rank; ii++)
        offsets_size *= (ashape[ii] < fshape[ii] ? ashape[ii] : fshape[ii]);

    // the size of the footprint array:
    npy_intp filter_size = 1;
    for(int i = 0; i < rank; ++i) 
        filter_size *= fshape[i];

    // calculate the number of non-zero elements in the footprint:
    npy_intp footprint_size = 0;
    if (footprint) {
        for(int i = 0; i < filter_size; ++i) footprint_size += footprint[i];
    } else {
        footprint_size = filter_size;
    }

    if (int(mode) < 0 || int(mode) > ExtendLast) {
        throw PythonException(PyExc_RuntimeError, "boundary mode not supported");
    }
    offsets.resize(offsets_size * footprint_size);
    if (coordinate_offsets) coordinate_offsets->resize(offsets_size * footprint_size);
    // from here on, we cannot fail anymore:

    for(int ii = 0; ii < rank; ii++) {
        forigins[ii] = fshape[ii]/2 + (origins ? *origins++ : 0);
    }

    std::fill(coordinates, coordinates + rank, 0);
    std::fill(position, position + rank, 0);


    // calculate all possible offsets to elements in the filter kernel, for all regions in the array(interior // and border regions): 

    unsigned poi = 0;
    npy_intp* pc = coordinate_offsets ? &(*coordinate_offsets)[0] : 0;

    // iterate over all regions: CAER
    for(int ll = 0; ll < offsets_size; ll++) {
        // iterate over the elements in the footprint array: CAER
        for(int kk = 0; kk < filter_size; kk++) {
            npy_intp offset = 0;
            // only calculate an offset if the footprint is 1: CAER
            if (!footprint || footprint[kk]) {
                // find offsets along all axes: CAER
                for(int ii = 0; ii < rank; ii++) {
                    const npy_intp orgn = forigins[ii];
                    npy_intp cc = coordinates[ii] - orgn + position[ii];
                    cc = fix_offset(mode, cc, ashape[ii]);

                    // calculate offset along current axis: CAER
                    if (cc == border_flag_value) {
                        // just flag that we are outside the border CAER
                        offset = border_flag_value;
                        if (coordinate_offsets)
                            pc[ii] = 0;
                        break;
                    } else {
                        // use an offset that is possibly mapped from outside the border:
                        cc -= position[ii];
                        offset += astrides[ii] * cc;
                        if (coordinate_offsets)
                            pc[ii] = cc;
                    }
                }

                // store the offset CAER
                offsets[poi++] = offset;
                if (coordinate_offsets)
                    pc += rank;
            }

            // next point in the filter: CAER
            for(int ii = rank - 1; ii >= 0; ii--) {
                if (coordinates[ii] < fshape[ii] - 1) {
                    coordinates[ii]++;
                    break;
                } else {
                    coordinates[ii] = 0;
                }
            }
        }

        // move to the next array region: CAER
        for(int ii = rank - 1; ii >= 0; ii--) {
            const int orgn = forigins[ii];
            if (position[ii] == orgn) {
                position[ii] += ashape[ii] - fshape[ii] + 1;
                if (position[ii] <= orgn)
                    position[ii] = orgn + 1;
            } else {
                position[ii]++;
            }
            if (position[ii] < ashape[ii]) {
                break;
            } else {
                position[ii] = 0;
            }
        }
    }
    assert(poi <= offsets.size());

    return footprint_size;
}


void init_filter_iterator(const int rank, const npy_intp *fshape,
                    const npy_intp filter_size, const npy_intp *ashape,
                    const npy_intp *origins,
                    npy_intp* strides, npy_intp* backstrides,
                    npy_intp* minbound, npy_intp* maxbound)
{
    // calculate the strides, used to move the offsets pointer through the offsets table: CAER
    if (rank > 0) {
        strides[rank - 1] = filter_size;
        for(int ii = rank - 2; ii >= 0; ii--) {
            const npy_intp step = ashape[ii + 1] < fshape[ii + 1] ? ashape[ii + 1] : fshape[ii + 1];
            strides[ii] = strides[ii + 1] * step;
        }
    }
    
    for(int ii = 0; ii < rank; ii++) {
        const npy_intp step = ashape[ii] < fshape[ii] ? ashape[ii] : fshape[ii];
        const npy_intp orgn = fshape[ii]/2 + (origins ? *origins++ : 0);
        // stride for stepping back to previous offsets: CAER
        backstrides[ii] = (step - 1) * strides[ii];
        // initialize boundary extension sizes: CAER
        minbound[ii] = orgn;
        maxbound[ii] = ashape[ii] - fshape[ii] + orgn;
    }

    std::reverse(strides, strides + rank);
    std::reverse(backstrides, backstrides + rank);
    std::reverse(minbound, minbound + rank);
    std::reverse(maxbound, maxbound + rank);
}