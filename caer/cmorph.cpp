#include <algorithm>
#include <queue>
#include <vector>
#include <cstdio>
#include <limits>
#include <iostream>

#include "numps/array.hpp"
#include "numps/dispatch.hpp"

#include "utils.hpp"
#include "filters.h"


namespace {

    const char TypeErrorMsg[] =
        "Type not understood. "
        "This is caused by either a direct call to morph (which is dangerous: types are not checked!) or a bug in caer.\n";

    // Using std::abs fails on Windows as it is not defined for all integer types:
    template <typename T>
    inline T t_abs(T val) { return (val >= 0 ? val : -val); }

    template <typename T>
    std::vector<numpy::position> neighbours(const numpy::aligned_array<T>& Bc, bool include_centre = false) {
        numpy::position centre = central_position(Bc);
        const unsigned N = Bc.size();
        typename numpy::aligned_array<T>::const_iterator startc = Bc.begin();
        std::vector<numpy::position> res;
        for (unsigned i = 0; i != N; ++i, ++startc) {
            if (!*startc) continue;
            if (startc.position() != centre || include_centre) {
                res.push_back(startc.position() - centre);
            }
        }
        return res;
    }


    template <typename T>
    std::vector<numpy::position> neighbours_delta(const numpy::aligned_array<T>& Bc, bool include_centre = false) {
        std::vector<numpy::position> rs = neighbours(Bc, include_centre);
        numpy::position accumulated = rs[0];
        for (unsigned i = 1; i < rs.size(); ++i) {
            rs[i] -= accumulated;
            accumulated += rs[i];
        }
        return rs;
    }

    template<typename T>
    numpy::index_type margin_of(const numpy::position& position, const numpy::array_base<T>& ref) {
        numpy::index_type margin = std::numeric_limits<numpy::index_type>::max();
        const numpy::index_type nd = ref.ndims();
        for (numpy::index_type d = 0; d != nd; ++d) {
            if (position[d] < margin) margin = position[d];
            const numpy::index_type rmargin = ref.dim(d) - position[d] - 1;
            if (rmargin < margin) margin = rmargin;
        }
    return margin;
    }

    template<typename T>
    T erode_sub(T a, T b) {
        if (b == std::numeric_limits<T>::min()) return std::numeric_limits<T>::max();
        if (!std::numeric_limits<T>::is_signed && (b > a)) return T(0);
        const T r = a - b;
        if ( std::numeric_limits<T>::is_signed && (r > a)) return std::numeric_limits<T>::min();
        return r;
    }

    template<>
    bool erode_sub<bool>(bool a, bool b) {
        return a && b;
    }

    template<typename T> bool is_bool(T) { 
        return false; 
    }
    template<> bool is_bool<bool>(bool) { 
        return true; 
    }

    template<typename T>
    void erode(numpy::aligned_array<T> res, const numpy::aligned_array<T> array, const numpy::aligned_array<T> Bc) {
        gil_release nogil;
        const numpy::index_type N = res.size();
        typename numpy::aligned_array<T>::const_iterator iter = array.begin();
        filter_iterator<T> filter(array.raw_array(), Bc.raw_array(), ExtendNearest, is_bool(T()));
        const numpy::index_type N2 = filter.size();
        T* rpos = res.data();
        if (!N2) return;

        for (numpy::index_type i = 0; i != N; ++i, ++rpos, filter.iterate_both(iter)) {
            T value = std::numeric_limits<T>::max();
            for (numpy::index_type j = 0; j != N2; ++j) {
                T arr_val = T();
                filter.retrieve(iter, j, arr_val);
                value = std::min<T>(value, erode_sub(arr_val, filter[j]));
                if (value == std::numeric_limits<T>::min()) break;
            }
            *rpos = value;
        }
    }

    void fast_binary_dilate_erode_2d(numpy::aligned_array<bool> res, const numpy::aligned_array<bool> array, numpy::aligned_array<bool> Bc, const bool is_erosion) {
        assert(res.is_carray());
        assert(array.is_carray());
        assert(res.ndim() == 2);

        const numpy::index_type Ny = array.dim(0);
        const numpy::index_type Nx = array.dim(1);
        const numpy::index_type N = Ny * Nx;

        const numpy::index_type By = Bc.dim(0);
        const numpy::index_type Bx = Bc.dim(1);

        const numpy::index_type Cy = By/2;
        const numpy::index_type Cx = Bx/2;

        std::vector<numpy::index_type> positions;
        for (numpy::index_type y = 0; y != By; ++y) {
            for (numpy::index_type x = 0; x != Bx; ++x) {
                if (!Bc.at(y,x)) continue;
                const numpy::index_type dy = y-Cy;
                const numpy::index_type dx = x-Cx;
                if (t_abs(dy) >= Ny || t_abs(dx) >= Nx) continue;
                if (dy || dx) {
                    positions.push_back(is_erosion ? dy: -dy);
                    positions.push_back(is_erosion ? dx: -dx);
                }
            }
        }
        const numpy::index_type N2 = positions.size()/2;
        if (Bc.at(Cy,Cx)) std::copy(array.data(), array.data() + N, res.data());
        else std::fill_n(res.data(), N, is_erosion);
        if (positions.empty()) return;

        for (numpy::index_type y = 0; y != Ny; ++y) {
            bool* const orow = res.data(y);
            for (numpy::index_type j = 0; j != N2; ++j) {
                numpy::index_type dy = positions[2*j];
                numpy::index_type dx = positions[2*j + 1];
                assert(dx || dy);
                if ((y + dy) < 0) dy = -y;
                if ((y + dy) >= Ny) {
                    dy = -y+(Ny-1);
                }
                bool* out = orow;
                const bool* in = array.data(y + dy);
                numpy::index_type n = Nx - t_abs(dx);
                if (dx > 0) {
                    for (numpy::index_type i = 0; i != (dx-1); ++i) {
                        if (is_erosion) {
                            out[Nx-i-1] &= in[Nx-1];
                        } else {
                            out[Nx-i-1] |= in[Nx-1];
                        }
                    }
                    in += dx;
                } else if (dx < 0) {
                    for (numpy::index_type i = 0; i != -dx; ++i) {
                        if (is_erosion) {
                            out[i] &= in[0];
                        } else {
                            out[i] |= in[0];
                        }
                    }
                    out += -dx;
                }
                if (is_erosion) {
                    for (numpy::index_type i = 0; i != n; ++i) *out++ &= *in++;
                } else {
                    for (numpy::index_type i = 0; i != n; ++i) *out++ |= *in++;
                }
            }
        }
    }


    PyObject* py_erode(PyObject* self, PyObject* args) {
        PyArrayObject* array;
        PyArrayObject* Bc;
        PyArrayObject* output;
        if (!PyArg_ParseTuple(args, "OOO", &array, &Bc, &output)) return NULL;
        if (!numpy::are_arrays(array, Bc, output) || !numpy::same_shape(array, output) ||
            !numpy::equiv_typenums(array, Bc, output) ||
            PyArray_NDIM(array) != PyArray_NDIM(Bc)
        ) {
            PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
            return NULL;
        }
        holdref r_o(output);

        if (numpy::check_type<bool>(array) && PyArray_NDIM(array) == 2 && PyArray_ISCARRAY(array)) {
            fast_binary_dilate_erode_2d(numpy::aligned_array<bool>(output), numpy::aligned_array<bool>(array), numpy::aligned_array<bool>(Bc), true);
        } else {
    #define HANDLE(type) \
        erode<type>(numpy::aligned_array<type>(output), numpy::aligned_array<type>(array), numpy::aligned_array<type>(Bc));
        SAFE_SWITCH_ON_INTEGER_TYPES_OF(array);
    #undef HANDLE
        }

        Py_XINCREF(output);
        return PyArray_Return(output);
    }


    template <typename T>
    T dilate_add(T a, T b) {
        if (a == std::numeric_limits<T>::min()) return a;
        if (b == std::numeric_limits<T>::min()) return b;
        const T r = a + b;
        // if overflow, saturate
        if (r < std::max<T>(a,b)) return std::numeric_limits<T>::max();
        return r;
    }

    template<>
    bool dilate_add(bool a, bool b) {
        return a && b;
    }

    template<typename T>
    void dilate(numpy::aligned_array<T> res, const numpy::array<T> array, const numpy::aligned_array<T> Bc) {
        gil_release nogil;
        const numpy::index_type N = res.size();
        typename numpy::array<T>::const_iterator iter = array.begin();
        filter_iterator<T> filter(res.raw_array(), Bc.raw_array(), ExtendNearest, is_bool(T()));
        const numpy::index_type N2 = filter.size();
        // T* is a fine iterator type.
        T* rpos = res.data();
        std::fill(rpos, rpos + res.size(), std::numeric_limits<T>::min());
        if (!N2) return;

        for (numpy::index_type i = 0; i != N; ++i, ++rpos, filter.iterate_both(iter)) {
            const T value = *iter;
            if (value == std::numeric_limits<T>::min()) continue;
            for (numpy::index_type j = 0; j != N2; ++j) {
                const T nval = dilate_add(value, filter[j]);
                T arr_val = T();
                filter.retrieve(rpos, j, arr_val);
                if (nval > arr_val) filter.set(rpos, j, nval);
            }
        }
    }


    PyObject* py_dilate(PyObject* self, PyObject* args) {
        PyArrayObject* array;
        PyArrayObject* Bc;
        PyArrayObject* output;
        if (!PyArg_ParseTuple(args,"OOO", &array, &Bc, &output)) return NULL;
        if (!numpy::are_arrays(array, Bc, output) ||
            !numpy::same_shape(array, output) ||
            !numpy::equiv_typenums(array, Bc, output) ||
            PyArray_NDIM(array) != PyArray_NDIM(Bc)
        ) {
            PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
            return NULL;
        }
        holdref r_o(output);
        if (numpy::check_type<bool>(array) && PyArray_NDIM(array) == 2 && PyArray_ISCARRAY(array)) {
            fast_binary_dilate_erode_2d(numpy::aligned_array<bool>(output), numpy::aligned_array<bool>(array), numpy::aligned_array<bool>(Bc), false);
        } else {
    #define HANDLE(type) \
            dilate<type>(numpy::aligned_array<type>(output),numpy::array<type>(array),numpy::aligned_array<type>(Bc));
            SAFE_SWITCH_ON_INTEGER_TYPES_OF(array);
    #undef HANDLE
        }

        Py_XINCREF(output);
        return PyArray_Return(output);
    }

    template <typename CostType>
    struct MarkerInfo {
        CostType cost;
        npy_intp idx;
        npy_intp position;
        npy_intp margin;
        MarkerInfo(CostType cost, npy_intp idx, npy_intp position, npy_intp margin)
            :cost(cost)
            ,idx(idx)
            ,position(position)
            ,margin(margin) { }
            
        bool operator < (const MarkerInfo& other) const {
            // We want the smaller argument to compare higher, so we reverse the order here:
            if (cost == other.cost) return idx > other.idx;
            return cost > other.cost;
        }
    };

    struct NeighbourElem {
        // This represents a neighbour
        //  - delta: how to get from the current element to the next by pointer manipulation
        //  - step: how large the distance to the centre is
        //  - delta_position: the difference as a numpy::position
        NeighbourElem(npy_intp delta, npy_intp step, const numpy::position& delta_position)
            :delta(delta)
            ,step(step)
            ,delta_position(delta_position)
            { }
        npy_intp delta;
        npy_intp step;
        numpy::position delta_position;
    };

    template<typename BaseType>
    void cwatershed(numpy::aligned_array<npy_int64> res,
                            numpy::aligned_array<bool>* lines,
                            const numpy::aligned_array<BaseType> array,
                            const numpy::aligned_array<npy_int64> markers,
                            const numpy::aligned_array<BaseType> Bc) {
        gil_release nogil;
        const npy_intp N = res.size();
        const npy_intp N2 = Bc.size();
        assert(res.is_carray());
        npy_int64* rdata = res.data();
        std::vector<NeighbourElem> neighbours;
        const numpy::position centre = central_position(Bc);
        typename numpy::aligned_array<BaseType>::const_iterator Bi = Bc.begin();
        for (npy_intp j = 0; j != N2; ++j, ++Bi) {
            if (*Bi) {
                numpy::position npos = Bi.position() - centre;
                npy_intp margin = 0;
                for (numpy::index_type d = 0; d != Bc.ndims(); ++d) {
                    margin = std::max<npy_intp>(t_abs(npy_intp(npos[d])), margin);
                }
                npy_intp delta = markers.pos_to_flat(npos);
                if (!delta) continue;
                neighbours.push_back(NeighbourElem(delta, margin, npos));
            }
        }
        npy_intp idx = 0;

        enum { white, grey, black };
        std::vector<unsigned char> status(array.size(), static_cast<unsigned char>(white));

        std::priority_queue<MarkerInfo<BaseType> > hqueue;

        typename numpy::aligned_array<npy_int64>::const_iterator miter = markers.begin();
        for (numpy::index_type i = 0; i != N; ++i, ++miter) {
            if (*miter) {
                assert(markers.validposition(miter.position()));
                const numpy::position mpos = miter.position();
                const npy_intp margin = margin_of(mpos, markers);

                hqueue.push(MarkerInfo<BaseType>(array.at(mpos), idx++, markers.pos_to_flat(mpos), margin));
                res.at(mpos) = *miter;
                status[markers.pos_to_flat(mpos)] = grey;
            }
        }

        while (!hqueue.empty()) {
            const MarkerInfo<BaseType> next = hqueue.top();
            hqueue.pop();
            status[next.position] = black;
            npy_intp margin = next.margin;
            for (typename std::vector<NeighbourElem>::const_iterator neighbour = neighbours.begin(),
                                past = neighbours.end();
                        neighbour != past;
                        ++neighbour) {
                const numpy::index_type npos = next.position + neighbour->delta;
                numpy::index_type nmargin = margin - neighbour->step;
                if (nmargin < 0) {
                    // nmargin is a lower bound on the margin, so we must recompute the actual value
                    numpy::position pos = markers.flat_to_pos(next.position);
                    assert(markers.validposition(pos));
                    numpy::position long_pos = pos + neighbour->delta_position;
                    nmargin = margin_of(long_pos, markers);
                    if (nmargin < 0) {
                        // We are outside the image.
                        continue;
                    }
                    // we are good with the recomputed margin
                    assert(markers.validposition(long_pos));
                    // Update lower bound
                    if ((nmargin - neighbour->step) > margin) margin = nmargin - neighbour->step;
                }
                assert(npos < npy_intp(status.size()));
                switch (status[npos]) {
                    case white: {
                        const BaseType ncost = array.at_flat(npos);
                        rdata[npos] = rdata[next.position];
                        hqueue.push(MarkerInfo<BaseType>(ncost, idx++, npos, nmargin));
                        status[npos] = grey;
                        break;
                    }
                    case grey: {
                        if (lines && rdata[next.position] != rdata[npos]) {
                            lines->at_flat(npos) = true;
                        }
                        break;
                    }
                }
            }
        }
    }

    PyObject* py_cwatershed(PyObject* self, PyObject* args) {
        PyArrayObject* array;
        PyArrayObject* markers;
        PyArrayObject* Bc;
        int return_lines;
        if (!PyArg_ParseTuple(args,"OOOi", &array, &markers, &Bc,&return_lines)) {
            return NULL;
        }
        if (!numpy::are_arrays(array, markers, Bc) ||
            !numpy::check_type<npy_int64>(markers)) {
            PyErr_SetString(PyExc_RuntimeError, "caer._cwatershed: markers should be an int32 array.");
            return NULL;
        }
        PyArrayObject* res_a = (PyArrayObject*)PyArray_SimpleNew(
                                                        PyArray_NDIM(array),
                                                        PyArray_DIMS(array),
                                                        NPY_INT64);
        if (!res_a) return NULL;
        PyArrayObject* lines =  0;
        numpy::aligned_array<bool>* lines_a = 0;
        if (return_lines) {
            lines = (PyArrayObject*)PyArray_SimpleNew(PyArray_NDIM(array), PyArray_DIMS(array), NPY_BOOL);
            if (!lines) return NULL;
            lines_a = new numpy::aligned_array<bool>(lines);
        }
    #define HANDLE(type) \
        cwatershed<type>(numpy::aligned_array<npy_int64>(res_a),lines_a,numpy::aligned_array<type>(array),numpy::aligned_array<npy_int64>(markers),numpy::aligned_array<type>(Bc));
        SAFE_SWITCH_ON_TYPES_OF(array);
    #undef HANDLE
        if (return_lines) {
            delete lines_a;
            PyObject* ret_val = PyTuple_New(2);
            PyTuple_SetItem(ret_val,0,(PyObject*)res_a);
            PyTuple_SetItem(ret_val,1,(PyObject*)lines);
            return ret_val;
        }
        return PyArray_Return(res_a);
    }


    double compute_euc2_dist(const numpy:: position& a, const numpy::position b) {
        assert(a.ndim() == b.ndim());
        double r = 0.;
        const numpy::index_type n = a.ndim();
        for (numpy::index_type i = 0; i != n; ++i) {
            r += (a[i]-b[i])*(a[i]-b[i]);
        }
        return r;
    }


    // Arguably the function distance_multi should have been in file distance.cpp,
    // but it uses several functions from this module like neighbours().
    template<typename BaseType>
    void distance_multi(numpy::aligned_array<BaseType> res,
                            const numpy::aligned_array<bool> array,
                            const numpy::aligned_array<bool> Bc) {
        gil_release nogil;
        const numpy::index_type N = res.size();
        const std::vector<numpy::position> Bcs = neighbours_delta(Bc);
        const numpy::index_type N2 = Bcs.size();

        typename numpy::aligned_array<bool>::const_iterator aiter = array.begin();
        typename numpy::aligned_array<BaseType>::iterator riter = res.begin();

        numpy::position_queue cur_q(res.ndim());
        numpy::position_queue orig_q(res.ndim());
        std::queue<double> dist_q;
        for (numpy::index_type i = 0; i != N; ++i, ++riter, ++aiter) {
            if (!*aiter) {
                *riter = 0;
                const numpy::position p = aiter.position();
                numpy::position next = p;
                for (numpy::index_type j = 0; j != N2; ++j) {
                    next += Bcs[j];
                    if (array.validposition(next) && array.at(next)) {
                        const double dist = compute_euc2_dist(next, p);
                        BaseType* rpos = res.data(next);
                        if (*rpos > dist) {
                            *rpos = dist;

                            cur_q.push(next);
                            orig_q.push(p);
                            dist_q.push(dist);
                        }
                    }
                }
            }
        }

        while (!dist_q.empty()) {
            numpy::position next = cur_q.top_pop();
            const numpy::position orig = orig_q.top_pop();
            const BaseType dist = dist_q.front();
            dist_q.pop();

            assert(dist == compute_euc2_dist(next, orig));

            if (res.at(next) < dist) continue;
            for (numpy::index_type j = 0; j != N2; ++j) {
                next += Bcs[j];
                if (array.validposition(next)) {
                    const double next_dist = compute_euc2_dist(next, orig);
                    BaseType* rpos = res.data(next);
                    if (*rpos > next_dist) {
                        *rpos = next_dist;
                        cur_q.push(next);
                        orig_q.push(orig);
                        dist_q.push(next_dist);
                    }
                }
            }
        }
    }

    PyObject* py_distance_multi(PyObject* self, PyObject* args) {
        PyArrayObject* array;
        PyArrayObject* res;
        PyArrayObject* Bc;
        if (!PyArg_ParseTuple(args,"OOO", &res, &array, &Bc)) {
            return NULL;
        }
        if (!numpy::are_arrays(array, res, Bc) ||
            !numpy::check_type<bool>(array) ||
            !numpy::check_type<bool>(Bc) ||
            !numpy::same_shape(array, res)
            ) {
            PyErr_SetString(PyExc_RuntimeError, "caer._distance_multi: res and input array should have same shape. input & Bc arrays maust be boolean arrays.");
            return NULL;
        }
    #define HANDLE(type) \
        distance_multi<type>(numpy::aligned_array<type>(res), \
                            numpy::aligned_array<bool>(array),\
                            numpy::aligned_array<bool>(Bc));

        SAFE_SWITCH_ON_TYPES_OF(res);
    #undef HANDLE

        Py_RETURN_NONE;
    }

    struct HitMissNeighbour {
        HitMissNeighbour(numpy::index_type delta, int value)
            :delta(delta)
            ,value(value)
            { }
        numpy::index_type delta;
        int value;
    };

    template <typename T>
    void hitmiss(numpy::aligned_array<T> res, const numpy::aligned_array<T>& input, const numpy::aligned_array<T>& Bc) {
        gil_release nogil;
        typedef typename numpy::aligned_array<T>::const_iterator const_iterator;
        const numpy::index_type N = input.size();
        const numpy::index_type N2 = Bc.size();
        const numpy::position centre = central_position(Bc);
        numpy::index_type Bc_margin = 0;
        for (numpy::index_type d = 0; d != Bc.ndims(); ++d) {
            numpy::index_type cmargin = Bc.dim(d)/2;
            if (cmargin > Bc_margin) Bc_margin = cmargin;
        }

        std::vector<HitMissNeighbour> neighbours;
        const_iterator Bi = Bc.begin();
        for (numpy::index_type j = 0; j != N2; ++j, ++Bi) {
            if (*Bi != 2) {
                numpy::position npos = Bi.position() - centre;
                numpy::index_type delta = input.pos_to_flat(npos);
                neighbours.push_back(HitMissNeighbour(delta, *Bi));
            }
        }

        // This is a micro-optimisation for templates with structure
        // It makes it more likely that matching will fail earlier
        // in uniform regions than otherwise would be the case.
        std::random_shuffle(neighbours.begin(), neighbours.end());
        numpy::index_type slack = 0;
        for (npy_intp i = 0; i != N; ++i) {
            while (!slack) {
                numpy::position cur = input.flat_to_pos(i);
                bool moved = false;
                for (numpy::index_type d = 0; d != input.ndims(); ++d) {
                    numpy::index_type margin = std::min<numpy::index_type>(cur[d], input.dim(d) - cur[d] - 1);
                    if (margin < Bc.dim(d)/2) {
                        numpy::index_type size = 1;
                        for (numpy::index_type dd = d+1; dd < input.ndims(); ++dd) size *= input.dim(dd);
                        for (numpy::index_type j = 0; j != size; ++j) {
                            res.at_flat(i++) = 0;
                            if (i == N) return;
                        }
                        moved = true;
                        break;
                    }
                }
                if (!moved) slack = input.dim(input.ndims() - 1) - Bc.dim(input.ndims() - 1) + 1;
            }
            --slack;
            T value = 1;
            for (std::vector<HitMissNeighbour>::const_iterator neighbour = neighbours.begin(), past = neighbours.end();
                neighbour != past;
                ++neighbour) {
                if (input.at_flat(i + neighbour->delta) != static_cast<T>(neighbour->value)) {
                    value = 0;
                    break;
                }
            }
            res.at_flat(i) = value;
        }
    }

    PyObject* py_hitmiss(PyObject* self, PyObject* args) {
        PyArrayObject* array;
        PyArrayObject* Bc;
        PyArrayObject* res_a;
        if (!PyArg_ParseTuple(args, "OOO", &array, &Bc, &res_a)) {
            PyErr_SetString(PyExc_RuntimeError,TypeErrorMsg);
            return NULL;
        }
        holdref r(res_a);

    #define HANDLE(type) \
        hitmiss<type>(numpy::aligned_array<type>(res_a), numpy::aligned_array<type>(array), numpy::aligned_array<type>(Bc));
        SAFE_SWITCH_ON_INTEGER_TYPES_OF(array);
    #undef HANDLE

        Py_INCREF(res_a);
        return PyArray_Return(res_a);
    }


    PyMethodDef methods[] = {
    {"subm",(PyCFunction)py_subm, METH_VARARGS, NULL},
    {"dilate",(PyCFunction)py_dilate, METH_VARARGS, NULL},
    {"disk_2d",(PyCFunction)py_disk_2d, METH_VARARGS, NULL},
    {"erode",(PyCFunction)py_erode, METH_VARARGS, NULL},
    {"close_holes",(PyCFunction)py_close_holes, METH_VARARGS, NULL},
    {"cwatershed",(PyCFunction)py_cwatershed, METH_VARARGS, NULL},
    {"distance_multi",(PyCFunction)py_distance_multi, METH_VARARGS, NULL},
    {"locmin_max",(PyCFunction)py_locminmax, METH_VARARGS, NULL},
    {"regmin_max",(PyCFunction)py_regminmax, METH_VARARGS, NULL},
    {"hitmiss",(PyCFunction)py_hitmiss, METH_VARARGS, NULL},
    {NULL, NULL,0,NULL},
    };

} // namespace

DECLARE_MODULE(cmorph)