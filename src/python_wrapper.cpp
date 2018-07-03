#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <boost/python.hpp>
#include <numpy/arrayobject.h>

#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#else
void omp_set_num_threads(int) {}
int omp_get_max_threads() { return 1; }
#endif

#include "tsne.h"

namespace bp = boost::python;

void check_array(PyArrayObject* arr)
{
    if (!(PyArray_FLAGS(arr) & NPY_ARRAY_C_CONTIGUOUS))
        throw std::runtime_error("Passed array must be contiguous.");
    if (PyArray_NDIM(arr) != 2)
        throw std::runtime_error("Passed array must be 2D.");
    if (PyArray_TYPE(arr) != NPY_FLOAT64)
        throw std::runtime_error("Passed array must be of type double64.");
}

std::vector<npy_intp> get_dims(PyArrayObject* arr)
{
    std::vector<npy_intp> dims(2);
    dims[0] = PyArray_DIMS(arr)[0];
    dims[1] = PyArray_DIMS(arr)[1];
    return dims;
}

std::vector<double> np_to_std_vec(bp::object obj, std::vector<npy_intp>& dims)
{
    auto arr_obj = reinterpret_cast<PyArrayObject*>(obj.ptr());
    check_array(arr_obj);
    dims = get_dims(arr_obj);
    auto arr_value_ptr = static_cast<double*>(PyArray_DATA(arr_obj));
    return std::vector<double>(arr_value_ptr, arr_value_ptr + dims[0] * dims[1]);
}

void write_results(PyObject* obj, const std::vector<double>& res,
    const std::vector<npy_intp>& dims)
{
    auto arr_obj = reinterpret_cast<PyArrayObject*>(obj);
    check_array(arr_obj);
    auto arr_value_ptr = static_cast<double*>(PyArray_DATA(arr_obj));

    for (int i = 0; i < static_cast<int>(res.size()); i++)
        *(arr_value_ptr++) = res[i];
}

PyObject* tsneRun(bp::object obj, int no_dims, double perplexity, double theta,
    double learning_rate, int max_iter, int stop_lying_iter, int mom_switch_iter,
    std::string dist_measure, int num_threads)
{
    if (num_threads > 0)
        omp_set_num_threads(num_threads);
    else
        omp_set_num_threads(omp_get_max_threads());

    std::vector<npy_intp> dims;
    auto vec = np_to_std_vec(obj, dims);

    auto Y = std::vector<double>(dims[0] * no_dims);

    if (dist_measure != "euclidean" && dist_measure != "jaccard")
        throw std::runtime_error("Unknown distance measure in cpp tsne module");

    tsne::Distance dist = (dist_measure == "euclidean") ? tsne::Distance::Euclidean
        : tsne::Distance::Jaccard;
    tsne::TSNE::run(vec, dims[0], dims[1], Y, no_dims, perplexity, theta, learning_rate,
        false, max_iter, stop_lying_iter, mom_switch_iter, dist);

    dims[1] = no_dims;
    PyObject* out_array = PyArray_SimpleNew(2, dims.data(), NPY_FLOAT64);
    write_results(out_array, Y, dims);
    return out_array;
}

BOOST_PYTHON_MODULE(tsne_module)
{
    using namespace boost::python;
    def("run", tsneRun, bp::default_call_policies(),
        (bp::arg("obj"), bp::arg("no_dims") = 2, bp::arg("perplexity") = 30.0,
            bp::arg("theta") = 0.5, bp::arg("learning_rate") = 200.0,
            bp::arg("max_iter") = 1000, bp::arg("stop_lying_iter") = 250,
            bp::arg("mom_switch_iter") = 250, bp::arg("dist_measure") = "euclidean",
            bp::arg("num_threads") = 0));

    import_array(); // needed for PyArray_SimpleNew
}
