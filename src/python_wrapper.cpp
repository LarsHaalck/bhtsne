#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include <algorithm>
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
namespace np = boost::python::numpy;

void check_array(const np::ndarray& array)
{
    if (!(array.get_flags() & np::ndarray::C_CONTIGUOUS))
        throw std::runtime_error("Passed array must be contiguous.");
    if (array.get_nd() != 2)
        throw std::runtime_error("Passed array must be 2D.");
    if (array.get_dtype() != np::dtype::get_builtin<double>())
        throw std::runtime_error("Passed array must be of type float64 (double).");
}

std::vector<int> get_dims(const np::ndarray& array)
{
    std::vector<int> dims(2);
    dims[0] = array.shape(0);
    dims[1] = array.shape(1);
    return dims;
}

std::vector<double> np_to_std_vec(const np::ndarray& array, std::vector<int>& dims)
{
    check_array(array);
    dims = get_dims(array);
    auto arr_value_ptr = reinterpret_cast<double*>(array.get_data());
    return std::vector<double>(arr_value_ptr, arr_value_ptr + dims[0] * dims[1]);
}


np::ndarray tsneRun(bp::object obj, int no_dims, double perplexity, double theta,
    double learning_rate, int max_iter, int stop_lying_iter, int mom_switch_iter,
    std::string dist_measure, int num_threads)
{
    if (num_threads > 0)
        omp_set_num_threads(num_threads);
    else
        omp_set_num_threads(omp_get_max_threads());

    auto numpy_array = np::from_object(obj, np::ndarray::C_CONTIGUOUS);
    std::vector<int> dims;
    auto vec = np_to_std_vec(numpy_array, dims);

    auto Y = std::vector<double>(dims[0] * no_dims);

    if (dist_measure != "euclidean" && dist_measure != "jaccard")
        throw std::runtime_error("Unknown distance measure in cpp tsne module");

    tsne::Distance dist = (dist_measure == "euclidean") ? tsne::Distance::Euclidean
        : tsne::Distance::Jaccard;
    tsne::TSNE::run(vec, dims[0], dims[1], Y, no_dims, perplexity, theta, learning_rate,
        false, max_iter, stop_lying_iter, mom_switch_iter, dist);

    // out_array is an ndarray of size dims[0] * no_dims
    auto shape = bp::make_tuple(dims[0] * no_dims);
    auto dtype = np::dtype::get_builtin<double>();
    auto out_array = np::empty(shape, dtype);

    std::copy(std::begin(Y), std::end(Y), reinterpret_cast<double*>(out_array.get_data()));
    return out_array.reshape(bp::make_tuple(dims[0], no_dims));
}

BOOST_PYTHON_MODULE(tsne_module)
{
    Py_Initialize();
    np::initialize();

    using namespace boost::python;
    def("run", tsneRun, bp::default_call_policies(),
        (bp::arg("obj"), bp::arg("no_dims") = 2, bp::arg("perplexity") = 30.0,
            bp::arg("theta") = 0.5, bp::arg("learning_rate") = 200.0,
            bp::arg("max_iter") = 1000, bp::arg("stop_lying_iter") = 250,
            bp::arg("mom_switch_iter") = 250, bp::arg("dist_measure") = "euclidean",
            bp::arg("num_threads") = 0));
}
