#include "pybind11/pybind11.h"

#include "xtensor/xmath.hpp"
#include "xtensor/xarray.hpp"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyvectorize.hpp"

#include <iostream>
#include <numeric>
#include <cmath>

namespace py = pybind11;

// Examples

inline double example1(xt::pyarray<double> &m)
{
    return m(0);
}

inline xt::pyarray<double> example2(xt::pyarray<double> &m)
{
    return m + 2;
}

// Readme Examples

inline double readme_example1(xt::pyarray<double> &m)
{
    auto sines = xt::sin(m);
    return std::accumulate(sines.begin(), sines.end(), 0.0);
}

inline double sum(xt::pyarray<double> &m)
{
    py::gil_scoped_release release;
    return std::accumulate(m.begin(), m.end(), 0.0);
}

inline double bin1d(xt::pytensor<double,1> &grid, xt::pytensor<double,1> &x, double xmin, double xmax)
{
    py::gil_scoped_release release;
    double scale = 1/(xmax - xmin);
    double vmin = xmin;
    const int shape0 = grid.shape()[0];
    for (const double& value : x) {
        double scaled = (value - vmin) * scale;
        if( (scaled >= 0) & (scaled < 1) ) {
            int index = (int)(scaled * shape0);
            grid(index) += 1;
        }
    }
}

// Vectorize Examples

inline double scalar_func(double i, double j)
{
    return std::sin(i) + std::cos(j);
}

// Python Module and Docstrings

PYBIND11_PLUGIN(statx)
{
    xt::import_numpy();

    py::module m("statx", R"docu(
        Statistics on Nd-gritrds/tensors

        .. currentmodule:: statx

        .. autosummary::
           :toctree: _generate

           example1
           example2
           readme_example1
           vectorize_example1
    )docu");

    m.def("sum", sum, "sums");
    m.def("bin1d", bin1d, "bin1d");
    m.def("example1", example1, "Return the first element of an array, of dimension at least one");
    m.def("example2", example2, "Return the the specified array plus 2");

    m.def("readme_example1", readme_example1, "Accumulate the sines of all the values of the specified array");

    m.def("vectorize_example1", xt::pyvectorize(scalar_func), "Add the sine and and cosine of the two specified values");

    return m.ptr();
}
