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

inline double sum(xt::pyarray<double> &m)
{
    py::gil_scoped_release release;
    return std::accumulate(m.begin(), m.end(), 0.0);
}


inline void _count(xt::pytensor<double,1> &grid, xt::pytensor<double,1> &x, double xmin, double xmax)
{
    py::gil_scoped_release release;
    double scale = 1/(xmax - xmin);
    double vmin = xmin;
    const int shape0 = grid.shape()[0];
    for (const double& x_value : x) {
        double scaled = (x_value - vmin) * scale;
        if( (scaled >= 0) & (scaled < 1) ) {
            int index = (int)(scaled * shape0);
            grid(index) += 1;
        }
    }
}


inline xt::pyarray<double> count(xt::pytensor<double,1> &x, double xmin, double xmax, int shape)
{
    xt::pyarray<double> grid(shape);
    // here it should do multithreading (simple map-reduce algo)
    //_count(grid, x, xmin, xmax);
    return grid;

}

inline void _sum(xt::pytensor<double,1> &grid, xt::pytensor<double,1> &values, xt::pytensor<double,1> &x, double xmin, double xmax)
{
    py::gil_scoped_release release;
    double scale = 1/(xmax - xmin);
    double vmin = xmin;
    const int shape0 = grid.shape()[0];
    long long int length = values.shape()[0];
    //for (const double& x_value : x) {
    for(long long int i = 0; i < length; i++) {
        double x_value = x[i];
        double scaled = (x_value - vmin) * scale;
        if( (scaled >= 0) & (scaled < 1) ) {
            int index = (int)(scaled * shape0);
            grid(index) += values[i];
        }
    }
}


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

    //def("count", count, "counts values on a regular N-d grid (multithreaded)");
    m.def("_count", _count, "implementation of count1d, but single threaded, and does not allocate a grid");
    //m.def("sum", sum, "calculates sum values on a regular N-d grid (multithreaded)");
    m.def("_sum", _sum, "implementation of sum , but single threaded, and does not allocate a grid");
    return m.ptr();
}
