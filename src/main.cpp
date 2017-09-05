#include "pybind11/pybind11.h"

#include "xtensor/xmath.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xview.hpp"

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
    //py::gil_scoped_release release;
    return std::accumulate(m.begin(), m.end(), 0.0);
}

template<typename E_GRID, typename X_TYPE>
//inline void _countX(xt::xexpression<E_GRID>& grid_expression, const int i, const xt::pytensor<X_TYPE,1> &x, const double xmin, const double xmax)
inline void  __attribute__((always_inline)) _countX(E_GRID& grid, const int i, const xt::pytensor<X_TYPE,1> &x, const double& xmin, const double& xmax, const int& shape0)
{
    //E_GRID& grid = grid_expression.derived_cast();
    //const double scale = 1/(xmax - xmin);
    //const double vmin = xmin;
    //const int shape0 = grid.shape()[0];
    const double x_value = x[i];
    const double scaled = (x_value - xmin) * xmax;
    if( (scaled >= 0) & (scaled < 1) ) {
        int index = (int)(scaled * shape0);
        grid(index) += 1;
    }
}


template<typename E_GRID, typename X_TYPE, typename...TAIL>
//inline void _countX(xt::xexpression<E_GRID>& grid_expression, const int i, const xt::pytensor<X_TYPE,1> &x, const double xmin, const double xmax, TAIL... tail)
Cinline void  __attribute__((always_inline)) _countX(E_GRID& grid, const int i, const xt::pytensor<X_TYPE,1> &x, const double& xmin, const double& xmax, const int& shape0, TAIL... tail)
{
    //E_GRID& grid = grid_expression.derived_cast();
    //const double scale = 1/(xmax - xmin);
    //const double vmin = xmin;
    //const int shape0 = grid.shape()[0];
    const double x_value = x[i];
    const double scaled = (x_value - xmin) * xmax;
    if( (scaled >= 0) & (scaled < 1) ) {
        const int index = (int)(scaled * shape0);
        auto&& subgrid = xt::view(grid, index);
        _countX(subgrid, i, tail...);
    }
}

inline void _count2(xt::pytensor<double,2> &grid, const xt::pytensor<double,1> &x, const double xmin, const double xmax, const xt::pytensor<double,1> &y, const double ymin, const double ymax)
{
    py::gil_scoped_release release;
    const int length = x.shape()[0];
    const double scaley = 1/(ymax-ymin);
    const double scalex = 1/(xmax-xmin);
    const int shapex = grid.shape()[0];
    const int shapey = grid.shape()[1];
    //*
    for(int i = 0 ; i < length; i++)
        _countX(grid, i, x, xmin, scalex, shapex, y, ymin, scaley, shapey);
    /*/
    const int shapex = grid.shape()[0];
    const int shapey = grid.shape()[1];
    for(int i = 0 ; i < length; i++) {
        const double x_value = x[i];
        const double scaledx = (x_value - xmin) * scalex;
        if( (scaledx >= 0) & (scaledx < 1) ) {
            int indexx = (int)(scaledx * shapex);
            const double y_value = y[i];
            const double scaledy = (y_value - ymin) * scaley;
            auto subgrid = xt::view(grid, indexx);
            if( (scaledy >= 0) & (scaledy < 1) ) {
                int indexy = (int)(scaledy * shapey);
                subgrid(indexy) += 1;
            }
        }
    }

   /**/
}

inline void _count(xt::pytensor<double,1> &grid, xt::pytensor<double,1> &x, double xmin, double xmax)
{
    py::gil_scoped_release release;
    int length = x.shape()[0];
    for(int i = 0 ; i < length; i++)
        _countX(grid, i, x, xmin, 1/(xmax-xmin), grid.shape()[0]);
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
    m.def("_count2", _count2, "implementation of count2d, but single threaded, and does not allocate a grid");
    //m.def("sum", sum, "calculates sum values on a regular N-d grid (multithreaded)");
    m.def("_sum", _sum, "implementation of sum , but single threaded, and does not allocate a grid");
    return m.ptr();
}
