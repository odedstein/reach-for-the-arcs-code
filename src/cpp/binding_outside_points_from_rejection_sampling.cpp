#include "outside_points_from_rejection_sampling.h"
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <string>
#include <iostream>

using namespace Eigen;
namespace py = pybind11;
using EigenDStride = Stride<Eigen::Dynamic, Eigen::Dynamic>;
template <typename MatrixType>
using EigenDRef = Ref<MatrixType, 0, EigenDStride>; //allows passing column/row order matrices easily

void binding_outside_points_from_rejection_sampling(py::module& m) {
    m.def("_outside_points_from_rejection_sampling_cpp_impl",[](EigenDRef<MatrixXd> _sdf_points,
                         EigenDRef<MatrixXd> _sphere_radii,
                         int rng_seed,
                         int num_samples)
        {
            Eigen::MatrixXd sdf_points(_sdf_points), sphere_radii(_sphere_radii);
            Eigen::MatrixXd outside_points;
            const int dim = sdf_points.cols();
            if(dim==2) {
                outside_points_from_rejection_sampling<2>(sdf_points, sphere_radii,
                    rng_seed, num_samples,
                    outside_points);
            } else if(dim==3) {
                outside_points_from_rejection_sampling<3>(sdf_points, sphere_radii,
                    rng_seed, num_samples,
                    outside_points);
            }
            return outside_points;
        });
    
}
