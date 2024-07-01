#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

// using namespace Eigen;
namespace py = pybind11;
// using EigenDStride = Stride<Eigen::Dynamic, Eigen::Dynamic>;
// template <typename MatrixType>
// using EigenDRef = Ref<MatrixType, 0, EigenDStride>; //allows passing column/row order matrices easily

//forward declare all bindings
void binding_outside_points_from_rasterization(py::module& m);
void binding_outside_points_from_rejection_sampling(py::module& m);
void binding_locally_make_feasible(py::module& m);
void binding_fine_tune_point_cloud_iter(py::module& m);
void binding_point_cloud_to_mesh(py::module& m);

PYBIND11_MODULE(rfta_bindings, m) {

    /// call all bindings declared above  
    binding_outside_points_from_rasterization(m);
    binding_locally_make_feasible(m);
    binding_fine_tune_point_cloud_iter(m);
    binding_point_cloud_to_mesh(m);

    // remove this later
    binding_outside_points_from_rejection_sampling(m);

    m.def("help", [&]() {printf("hi"); });
}