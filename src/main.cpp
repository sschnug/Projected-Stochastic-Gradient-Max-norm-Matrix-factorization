#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Eigen>
#include "PSG.hpp"

typedef Eigen::Array<int, Eigen::Dynamic,1> ArrayXi;
typedef Eigen::Array<float, Eigen::Dynamic,1> ArrayXf;
typedef Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> ArrayXXf;

namespace py = pybind11;

PYBIND11_MODULE(py_mf_maxnorm, m)
{
  m.doc() = "PSG";

  py::class_<PSG>(m, "PSG")
  .def(py::init<int, int, int, int, float, int, float,
                ArrayXi&, ArrayXi&, ArrayXf&>())
  .def("optimize", &PSG::optimize, "optimize")
  .def("get_L", &PSG::get_L, "get_L")
  .def("get_R", &PSG::get_R, "get_R")
  .def("set_testset", &PSG::set_testset, "set_testset")
  .def("get_train_metrics_history", &PSG::get_train_metrics_history, "get_train_metrics_history")
  .def("get_test_metrics_history", &PSG::get_test_metrics_history, "get_test_metrics_history")
  ;
}
