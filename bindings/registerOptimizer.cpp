#include "optimizer.hpp"
#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace py = pybind11;

void registerOptimizer(py::module_& m)
{
  py::enum_<OptimizerType>(m, "OptimizerType")
  .value("Adam", OptimizerType::Adam)
  .value("AdamW", OptimizerType::AdamW)
  .value("RMSprop", OptimizerType::RMSprop)
  .value("SGD", OptimizerType::SGD);

  py::class_<OptimizerConfig>(m, "OptimizerConfig")
  .def(
    py::init<double, double, double, double, double, bool, bool>(),
    py::arg("lr"),
    py::arg("momentum") = 0.0,
    py::arg("alpha") = 0.99,
    py::arg("eps") = 1e-8,
    py::arg("decay") = 0.0,
    py::arg("nesterov") = false,
    py::arg("amsgrad") = false
  )
  .def_readwrite("lr", &OptimizerConfig::lr)
  .def_readwrite("momentum", &OptimizerConfig::momentum)
  .def_readwrite("alpha", &OptimizerConfig::alpha)
  .def_readwrite("eps", &OptimizerConfig::eps)
  .def_readwrite("decay", &OptimizerConfig::decay)
  .def_readwrite("nesterov", &OptimizerConfig::nesterov)
  .def_readwrite("amsgrad", &OptimizerConfig::amsgrad);
}
