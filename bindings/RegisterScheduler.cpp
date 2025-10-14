#include "Scheduler.h"

#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace py = pybind11;

void RegisterScheduler(py::module_ &m)
{
    py::class_<ReduceLROnPlateau>(m, "ReduceLROnPlateau")
    .def(
        py::init<const std::string &, float, uint>(),
        py::arg("mode"),
        py::arg("factor"),
        py::arg("patience")
    )
    .def(
        "update_lr", &ReduceLROnPlateau::UpdateLR,
        py::arg("metric")
    );
}
