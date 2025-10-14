#include "Model.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <torch/extension.h>

namespace py = pybind11;

void RegisterModel(py::module_ & m)
{
    py::class_<MusicModel>(m, "MusicModel")
    .def(py::init<>())
    .def(
        "parameters", [](MusicModel & self)
        {
            return self->parameters();
        }
    )
    .def(
        "__call__", [](MusicModel & self, torch::Tensor x)
        {
            return self->forward(x);
        }
    )
    .def(
        "forward", [](MusicModel & self, torch::Tensor x)
        {
            return self->forward(x);
        }
    )
    .def(
        "to", [](MusicModel & self, const std::string & device_str) -> MusicModel&
        {
            torch::Device device{device_str};
            self.to(device);
            return self;
        },
        py::arg("device")
    )
    .def(
        "train", [](MusicModel & self, bool on) -> MusicModel&
        {
            self->train(on);
            return self;
        },
        py::arg("on") = true,
        py::return_value_policy::reference
    )
    .def(
        "eval", [](MusicModel & self) -> MusicModel&
        {
            self->eval();
            return self;
        },
        py::return_value_policy::reference
    )
    .def(
        "is_training", [](const MusicModel & self)
        {
            return self->is_training();
        }
    )
    .def(
        "save", &MusicModel::Save,
        py::arg("file_path")
    )
    .def(
        "load", &MusicModel::Load,
        py::arg("file_path")
    );
}