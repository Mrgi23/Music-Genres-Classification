#include "Dataset.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <torch/extension.h>

namespace py = pybind11;

void RegisterDataset(py::module_ & m)
{
    py::class_<AudioDataset>(m, "AudioDataset")
    .def(
        py::init<const fs::path &, Preprocessor *>(),
        py::arg("root_path"),
        py::arg("preprocessor"),
        py::keep_alive<1, 3>()
    )
    .def(
        "get_classes", [](const AudioDataset& self)
        {
            py::dict out;
            auto cls = self.GetClasses();
            for (auto it = cls.begin(); it != cls.end(); ++it)
                out[py::int_(it->key())] = py::str(it->value());
            return out;
        }
    )
    .def("get_preprocessor", &AudioDataset::GetPreprocessor)
    .def(
        "__getitem__", [](AudioDataset & self, size_t index)
        {
            auto ex = self.get(index);
            return py::make_tuple(ex.data, ex.target);
        }
    )
    .def(
        "__len__", [](const AudioDataset & self)
        {
            auto s = self.size();
            return s.has_value() ? static_cast<size_t>(*s) : 0;
        }
    );
}

void RegisterSubset(py::module_ & m)
{
    py::class_<AudioSubset>(m, "AudioSubset")
    .def(
        py::init<AudioDataset *, std::vector<size_t>>(),
        py::arg("dataset"),
        py::arg("indices"),
        py::keep_alive<1, 2>()
    )
    .def("get_dataset", &AudioSubset::GetDataset)
    .def("get_stacked_data", &AudioSubset::GetStackedData)
    .def(
        "__getitem__", [](AudioSubset & self, size_t index)
        {
            auto ex = self.get(index);
            return py::make_tuple(ex.data, ex.target);
        }
    )
    .def(
        "__len__", [](const AudioSubset & self)
        {
            auto s = self.size();
            return s.has_value() ? static_cast<size_t>(*s) : 0;
        }
    );
}
