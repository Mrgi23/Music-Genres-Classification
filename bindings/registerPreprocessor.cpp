#include "preprocessor.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl/filesystem.h>
#include <torch/extension.h>

namespace py = pybind11;

void registerPreprocessor(py::module_& m)
{
  py::class_<PreprocessorConfig>(m, "PreprocessorConfig")
  .def(
    py::init<uint, uint, uint, uint, uint>(),
    py::arg("size") = 660000U,
    py::arg("nfft") = 1024U,
    py::arg("hop") = 512U,
    py::arg("nmels") = 128U,
    py::arg("nmfcc") = 13U
  )
  .def_readwrite("size", &PreprocessorConfig::size)
  .def_readwrite("nfft", &PreprocessorConfig::nfft)
  .def_readwrite("hop", &PreprocessorConfig::hop)
  .def_readwrite("nmels", &PreprocessorConfig::nmels)
  .def_readwrite("nmfcc", &PreprocessorConfig::nmfcc);

  py::class_<Preprocessor>(m, "Preprocessor")
  .def(
    py::init<const PreprocessorConfig&>(),
    py::arg("cfg") = PreprocessorConfig()
  )
  .def(
    "process_file", &Preprocessor::processFile,
    py::arg("file_path")
  )
  .def(
    "normalize_data", &Preprocessor::normalizeData,
    py::arg("x")
  )
  .def("mean", &Preprocessor::mean)
  .def("set_mean", &Preprocessor::setMean)
  .def("std", &Preprocessor::std)
  .def("set_std", &Preprocessor::setStd);
}
