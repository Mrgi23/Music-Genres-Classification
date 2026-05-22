#include <pybind11/pybind11.h>

namespace py = pybind11;

void registerDownloader(py::module_&);
void registerPreprocessor(py::module_&);
void registerDataset(py::module_&);
void registerSubset(py::module_&);
void registerModel(py::module_&);
void registerScheduler(py::module_&);
void registerOptimizer(py::module_&);
void registerDeviceManager(py::module_&);
void registerTrainer(py::module_&);

PYBIND11_MODULE(musicnet, m)
{
  try
  {
    py::module_::import("torch");
  }
  catch (...)
  {
    throw std::runtime_error("Failed to import torch. Make sure PyTorch is installed.");
  }

  registerDownloader(m);
  registerPreprocessor(m);
  registerDataset(m);
  registerSubset(m);
  registerModel(m);
  registerScheduler(m);
  registerOptimizer(m);
  registerDeviceManager(m);
  registerTrainer(m);
}