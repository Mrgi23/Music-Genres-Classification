#include "Trainer.h"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;

void registerDeviceManager(py::module_& m)
{
  py::class_<DeviceManager>(m, "DeviceManager")
  .def_static(
    "get", []()
    {
      return DeviceManager::get().str();
    }
  );
}

void registerTrainer(py::module_& m)
{
  py::class_<Trainer>(m, "Trainer")
  .def(
    py::init<MusicModel&, const OptimizerType&, const OptimizerConfig&>(),
    py::arg("model"),
    py::arg("type"),
    py::arg("cfg")
    )
  .def(
    "attach", &Trainer::attach,
    py::arg("scheduler")
  )
  .def(
    "train", [](Trainer& self, AudioSubset* subset, size_t batch_size, int num_workers)
    {
      float loss = 0.0f, acc = 0.0f;
      auto loader = torch::data::make_data_loader<RandomSampler>
      (
        subset->map(Stack<>()),
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(num_workers)
      );

      {
        py::gil_scoped_release release;
        self.train(*loader, loss, acc);
      }
      return py::make_tuple(loss, acc);
    },
    py::arg("subset"),
    py::arg("batch_size"),
    py::arg("num_workers")
  )
  .def(
    "eval", [](Trainer& self, AudioSubset* subset, size_t batch_size, int num_workers)
    {
      float loss = 0.0f, acc = 0.0f;
      auto loader = torch::data::make_data_loader<SequentialSampler>
      (
        subset->map(Stack<>()),
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(num_workers)
      );

      {
        py::gil_scoped_release release;
        self.eval(*loader, loss, acc);
      }
      return py::make_tuple(loss, acc);
    },
    py::arg("subset"),
    py::arg("batch_size"),
    py::arg("num_workers")
  );
}
