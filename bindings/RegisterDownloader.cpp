#include "Downloader.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl/filesystem.h>

namespace py = pybind11;

void registerDownloader(py::module_& m)
{
  py::class_<Downloader>(m, "Downloader")
  .def(
    py::init<const fs::path&, const std::string&>(),
    py::arg("root_path"),
    py::arg("url") = "https://s3.mrgi23.com/artifacts/Music-Genres-Classification/dataset/dataset.tar.zst"
    )
  .def("download_and_extract", &Downloader::downloadAndExtract)
  .def("root_path", &Downloader::rootPath)
  .def_static(
    "download_from_url", &Downloader::downloadFromUrl,
    py::arg("file_path"),
    py::arg("url")
  );
}
