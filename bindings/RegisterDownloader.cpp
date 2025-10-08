#include "Downloader.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl/filesystem.h>

namespace py = pybind11;

void RegisterDownloader(py::module_ & m)
{

    py::class_<Downloader>(m, "Downloader")
    .def(
        py::init<const fs::path &, const std::string &>(),
        py::arg("root_path"),
        py::arg("url") = "https://www.kaggle.com/api/v1/datasets/download/andradaolteanu/gtzan-dataset-music-genre-classification"
    )
    .def("download_and_extract", &Downloader::DownloadAndExtract)
    .def("get_root_path", &Downloader::GetRootPath)
    .def_static(
        "download_from_url", &Downloader::DownloadFromUrl,
        py::arg("file_path"),
        py::arg("url")
    );
}
