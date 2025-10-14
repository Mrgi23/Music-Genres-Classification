#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declare registration functions
void RegisterDownloader(py::module_ &);
void RegisterPreprocessor(py::module_ &);
void RegisterDataset(py::module_ &);
void RegisterSubset(py::module_ &);
void RegisterModel(py::module_ &);
void RegisterScheduler(py::module_ &);
void RegisterOptimizer(py::module_ &);
void RegisterDeviceManager(py::module_ &);
void RegisterTrainer(py::module_ &);

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

    m.doc() = R"doc(
MusicNet: A library for music genre classification.

This module provides end-to-end components for training and evaluating
neural networks on music data. It includes:

- Downloader: utilities for fetching datasets (e.g., GTZAN).
- Preprocessor: feature extraction (spectrograms, MFCCs, normalization).
- AudioDataset: a dataset interface that applies preprocessing and provides samples ready for batching.
- AudioSubset: a subset wrapper for creating train/validation/test splits.
- MusicModel: CNN-based architectures for spectrogram and MFCC input.
- ReduceLROnPlateau: learning rate scheduling utilities.
- Optimizer: wrappers for common optimizers (Adam, SGD, RMSprop, etc.).
- Trainer: training and evaluation loops with optimizer and scheduler support.

The design allows mixing Python and C++:
- Heavy preprocessing and training can be run from C++ for performance.
- High-level orchestration, experiments, and prototyping can be run from Python.

Example:

    >>> import musicnet as mn

    >>> downloader = mn.Downloader('./resources')
    >>> downloader.download_and_extract()

    >>> preprocessor = mn.Preprocessor(downloader.get_root_path())
    >>> dataset = mn.AudioDataset(downloader.get_root_path(), preprocessor)
    >>> train_indices, val_indices, test_indices = random_split_indices(dataset)
    >>> train_subset, val_subset, test_subset = \
    >>>     mn.AudioSubset(dataset, train_indices), mn.AudioSubset(dataset, val_indices), mn.AudioSubset(dataset, test_indices)

    >>> model = mn.Model()
    >>> scheduler = mn.ReduceLROnPlateau("max", 0.5, 10)
    >>> trainer = mn.Trainer(model, mn.OptimizerType.Adam, mn.OptimizerConfig(lr=0.001))
    >>> trainer.attach_scheduler(scheduler)

    >>> for epoch in range(1, num_epochs + 1):
    >>>     train_loss, train_acc = trainer.train_model(train_subset, batch_size=16, num_workers=4)
    >>>     val_loss, val_acc = trainer.eval_model(val_subset, batch_size=16, num_workers=4)
    >>>     scheduler.update_lr(val_acc)

    >>> loss, acc = trainer.eval_model(test_subset, batch_size=16, num_workers=4)
    >>> print(f"loss={loss}, acc={acc}")
)doc";

    RegisterDownloader(m);
    RegisterPreprocessor(m);
    RegisterDataset(m);
    RegisterSubset(m);
    RegisterModel(m);
    RegisterScheduler(m);
    RegisterOptimizer(m);
    RegisterDeviceManager(m);
    RegisterTrainer(m);
}