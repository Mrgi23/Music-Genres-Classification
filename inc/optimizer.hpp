#pragma once

#include <stdexcept>
#include <torch/optim/optimizer.h>
#include <torch/optim/adam.h>
#include <torch/optim/adamw.h>
#include <torch/optim/rmsprop.h>
#include <torch/optim/sgd.h>
#include <torch/types.h>

/**
 * @enum OptimizerType
 * @brief Enumeration of supported optimizer types.
 */
enum class OptimizerType {Adam, AdamW, RMSprop, SGD};

/**
 * @struct OptimizerConfig
 * @brief Configuration parameters for optimizer creation.
 *
 * This struct defines the hyperparameters required to
 * construct a torch optimizer.
 *
 * @param[in] lr Learning rate.
 * @param[in] momentum Momentum factor. Defaults to 0.0.
 * @param[in] alpha Alpha parameter for RMSprop. Defaults to 0.99.
 * @param[in] eps Epsilon for numerical stability. Defaults to 1e-8.
 * @param[in] decay Weight decay factor (L2 regularization).  Defaults to 0.0.
 * @param[in] nesterov Enable Nesterov momentum for SGD. Defaults to false.
 * @param[in] amsgrad Enable AMSGrad variant for Adam/AdamW. Defaults to false.
 */
struct OptimizerConfig
{
  double lr;
  double momentum = 0.0;
  double alpha = 0.99;
  double eps = 1e-8;
  double decay = 0.0;
  bool nesterov = false;
  bool amsgrad = false;
};

/**
 * @brief Factory function to create a torch optimizer.
 *
 * Creates one of the supported optimizers (Adam, AdamW, RMSprop, SGD)
 * with the provided configuration and attaches it to the given parameters.
 *
 * @param[in] parameters Model parameters to optimize.
 * @param[in] type Optimizer type to create.
 * @param[in] cfg Optimizer configuration.
 * @param[in, out] opt Reference to unique_ptr where the created optimizer will be stored.
 * @throws std::invalid_argument If an unsupported optimizer type is requested.
 */
inline void createOptimizer(
  const std::vector<torch::Tensor>& parameters,
  OptimizerType type,
  const OptimizerConfig& cfg,
  std::unique_ptr<torch::optim::Optimizer>& opt
)
{
  switch(type)
  {
    case OptimizerType::Adam:
      opt = std::make_unique<torch::optim::Adam>(
        parameters,
        torch::optim::AdamOptions(cfg.lr)
        .weight_decay(cfg.decay)
        .amsgrad(cfg.amsgrad)
      );
      break;
    case OptimizerType::AdamW:
      opt = std::make_unique<torch::optim::AdamW>(
        parameters,
        torch::optim::AdamWOptions(cfg.lr)
        .weight_decay(cfg.decay)
        .amsgrad(cfg.amsgrad)
      );
      break;
    case OptimizerType::RMSprop:
      opt = std::make_unique<torch::optim::RMSprop>(
        parameters,
        torch::optim::RMSpropOptions(cfg.lr)
        .momentum(cfg.momentum)
        .alpha(cfg.alpha)
        .eps(cfg.eps)
        .weight_decay(cfg.decay)
      );
      break;
    case OptimizerType::SGD:
      opt = std::make_unique<torch::optim::SGD>(
        parameters,
        torch::optim::SGDOptions(cfg.lr)
        .momentum(cfg.momentum)
        .weight_decay(cfg.decay)
        .nesterov(cfg.nesterov)
      );
      break;
    default:
      throw std::invalid_argument("CreateOptimizer: Invalid optimizer type.");
  }
}
