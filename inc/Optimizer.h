#pragma once

#ifdef __cplusplus

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
 */
struct OptimizerConfig
{
    double lr, momentum, alpha, eps, decay;
    bool nesterov, amsgrad;

    /**
     * @brief Construct a new OptimizerConfig with optional overrides.
     *
     * Initializes an optimizer configuration with a required learning rate
     * and optional overrides for momentum, weight decay, and other optimizer
     * hyperparameters.
     *
     * @param[in] lr_ Learning rate.
     * @param[in] momentum_ Momentum factor. Defaults to 0.0.
     * @param[in] alpha_ Alpha parameter for RMSprop. Defaults to 0.99.
     * @param[in] eps_ Epsilon for numerical stability. Defaults to 1e-8.
     * @param[in] decay_ Weight decay factor (L2 regularization).  Defaults to 0.0.
     * @param[in] nesterov_ Enable Nesterov momentum for SGD. Defaults to false.
     * @param[in] amsgrad_ Enable AMSGrad variant for Adam/AdamW. Defaults to false.
     */
    OptimizerConfig(
        double lr_,
        double momentum_ = 0.0,
        double alpha_ = 0.99,
        double eps_ = 1e-8,
        double decay_ = 0.0,
        bool nesterov_ = false,
        bool amsgrad_ = false
    ) : lr(lr_), momentum(momentum_), alpha(alpha_), eps(eps_), decay(decay_), nesterov(nesterov_), amsgrad(amsgrad_) {}
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
 *
 * @throws std::invalid_argument If an unsupported optimizer type is requested.
 */
inline void CreateOptimizer(
    const std::vector<torch::Tensor> & parameters,
    OptimizerType type,
    const OptimizerConfig & cfg,
    std::unique_ptr<torch::optim::Optimizer> & opt
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

#endif
