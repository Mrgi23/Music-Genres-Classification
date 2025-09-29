#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#ifdef __cplusplus

#include <stdexcept>
#include <torch/torch.h>

enum class OptimizerType {Adam, AdamW, SGD, RMSprop};

struct OptimizerConfig
{
    double lr, momentum, alpha, eps, decay;
    bool nesterov, amsgrad;

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

inline void createOptimizer(
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
        case OptimizerType::SGD:
            opt = std::make_unique<torch::optim::SGD>(
                parameters,
                torch::optim::SGDOptions(cfg.lr)
                .momentum(cfg.momentum)
                .weight_decay(cfg.decay)
                .nesterov(cfg.nesterov)
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
        default:
            throw std::invalid_argument("createOptimizer: Invalid optimizer type.");
    }
}

#endif

#endif