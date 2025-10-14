#pragma once

#ifdef __cplusplus

#include <string>
#include <torch/optim/optimizer.h>

/**
 * @class ReduceLROnPlateau
 * @brief Learning rate scheduler that reduces the LR when a metric has stopped improving.
 *
 * This scheduler monitors a validation metric and reduces the optimizer's learning rate
 * when no improvement has been seen for a number of epochs (patience).
 * It supports both minimization (e.g., loss) and maximization (e.g., accuracy) modes.
 */
class ReduceLROnPlateau
{
    public:
        /**
         * @brief Construct a new ReduceLROnPlateau scheduler.
         *
         * @param[in] mode Either "min" (lower metric is better) or "max" (higher metric is better).
         * @param[in] factor Multiplicative factor for reducing the learning rate.
         * @param[in] patience Number of epochs with no improvement after which LR will be reduced.
         *
         * @throws std::invalid_argument If mode is not "min" or "max".
         */
        ReduceLROnPlateau(const std::string & mode, float factor, uint patience);
        /**
         * @brief Destructor.
         *
         * Default cleanup.
         */
        ~ReduceLROnPlateau();

        /**
         * @brief Attach an optimizer to this scheduler.
         *
         * @param[in,out] opt Pointer to the torch optimizer whose learning rate will be scheduled.
         */
        void AttachOptimizer(torch::optim::Optimizer * opt);

        /**
         * @brief Perform a scheduler step based on the given metric.
         *
         * If the metric improves according to the mode, the scheduler resets.
         * Otherwise, after `patience` epochs without improvement, the learning rate
         * is multiplied by `factor` for all optimizer parameter groups.
         *
         * @param[in] metric The current value of the monitored metric.
         *
         * @throws std::runtime_error If called before attaching an optimizer.
         */
        void UpdateLR(float metric);
    private:
        torch::optim::Optimizer * m_opt;
        bool m_maximize;
        float m_factor;
        uint m_patience;
        uint m_badEpochs;
        float m_bestMetric;
};

#endif
