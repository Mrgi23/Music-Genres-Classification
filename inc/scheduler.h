#ifndef SCHEDULER_H
#define SCHEDULER_H

#ifdef __cplusplus

#include <torch/torch.h>

class ReduceLROnPlateau
{
    public:
        ReduceLROnPlateau(std::string mode, float factor, uint patience);
        ~ReduceLROnPlateau();

        void attachOptimizer(torch::optim::Optimizer * opt);
        void step(float metric);
    private:
        torch::optim::Optimizer * m_opt;
        bool m_maximize;
        float m_factor;
        uint m_patience;
        uint m_badEpochs;
        float m_bestMetric;
};

#endif

#endif