#include <limits>
#include <stdexcept>
#include "scheduler.h"

using namespace std;

ReduceLROnPlateau::ReduceLROnPlateau(string mode, float factor, uint patience) : m_factor(factor), m_patience(patience)
{
    m_opt = nullptr;

    if (mode == "min")
    {
        m_maximize = false;
        m_bestMetric = numeric_limits<float>::infinity();
    }
    else if (mode == "max")
    {
        m_maximize = true;
        m_bestMetric = -numeric_limits<float>::infinity();
    }
    else
    {
        throw invalid_argument("ReduceLROnPlateau::ReduceLROnPlateau: Mode must be 'min' or 'max'");
    }
}

ReduceLROnPlateau::~ReduceLROnPlateau()
{

}

void ReduceLROnPlateau::attachOptimizer(torch::optim::Optimizer * opt)
{
    m_opt = opt;
}

void ReduceLROnPlateau::step(float metric)
{
    if (!m_opt)
    {
        throw runtime_error("ReduceLROnPlateau::step: Optimizer is not attached.");
    }

    bool improved = m_maximize ? (metric > m_bestMetric) : (metric < m_bestMetric);
    if (improved)
    {
        // Metric improved, reset scheduler.
        m_bestMetric = metric;
        m_badEpochs = 0;
    }
    else
    {
        m_badEpochs++;
        if (m_badEpochs > m_patience)
        {
            // Decay learning rate for all parameter groups, and reset scheduler.
            for (auto& group : m_opt->param_groups())
            {
                float newLR = m_factor * group.options().get_lr();
                group.options().set_lr(newLR);
            }
            m_badEpochs = 0;
        }
    }
}
