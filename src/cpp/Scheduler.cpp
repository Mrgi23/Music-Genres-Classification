#include "Scheduler.h"
#include <limits>
#include <stdexcept>

ReduceLROnPlateau::ReduceLROnPlateau(const std::string& mode, float factor, uint patience)
  : m_factor(factor), m_patience(patience)
{
  m_opt = nullptr;

  if (mode == "min")
  {
    m_maximize = false;
    m_bestMetric = std::numeric_limits<float>::infinity();
  }
  else if (mode == "max")
  {
    m_maximize = true;
    m_bestMetric = -std::numeric_limits<float>::infinity();
  }
  else
    throw std::invalid_argument("ReduceLROnPlateau::ReduceLROnPlateau: Mode must be 'min' or 'max'");
}

ReduceLROnPlateau::~ReduceLROnPlateau() = default;

void ReduceLROnPlateau::attach(torch::optim::Optimizer* opt)
{
  m_opt = opt;
}

void ReduceLROnPlateau::update(float metric)
{
  if (!m_opt)
    throw std::runtime_error("ReduceLROnPlateau::update: Optimizer is not attached.");

  bool improved = m_maximize ? (metric > m_bestMetric) : (metric < m_bestMetric);
  if (improved)
  {
    m_bestMetric = metric;
    m_badEpochs = 0;
  }
  else
  {
    m_badEpochs++;
    if (m_badEpochs > m_patience)
    {
      for (auto& group : m_opt->param_groups())
      {
        float newLR = m_factor * group.options().get_lr();
        group.options().set_lr(newLR);
      }
      m_badEpochs = 0;
    }
  }
}
