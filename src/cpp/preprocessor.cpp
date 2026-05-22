#include "preprocessor.hpp"
#include <aubio/fmat.h>
#include <aubio/cvec.h>
#include <aubio/io/source.h>
#include <aubio/musicutils.h>
#include <aubio/spectral/fft.h>
#include <aubio/spectral/mfcc.h>
#include <aubio/utils/log.h>
#include <cmath>
#include <stdexcept>

Preprocessor::Preprocessor(const PreprocessorConfig& cfg) : m_cfg(cfg)
{
  m_windowVector = new_aubio_window((char*)("hanning"), m_cfg.nfft);

  m_mean = torch::zeros({1 + m_cfg.size / m_cfg.hop, m_cfg.nmfcc});
  m_std = torch::ones({1 + m_cfg.size / m_cfg.hop, m_cfg.nmfcc});
}

Preprocessor::~Preprocessor()
{
  del_fvec(m_windowVector);
  aubio_cleanup();
}

torch::Tensor Preprocessor::processFile(const fs::path& filePath)
{
  AudioData audioData = loadAndCrop(filePath);

  cvec_t* stftVector = new_cvec(m_cfg.nfft);
  aubio_fft_t* fftObject = new_aubio_fft(m_cfg.nfft);
  aubio_mfcc_t* mfccObject = new_aubio_mfcc(m_cfg.nfft, m_cfg.nmels, m_cfg.nmfcc, audioData.sr);

  audioData.signal.insert(audioData.signal.begin(), m_cfg.nfft / 2, 0.0f);
  audioData.signal.insert(audioData.signal.end(), m_cfg.nfft / 2, 0.0f);

  std::vector<float> mfcc;
  uint numFrames = (audioData.signal.size() - m_cfg.nfft) / m_cfg.hop + 1;
  for (uint i = 0; i < numFrames; ++i)
  {
    fvec_t* frameVector = new_fvec(m_cfg.nfft);
    for (uint j = 0; j < m_cfg.nfft; ++j)
      frameVector->data[j] = audioData.signal[i*  m_cfg.hop + j]*  m_windowVector->data[j];

    aubio_fft_do(fftObject, frameVector, stftVector);

    fvec_t* mfccVector = new_fvec(m_cfg.nmfcc);
    aubio_mfcc_do(mfccObject, stftVector, mfccVector);
    for (uint j = 0; j < m_cfg.nmfcc; j++)
      mfcc.push_back(mfccVector->data[j]);

    del_fvec(frameVector);
    del_fvec(mfccVector);
  }

  del_cvec(stftVector);
  del_aubio_fft(fftObject);
  del_aubio_mfcc(mfccObject);
  return torch::tensor(mfcc, torch::kFloat32).view({1, static_cast<int64_t>(numFrames), static_cast<int64_t>(m_cfg.nmfcc)});
}

torch::Tensor Preprocessor::normalizeData(const torch::Tensor& x)
{
  return (x - m_mean) / m_std;
}

PreprocessorConfig Preprocessor::config() const
{
  return m_cfg;
}

torch::Tensor Preprocessor::mean() const
{
  return m_mean;
}

void Preprocessor::setMean(const torch::Tensor& mean)
{
  m_mean = mean;
}

torch::Tensor Preprocessor::std() const
{
  return m_std;
}

void Preprocessor::setStd(const torch::Tensor& std)
{
  m_std = std;
}

Preprocessor::AudioData::AudioData(std::vector<float> signal_, uint sr_) : signal(std::move(signal_)), sr(sr_)
{
}

void Preprocessor::silent_log(int level, const char* message, void* data)
{
}

Preprocessor::AudioData Preprocessor::loadAndCrop(const fs::path& filePath)
{
  aubio_log_set_level_function(AUBIO_LOG_ERR, silent_log, nullptr);

  uint sr = 0;
  aubio_source_t* sourceObject = new_aubio_source(filePath.c_str(), sr, m_cfg.size);
  if (!sourceObject)
    throw std::runtime_error("Preprocessor::loadAndCrop: File: " + filePath.string() + " is invalid or corrupted.");

  fvec_t* frameVector = new_fvec(m_cfg.size);
  uint read = 0;
  aubio_source_do(sourceObject, frameVector,& read);
  sr = aubio_source_get_samplerate(sourceObject);

  std::vector<float> signal(frameVector->data, frameVector->data + read);

  del_fvec(frameVector);
  del_aubio_source(sourceObject);
  return AudioData(std::move(signal), sr);
}
