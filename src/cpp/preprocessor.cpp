#include <cmath>
#include <stdexcept>
#include "preprocessor.h"

using namespace std;

Preprocessor::Preprocessor(uint size, uint nfft, uint hop, uint nmels, uint nmfcc)
: m_size(size), m_nfft(nfft), m_hop(hop), m_nmels(nmels), m_nmfcc(nmfcc)
{
    // Compute window.
    m_windowVector = new_aubio_window((char*)("hanning"), m_nfft);

    // Initialize mean and std.
    m_mean = torch::zeros({1 + m_size / m_hop, m_nmfcc});
    m_std = torch::ones({1 + m_size / m_hop, m_nmfcc});
}

Preprocessor::~Preprocessor()
{
    del_fvec(m_windowVector);
    aubio_cleanup();
}

torch::Tensor Preprocessor::run(const fs::path & filePath)
{
    // Load and crop audio signal.
    uint sampleRate;
    vector<float> audioSignal = loadAndCrop(filePath, sampleRate, m_size);

    // Prepare spectrogram and MFCC.
    cvec_t * stftVector = new_cvec(m_nfft);
    aubio_fft_t * fftObject = new_aubio_fft(m_nfft);
    aubio_mfcc_t * mfccObject = new_aubio_mfcc(m_nfft, m_nmels, m_nmfcc, sampleRate);

    // Padd the signal, for the window function.
    audioSignal.insert(audioSignal.begin(), m_nfft / 2, 0.0f);
    audioSignal.insert(audioSignal.end(), m_nfft / 2, 0.0f);

    // Process frame by frame.
    vector<float> mfcc;
    uint numFrames = (audioSignal.size() - m_nfft) / m_hop + 1;
    for (uint i = 0; i < numFrames; ++i)
    {
        // Perform signal windowing.
        fvec_t * frameVector = new_fvec(m_nfft);
        for (uint j = 0; j < m_nfft; ++j)
        {
            frameVector->data[j] = audioSignal[i * m_hop + j] * m_windowVector->data[j];
        }

        // Copmupte frame STFT.
        aubio_fft_do(fftObject, frameVector, stftVector);

        // Compute frame MFCC.
        fvec_t * mfccVector = new_fvec(m_nmfcc);
        aubio_mfcc_do(mfccObject, stftVector, mfccVector);
        for (uint j = 0; j < m_nmfcc; j++)
        {
            mfcc.push_back(mfccVector->data[j]);
        }

        // Clean up.
        del_fvec(frameVector);
        del_fvec(mfccVector);
    }

    // Clean up.
    del_cvec(stftVector);
    del_aubio_fft(fftObject);
    del_aubio_mfcc(mfccObject);
    return torch::tensor(mfcc, torch::kFloat32).view({1, static_cast<int64_t>(numFrames), static_cast<int64_t>(m_nmfcc)});
}

torch::Tensor Preprocessor::normalize(const torch::Tensor & x)
{
    // Normalize the data.
    torch::Tensor y = (x - m_mean) / m_std;
    return y;
}

torch::Tensor Preprocessor::mean() const
{
    // Return the mean.
    return m_mean;
}

void Preprocessor::setMean(torch::Tensor mean)
{
    // Set the mean.
    m_mean = mean;
}

torch::Tensor Preprocessor::std() const
{
    // Return the std.
    return m_std;
}

void Preprocessor::setStd(torch::Tensor std)
{
    // Set the std.
    m_std = std;
}

std::vector<float> Preprocessor::loadAndCrop(const fs::path & filePath, uint & sampleRate, uint size)
{
    // Silent the aubio errors.
    aubio_log_set_level_function(AUBIO_LOG_ERR, silent_log, nullptr);

    // Load audio signal from the file.
    sampleRate = 0;
    aubio_source_t * sourceObject = new_aubio_source(filePath.c_str(), sampleRate, m_size);
    if (!sourceObject)
    {
        del_aubio_source(sourceObject);
        throw runtime_error("Preprocessor::loadAndCrop: Invalid or corrupted file.");
    }

    // Load waveform and crop it.
    fvec_t * frameVector = new_fvec(m_size);
    uint read = 0;
    aubio_source_do(sourceObject, frameVector, &read);
    sampleRate = aubio_source_get_samplerate(sourceObject);

    // Copy to vector.
    vector<float> signal(frameVector->data, frameVector->data + read);

    // Clean up
    del_fvec(frameVector);
    del_aubio_source(sourceObject);
    return signal;
}

void Preprocessor::silent_log(int level, const char * message, void * data)
{
    // Do nothing!
}
