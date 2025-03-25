#include <cmath>
#include <stdexcept>
#include "preprocessor.h"

using namespace std;

Preprocessor::Preprocessor(uint size, uint nfft, uint hop, uint nmels, uint nmfcc)
: size(size), nfft(nfft), hop(hop), nmels(nmels), nmfcc(nmfcc) {
    // Compute window.
    windowVector = new_aubio_window((char*)("hanning"), nfft);

    // Initialize mean and std.
    mean = torch::zeros({1 + size / hop, nmfcc});
    std = torch::ones({1 + size / hop, nmfcc});
}

std::vector<float> Preprocessor::loadAndCrop(const fs::path& filePath, uint& sampleRate, uint size) {
    // Silent the aubio errors.
    aubio_log_set_level_function(AUBIO_LOG_ERR, silent_log, nullptr);

    // Load audio signal from the file.
    sampleRate = 0;
    aubio_source_t * sourceObject = new_aubio_source(filePath.c_str(), sampleRate, size);
    if (!sourceObject) {
        del_aubio_source(sourceObject);
        throw runtime_error("Preprocessor::loadAndCrop: Invalid or corrupted file.");
    }

    // Load waveform and crop it.
    fvec_t * frameVector = new_fvec(size);
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

torch::Tensor Preprocessor::run(
    const fs::path& filePath
) {
    // Load and crop audio signal.
    uint sampleRate;
    vector<float> audioSignal = loadAndCrop(filePath, sampleRate, size);

    // Prepare spectrogram and MFCC.
    cvec_t * stftVector = new_cvec(nfft);
    aubio_fft_t * fftObject = new_aubio_fft(nfft);
    aubio_mfcc_t * mfccObject = new_aubio_mfcc(nfft, nmels, nmfcc, sampleRate);

    // Padd the signal, for the window function.
    audioSignal.insert(audioSignal.begin(), nfft / 2, 0.0f);
    audioSignal.insert(audioSignal.end(), nfft / 2, 0.0f);
    uint numFrames = (audioSignal.size() - nfft) / hop + 1;

    // Process frame by frame.
    vector<float> mfcc;
    for (uint i = 0; i < numFrames; ++i) {
        // Perform signal windowing.
        fvec_t * frameVector = new_fvec(nfft);
        for (uint j = 0; j < nfft; ++j) { frameVector->data[j] = audioSignal[i * hop + j] * windowVector->data[j]; }

        // Copmupte frame STFT.
        aubio_fft_do(fftObject, frameVector, stftVector);

        // Compute frame MFCC.
        fvec_t * mfccVector = new_fvec(nmfcc);
        aubio_mfcc_do(mfccObject, stftVector, mfccVector);
        for (uint j = 0; j < nmfcc; j++) { mfcc.push_back(mfccVector->data[j]); }

        // Clean up.
        del_fvec(frameVector);
        del_fvec(mfccVector);
    }

    // Clean up.
    del_cvec(stftVector);
    del_aubio_fft(fftObject);
    del_aubio_mfcc(mfccObject);
    return torch::tensor(mfcc, torch::kFloat32).view({1, static_cast<int64_t>(numFrames), static_cast<int64_t>(nmfcc)});
}

torch::Tensor Preprocessor::normalize(const torch::Tensor& x) {
    // Normalize the data.
    torch::Tensor y = (x - mean) / std;
    return y;
}
