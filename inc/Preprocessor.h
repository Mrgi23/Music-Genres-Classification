#pragma once

#ifdef __cplusplus

#include <aubio/types.h>
#include <aubio/fvec.h>
#include <filesystem>
#include <torch/types.h>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

/**
 * @struct PreprocessorConfig
 * @brief Configuration parameters for audio preprocessing.
 */
struct PreprocessorConfig
{
    uint size;
    uint nfft;
    uint hop;
    uint nmels;
    uint nmfcc;

    /**
     * @brief Construct a new PreprocessorConfig with optional overrides.
     *
     * @param[in] size_ Input signal length. Defaults to 660,000 samples (≈30s at 22.05kHz).
     * @param[in] nfft_ FFT size for spectral analysis. Defaults to 1024.
     * @param[in] hop_ Hop size between frames. Defaults to 512.
     * @param[in] nmels_ Number of mel bands. Defaults to 128.
     * @param[in] nmfcc_ Number of MFCC coefficients. Defaults to 13.
     */
    PreprocessorConfig(
        uint size_ = 660000U,
        uint nfft_ = 1024U,
        uint hop_ = 512U,
        uint nmels_ = 128U,
        uint nmfcc_ = 13U
    ) : size(size_), nfft(nfft_), hop(hop_), nmels(nmels_), nmfcc(nmfcc_) {}
};


/**
 * @class Preprocessor
 * @brief Audio preprocessing pipeline for feature extraction and normalization.
 *
 * This class uses the Aubio library to extract MFCC features from audio files
 * and prepares them as Torch tensors for downstream ML tasks. It also supports
 * normalization using precomputed mean and standard deviation tensors.
 */
class Preprocessor
{
    public:
        /**
         * @brief Construct a new Preprocessor object.
         *
         * @param[in] cfg Configuration parameters for preprocessing. Defaults to default config.
         */
        Preprocessor(const PreprocessorConfig & cfg = PreprocessorConfig());
        /**
         * @brief Destructor.
         *
         * Cleans up Aubio objects.
         */
        ~Preprocessor();

        /**
         * @brief Get the preprocessing configuration.
         * @return PreprocessorConfig A copy of the configuration.
         */
        PreprocessorConfig GetCfg() const;

        /**
         * @brief Get the mean tensor used for normalization.
         * @return torch::Tensor Mean tensor.
         */
        torch::Tensor GetMean() const;
        /**
         * @brief Set the mean tensor used for normalization.
         * @param[in] mean Mean tensor.
         */
        void SetMean(torch::Tensor mean);
        /**
         * @brief Get the standard deviation tensor used for normalization.
         * @return torch::Tensor Standard deviation tensor.
         */
        torch::Tensor GetStd() const;
        /**
         * @brief Set the standard deviation tensor used for normalization.
         * @param[in] std Standard deviation tensor.
         */
        void SetStd(torch::Tensor std);

        /**
         * @brief Process an audio file and extract MFCC features.
         *
         * Loads an audio file, applies windowing and FFT, computes MFCCs for
         * each frame, and returns them as a Torch tensor.
         *
         * @param[in] filePath Path to the audio file.
         * @return torch::Tensor Extracted MFCC features of shape (1, NumFrames, NumMFCC).
         */
        virtual torch::Tensor ProcessFile(const fs::path & filePath);
        /**
         * @brief Normalize features using stored mean and standard deviation.
         *
         * @param[in] x Input tensor of features.
         * @return torch::Tensor Normalized tensor.
         */
        virtual torch::Tensor NormalizeData(const torch::Tensor & x);
    private:
        /**
         * @struct AudioData
         * @brief Holds raw audio samples and their sample rate.
         */
        struct AudioData
        {
            std::vector<float> signal;
            uint sr;

            /**
             * @brief Construct a new AudioData object.
             *
             * @param[in] signal_ Vector of audio samples (moved into the struct).
             * @param[in] sr_ Sampling rate in Hz.
             */
            AudioData(std::vector<float> signal_, uint sr_) : signal(std::move(signal_)), sr(sr_) {}
        };

        /**
         * @brief Silent Aubio log callback.
         */
        static void silent_log(int level, const char * message, void * data);

        /**
         * @brief Load audio file and crop it to the specified size.
         *
         * @param[in] filePath Path to the audio file.
         * @return AudioData Struct containing the waveform samples and sample rate.
         * 
         * @throws std::runtime_error If the file is invalid or corrupted.
         */
        AudioData LoadAndCrop(const fs::path & filePath);

        PreprocessorConfig m_cfg;
        fvec_t * m_windowVector;
        torch::Tensor m_mean;
        torch::Tensor m_std;
};

#endif
