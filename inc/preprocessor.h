#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#ifdef __cplusplus

#include <aubio/aubio.h>
#include <filesystem>
#include <torch/torch.h>
#include <vector>

namespace fs = std::filesystem;

class Preprocessor {
    private:
        uint size;
        uint nfft;
        uint hop;
        uint nmels;
        uint nmfcc;
        fvec_t * windowVector;
        torch::Tensor mean;
        torch::Tensor std;

        static void silent_log(int level, const char * message, void * data) { /* Do nothing. */ }
        std::vector<float> loadAndCrop(const fs::path& filePath, uint& sampleRate, uint size = 660000);
    public:
        Preprocessor(
            uint size = 660000U,
            uint nfft = 1024U,
            uint hop = 512U,
            uint nmels = 128U,
            uint nmfcc = 13U
        );
        ~Preprocessor() {
            del_fvec(windowVector);
            aubio_cleanup();
        }

        inline torch::Tensor getMean() const { return mean; }
        inline void setMean(torch::Tensor mean) { this->mean = mean; }
        inline torch::Tensor getStd() const { return std; }
        inline void setStd(torch::Tensor std) { this->std = std; }

        virtual torch::Tensor run(const fs::path& filePath);
        virtual torch::Tensor normalize(const torch::Tensor& x);
};

#endif

#endif
