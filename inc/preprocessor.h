#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#ifdef __cplusplus

#include <aubio/aubio.h>
#include <filesystem>
#include <torch/torch.h>
#include <vector>

namespace fs = std::filesystem;

class Preprocessor
{
    public:
        Preprocessor(
            uint size = 660000U,
            uint nfft = 1024U,
            uint hop = 512U,
            uint nmels = 128U,
            uint nmfcc = 13U
        );
        ~Preprocessor();

        virtual torch::Tensor run(const fs::path & filePath);
        virtual torch::Tensor normalize(const torch::Tensor & x);

        torch::Tensor mean() const;
        void setMean(torch::Tensor mean);
        torch::Tensor std() const;
        void setStd(torch::Tensor std);
    private:
        uint m_size;
        uint m_nfft;
        uint m_hop;
        uint m_nmels;
        uint m_nmfcc;
        fvec_t * m_windowVector;
        torch::Tensor m_mean;
        torch::Tensor m_std;

        std::vector<float> loadAndCrop(const fs::path & filePath, uint & sampleRate, uint size = 660000);
        static void silent_log(int level, const char * message, void * data);
};

#endif

#endif
