#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#ifdef __cplusplus

#include <aubio/aubio.h>
#include <filesystem>
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

        static void silent_log(int level, const char * message, void * data) { /* Do nothing. */ }
        std::vector<float> loadAndCrop(const fs::path& filePath, uint& sampleRate, uint size = 660000);
    public:
        Preprocessor(
            uint size = 660000U,
            uint nfft = 1024U,
            uint hop = 512U,
            uint nmels = 128U,
            uint nmfcc = 13U
        ) : size(size), nfft(nfft), hop(hop), nmels(nmels), nmfcc(nmfcc) {
            windowVector = new_aubio_window((char*)("hanning"), nfft);
        }
        ~Preprocessor() {
            del_fvec(windowVector);
            aubio_cleanup();
        }

        virtual std::vector<std::vector<float>> run(const fs::path& filePath);
};

#endif

#endif
