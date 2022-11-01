#ifndef SZ_GENERAL_COMPRESSOR_HPP
#define SZ_GENERAL_COMPRESSOR_HPP

#include "SZ3/compressor/Compressor.hpp"
#include "SZ3/frontend/Frontend.hpp"
#include "SZ3/encoder/Encoder.hpp"
#include "SZ3/lossless/Lossless.hpp"
#include "SZ3/utils/FileUtil.hpp"
#include "SZ3/utils/Config.hpp"
#include "SZ3/utils/Timer.hpp"
#include "SZ3/def.hpp"
#include <cstring>

namespace SZ {
    template<class T, uint N, class Frontend, class Encoder, class Lossless>
    class SZGeneralCompressor : public concepts::CompressorInterface<T> {
    public:


        SZGeneralCompressor(Frontend frontend, Encoder encoder, Lossless lossless) :
                frontend(frontend), encoder(encoder), lossless(lossless) {
            static_assert(std::is_base_of<concepts::FrontendInterface<T, N>, Frontend>::value,
                          "must implement the frontend interface");
            static_assert(std::is_base_of<concepts::EncoderInterface<int>, Encoder>::value,
                          "must implement the encoder interface");
            static_assert(std::is_base_of<concepts::LosslessInterface, Lossless>::value,
                          "must implement the lossless interface");
        }

        uchar *compress(const Config &conf, T *data, size_t &compressed_size) {

//            printf("compress start\n");
            std::vector<int> quant_inds = frontend.compress(data);

            printf("exporting the quantization bins");
            std::ofstream output_file("./quant_distribution.dat", std::ios::binary | std::ios::out);
            int quant_size = quant_inds.size();
            int magic_number = 2051;
            output_file.write((char*) &magic_number, sizeof(int));
            output_file.write((char*)&quant_size, sizeof(int));
            output_file.write((char*)&quant_inds[0], sizeof(int) * quant_size);
            output_file.close();

//            printf("compress 001\n");
            encoder.preprocess_encode(quant_inds, 0);
            size_t bufferSize = 1.2 * (frontend.size_est() + encoder.size_est() + sizeof(T) * quant_inds.size());

//            printf("preprocess encode end\n");
            uchar *buffer = new uchar[bufferSize];
            uchar *buffer_pos = buffer;

            frontend.save(buffer_pos);

//            printf("frontend save end\n");
            encoder.save(buffer_pos);
            encoder.encode(quant_inds, buffer_pos);
            encoder.postprocess_encode();

            assert(buffer_pos - buffer < bufferSize);

            uchar *lossless_data = lossless.compress(buffer, buffer_pos - buffer, compressed_size);
            lossless.postcompress_data(buffer);

//            printf("compress end\n");
            return lossless_data;
        }

        T *decompress(uchar const *cmpData, const size_t &cmpSize, size_t num) {
            T *dec_data = new T[num];
            return decompress(cmpData, cmpSize, dec_data);
        }

        T *decompress(uchar const *cmpData, const size_t &cmpSize, T *decData) {
            size_t remaining_length = cmpSize;

            Timer timer(true);
            auto compressed_data = lossless.decompress(cmpData, remaining_length);
            uchar const *compressed_data_pos = compressed_data;
//            timer.stop("Lossless");

            frontend.load(compressed_data_pos, remaining_length);

            encoder.load(compressed_data_pos, remaining_length);

            timer.start();
            auto quant_inds = encoder.decode(compressed_data_pos, frontend.get_num_elements());
            encoder.postprocess_decode();
//            timer.stop("Decoder");

            lossless.postdecompress_data(compressed_data);

            timer.start();
            frontend.decompress(quant_inds, decData);
//            timer.stop("Prediction & Recover");
            return decData;
        }


    private:
        Frontend frontend;
        Encoder encoder;
        Lossless lossless;
    };

    template<class T, uint N, class Frontend, class Encoder, class Lossless>
    std::shared_ptr<SZGeneralCompressor<T, N, Frontend, Encoder, Lossless>>
    make_sz_general_compressor(Frontend frontend, Encoder encoder, Lossless lossless) {
        return std::make_shared<SZGeneralCompressor<T, N, Frontend, Encoder, Lossless>>(frontend, encoder, lossless);
    }


}
#endif
