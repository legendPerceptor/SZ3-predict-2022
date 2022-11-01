#ifndef SZ3_FRONTEND
#define SZ3_FRONTEND
/**
 * This module is the implementation of general frontend in SZ3
 */

#include "Frontend.hpp"
#include "SZ3/def.hpp"
#include "SZ3/predictor/Predictor.hpp"
#include "SZ3/predictor/LorenzoPredictor.hpp"
#include "SZ3/quantizer/Quantizer.hpp"
#include "SZ3/utils/Iterator.hpp"
#include "SZ3/utils/Config.hpp"
#include "SZ3/utils/MemoryUtil.hpp"

namespace SZ {


    template<class T, uint N, class Predictor, class Quantizer>
    class SZGeneralFrontend : public concepts::FrontendInterface<T, N> {
    public:

        SZGeneralFrontend(const Config &conf, Predictor predictor, Quantizer quantizer) :
                fallback_predictor(LorenzoPredictor<T, N, 1>(conf.absErrorBound)),
                predictor(predictor),
                quantizer(quantizer),
                block_size(conf.blockSize),
                num_elements(conf.num) {
            std::copy_n(conf.dims.begin(), N, global_dimensions.begin());
        }

        ~SZGeneralFrontend() = default;

        std::vector<int> compress(T *data) {
            std::vector<int> quant_inds(num_elements);
            auto block_range = std::make_shared<SZ::multi_dimensional_range<T, N>>(
                    data, std::begin(global_dimensions), std::end(global_dimensions), block_size, 0);

            auto element_range = std::make_shared<SZ::multi_dimensional_range<T, N>>(
                    data, std::begin(global_dimensions), std::end(global_dimensions), 1, 0);

            predictor.precompress_data(block_range->begin());
            quantizer.precompress_data();
            size_t quant_count = 0;
            size_t current_min = 100000;
            size_t current_max = 0;
            int pre_num = 0;
            int pre_unp = 0;
            int pre_0 = 0;
            double prediction = 0;
            ska::unordered_map<T, size_t> pre_freq;
            int ii = 0;

            for (auto block = block_range->begin(); block != block_range->end(); ++block) {

                element_range->update_block_range(block, block_size);

                concepts::PredictorInterface<T, N> *predictor_withfallback = &predictor;
                if (!predictor.precompress_block(element_range)) {
                    predictor_withfallback = &fallback_predictor;
                }

                for (auto element = element_range->begin(); element != element_range->end(); ++element) {
                    ii++;
                    quant_inds[quant_count++] = quantizer.quantize_and_overwrite(
                            *element, predictor_withfallback->predict(element));
                    if (ii % 100 == 0) {
                        pre_num++;
                        pre_freq[quant_inds[quant_count-1]]++;
                    }
                }

            }
/*

            for (auto block = block_range->begin(); block != block_range->end(); ++block) {

                element_range->update_block_range(block, block_size);

                concepts::PredictorInterface<T, N> *predictor_withfallback = &predictor;
                if (!predictor.precompress_block(element_range)) {
                    predictor_withfallback = &fallback_predictor;
                }
                //predictor_withfallback->precompress_block_commit();

                for (auto element = element_range->begin(); element != element_range->end(); ++element) {
                    ii++;
                    if (ii % 100 == 0) {
                    quant_inds[quant_count++] = quantizer.quantize_and_overwrite(
                            *element, predictor_withfallback->predict(element));
                    if (current_min > quant_inds[quant_count-1]) current_min = quant_inds[quant_count-1];
                    if (current_max < quant_inds[quant_count-1]) current_max = quant_inds[quant_count-1];
                    //if (quant_inds[quant_count-1] == 0) pre_unp++;
                    //if (quant_inds[quant_count-1] == 512) pre_0++;
                    pre_num++;
                    pre_freq[quant_inds[quant_count-1]]++;
                    }
                }
            }
*/

            predictor.postcompress_data(block_range->begin());
            quantizer.postcompress_data();

            float temp_bit = 0;
            float p_0 = (float)pre_freq[512]/pre_num;
            float P_0;
            float C_1 = 1.0;
            float pre_lossless = 1.0;
            for (int i = 1; i < 1023; i++) {
                if (pre_freq[i] != 0) {
                    temp_bit = -log2((float)pre_freq[i]/pre_num);
                    //printf("%f %d\n", temp_bit, i);
                    if (temp_bit < 32) {
                        if (temp_bit < 1) {
//                            printf("layer: %d %f\n", i, temp_bit);
                            if (i == 512) prediction += ((float)pre_freq[i]/pre_num) * 1;
                            else if (i == 511) prediction += ((float)pre_freq[i]/pre_num) * 2.5;
                            else if (i == 513) prediction += ((float)pre_freq[i]/pre_num) * 2.5;
                            else prediction += ((float)pre_freq[i]/pre_num) * 4;
                        }
                        else
                            prediction += ((float)pre_freq[i]/pre_num) * temp_bit;
                    }
                }
            }
            if (pre_freq[0] != 0) 
                prediction += ((float)pre_freq[0]/pre_num) * 32;
            if (pre_freq[512] > pre_num/2)
                P_0 = (((float)pre_freq[512]/pre_num) * 1.0)/prediction;
            else
                P_0 = -(((float)pre_freq[512]/pre_num) * log2((float)pre_freq[512]/pre_num))/prediction;

            pre_lossless = 1 / (C_1 * (1 - p_0) * P_0 + (1 - P_0));
            if (pre_lossless < 1) pre_lossless = 1;

//            printf("p_0, P_0: %f %f\n", p_0, P_0);
//            printf("test %d %d %d \n", quant_inds[0], quant_inds[1], quant_inds[2]);
//            printf("minmax: %d, %d\n", current_min, current_max);
//            printf("num, nupredicted, zero: %d %d %d\n", pre_num, pre_freq[0], pre_freq[512]);
//            printf("test %d\n", quant_inds.size());
//            printf("predicted compression bit-rate: %f %f\n", prediction, 32/prediction);
    
            prediction = prediction / pre_lossless;
//            printf("with lossless: %f %f %f\n", pre_lossless, prediction, 32/prediction);


            printf("Predicted compression ratio: %f\n", 32/prediction);
            printf("Predicted compression bit-rate: %f\n", prediction);
//            exit(0);

            return quant_inds;
        }

        T *decompress(std::vector<int> &quant_inds, T *dec_data) {

            int const *quant_inds_pos = (int const *) quant_inds.data();
            std::array<size_t, N> intra_block_dims;
//            auto dec_data = new T[num_elements];
            auto block_range = std::make_shared<SZ::multi_dimensional_range<T, N>>(
                    dec_data, std::begin(global_dimensions), std::end(global_dimensions), block_size, 0);

            auto element_range = std::make_shared<SZ::multi_dimensional_range<T, N>>(
                    dec_data, std::begin(global_dimensions), std::end(global_dimensions), 1, 0);

            predictor.predecompress_data(block_range->begin());
            quantizer.predecompress_data();

            for (auto block = block_range->begin(); block != block_range->end(); ++block) {

                element_range->update_block_range(block, block_size);

                concepts::PredictorInterface<T, N> *predictor_withfallback = &predictor;
                if (!predictor.predecompress_block(element_range)) {
                    predictor_withfallback = &fallback_predictor;
                }
                for (auto element = element_range->begin(); element != element_range->end(); ++element) {
                    *element = quantizer.recover(predictor_withfallback->predict(element), *(quant_inds_pos++));
                }
            }
            predictor.postdecompress_data(block_range->begin());
            quantizer.postdecompress_data();
            return dec_data;
        }

        void save(uchar *&c) {
            write(global_dimensions.data(), N, c);
            write(block_size, c);

            predictor.save(c);
            quantizer.save(c);
        }

        void load(const uchar *&c, size_t &remaining_length) {
            read(global_dimensions.data(), N, c, remaining_length);
            num_elements = 1;
            for (const auto &d: global_dimensions) {
                num_elements *= d;
            }
            read(block_size, c, remaining_length);
            predictor.load(c, remaining_length);
            quantizer.load(c, remaining_length);
        }

        size_t size_est() {
            return quantizer.size_est();
        }

        void print() {
//            predictor.print();
//            quantizer.print();
        }

        void clear() {
            predictor.clear();
            fallback_predictor.clear();
            quantizer.clear();
        }

        int get_radius() const { return quantizer.get_radius(); }

        size_t get_num_elements() const { return num_elements; };

    private:
        Predictor predictor;
        LorenzoPredictor<T, N, 1> fallback_predictor;
        Quantizer quantizer;
        uint block_size;
        size_t num_elements;
        std::array<size_t, N> global_dimensions;

    };

    template<class T, uint N, class Predictor, class Quantizer>
    SZGeneralFrontend<T, N, Predictor, Quantizer>
    make_sz_general_frontend(const Config &conf, Predictor predictor, Quantizer quantizer) {
        return SZGeneralFrontend<T, N, Predictor, Quantizer>(conf, predictor, quantizer);
    }
}

#endif
