//
// Created by apple on 2022/11/28.
//
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <memory>

#include "SZ3/api/sz.hpp"
#include "SZ3/predictor/LorenzoPredictor.hpp"
#include "SZ3/utils/Iterator.hpp"
#include "SZ3/utils/FileUtil.hpp"
#include "qcat_dataAnalysis.h"

#include <tclap/CmdLine.h>
#include "qcat_dataAnalysis.h"
#include "csv.hpp"
#include <string>


struct LorenzoResult {
    double avg_err;
    double predict_cr;
    double predict_bitrate;
    double quant_entropy;
    double overhead_time;
    double p0;
    double P0;
};

double calculateQuantizationEntroy(const std::vector<int>& quant_inds, int binNumber, int nble){
    std::vector<int> bucket(binNumber, 0);
    for(auto iter=quant_inds.begin();iter!=quant_inds.end();iter++){
        ++bucket[*iter];
    }
    double entVal=0;
    for(auto iter = bucket.begin();iter!=bucket.end();iter++) {
        if(*iter != 0) {
            double prob = double(*iter) / nble;
            entVal -= prob * log(prob) / log(2);
        }
    }
    return entVal;
}

template<uint N>
LorenzoResult lorenzo_test(float* data, std::array<size_t, N> dims, float eb=1e-6, int r = 512){
    auto P_l = std::make_shared<SZ::LorenzoPredictor<float, N, 1>>(eb);
    auto quantizer = std::make_shared<SZ::LinearQuantizer<float>>(eb, r);
    int block_size = 6;
    auto element_range = std::make_shared<SZ::multi_dimensional_range<float, N>>(data, std::begin(dims),std::end(dims), 1,0);
    auto block_range = std::make_shared<SZ::multi_dimensional_range<float, N>>(data, std::begin(dims),std::end(dims), block_size,0);
    double avg_err = 0;
    int nble = 1;
    for(int i=0;i<N;i++){
        nble *= dims[i];
    }
//    double sum = 0;

    ska::unordered_map<int, size_t> pre_freq;
    int ii = 0;
    int pre_num = 0;

    std::vector<int> quant_inds(nble);
    int quant_count = 0;

    for (auto block = block_range->begin(); block != block_range->end(); ++block) {
        element_range->update_block_range(block, block_size);
        for (auto element = element_range->begin(); element != element_range->end(); ++element) {
            float org = *element;
            ii++;
            quant_inds[quant_count++] = quantizer->quantize_and_overwrite(
                    *element, P_l->predict(element));

            if (ii % 100 == 0) {
                pre_num++;
                pre_freq[quant_inds[quant_count-1]]++;
            }

//            float cur_err = P_l->estimate_error(element);
            float cur_err = fabs(*element - org);
//            sum += cur_err;
            avg_err += cur_err / (double) nble;
        }
    }

    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);
    double prediction = 0;
    float temp_bit = 0;
    float p_0 = (float)pre_freq[512]/pre_num;
    float P_0;
    float C_1 = 1.0;
    float pre_lossless = 1.0;
    for (int i = 1; i < 1023; i++) {
        if (pre_freq[i] != 0) {
            temp_bit = -log2((float)pre_freq[i]/pre_num);
            if (temp_bit < 32) {
                if (temp_bit < 1) {
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
    prediction = prediction / pre_lossless;

    std::cout << "Calculating quantization entropy updated!!" << std::endl;
    double quant_entropy = calculateQuantizationEntroy(quant_inds, r*2, nble);
    clock_gettime(CLOCK_REALTIME, &end);
    double overhead_time = (double) (end.tv_sec - start.tv_sec) +
                           (double) (end.tv_nsec - start.tv_nsec) / (double) 1000000000;
    printf("predicted compression bit-rate: %f %f\n", prediction, 32/prediction);
    LorenzoResult result = {.avg_err = avg_err,
            .predict_cr = 32/prediction,
            .predict_bitrate=prediction,
            .quant_entropy = quant_entropy,
            .overhead_time = overhead_time,
            .p0 = p_0,
            .P0 = P_0};

    return result;
}

int main(int argc, char**argv) {
    TCLAP::CmdLine cmd1("SZ3 predict", ' ', "0.1");
    TCLAP::ValueArg<std::string> inputFilePath("f","file", "The input data source file path",false,"","input file");
    TCLAP::SwitchArg debugArg("b", "debug", "Print additional information", cmd1, false);
    TCLAP::ValueArg<std::string> dimensionArg("d", "dimension", "the dimention of data", false, "", "dimension");
    TCLAP::ValueArg<std::string> ebmodeArg("m", "ebmode", "the error bound mode - ABS | REL", true, "", "eb mode");
    TCLAP::ValueArg<std::string> ebArg("e", "eb", "the error bound value", true, "", "string");
    TCLAP::ValueArg<std::string> confFilePath("c", "conf", "the config file path", true, "", "config path");
    TCLAP::ValueArg<std::string> csvOutputPath("o", "csvOutput", "The Output CSV file", false, "result.csv", "csv output");
    TCLAP::ValueArg<std::string> compressedFilePath("p", "compress", "The compressed file", false, "", "string");
    TCLAP::SwitchArg logcalculation("l", "log", "Whether use the log before anything", cmd1, false);

    cmd1.add(inputFilePath);
    cmd1.add(dimensionArg);
    cmd1.add(ebmodeArg);
    cmd1.add(ebArg);
    cmd1.add(confFilePath);
    cmd1.add(csvOutputPath);
    cmd1.add(compressedFilePath);
    cmd1.parse(argc, argv);
    bool debug = debugArg.getValue();
    size_t num = 0;
    bool use_log = logcalculation.getValue();
    auto data = SZ::readfile<float>(inputFilePath.getValue().c_str(), num);
    if(use_log) {
        for(int i=0;i<num;i++) {
            if(data[i] < 0) {
                data[i] = -log10(-data[i]);
            } else if(data[i]>0) {
                data[i] = log10(data[i]);
            }
        }
    }
    std::vector<size_t> dims;
    LorenzoResult lorenzoResult;
    QCAT_DataProperty* property = computeProperty(QCAT_FLOAT, data.get(), num);

    float eb;
    if(ebmodeArg.getValue() == "ABS") {
        eb = std::stof(ebArg.getValue());
    } else {
        eb = (property->maxValue - property->minValue) * std::stof(ebArg.getValue());
    }
    struct timespec start, end;
    if(dimensionArg.isSet()) {
        std::string dimsString = dimensionArg.getValue();
        {
            std::stringstream ss;
            ss << dimsString;
            int tmp_dim;
            while(ss >> tmp_dim) {
                dims.push_back(tmp_dim);
            }
        }
        clock_gettime(CLOCK_REALTIME, &start);
        if(dims.size()==3){
            std::array<size_t, 3> _dims = {{dims[0], dims[1], dims[2]}};
            lorenzoResult = lorenzo_test<3>(data.get(), _dims, eb);
        }else if(dims.size()==2) {
            std::array<size_t, 2> _dims = {{dims[0], dims[1]}};
            lorenzoResult = lorenzo_test<2>(data.get(), _dims, eb);
        }else if(dims.size()==1) {
            std::array<size_t, 1> _dims = {{dims[0]}};
            lorenzoResult = lorenzo_test<1>(data.get(), _dims, eb);
        }
        clock_gettime(CLOCK_REALTIME, &end);
        if(debug) {
            std::cout << "Avg Error in Lorenzo: " << lorenzoResult.avg_err << std::endl;
            std::cout << "Predicted compression ratio: " << lorenzoResult.predict_cr << std::endl;
            std::cout << "Predicted bitrate: " << lorenzoResult.predict_bitrate << std::endl;
        }
    }
    double overhead_time = (double) (end.tv_sec - start.tv_sec) +
                           (double) (end.tv_nsec - start.tv_nsec) / (double) 1000000000;
    // Do the actual compression
    SZ::Config conf;
    if (dims.size()==1) {
        conf = SZ::Config(dims[0]);
    } else if (dims.size()==2) {
        conf = SZ::Config(dims[1], dims[0]);
    } else if (dims.size()==3) {
        conf = SZ::Config(dims[2], dims[1], dims[0]);
    } else {
        conf = SZ::Config(dims[3], dims[2], dims[1], dims[0]);
    }
    if(confFilePath.isSet()) {
        conf.loadcfg(confFilePath.getValue());
    }
    conf.absErrorBound = eb;
    conf.errorBoundMode = SZ::EB_ABS;
    conf.num = property->numOfElem;

    std::stringstream ss;
    auto writer = csv::make_csv_writer(ss);
    writer << std::vector<std::string>({"filename", "size", "num", "min", "max", "valueRange","avgValue", "entropy", "zeromean_variance", "total_overhead_time",
                                        "prediction_overhead_time", "ABS Error Bound", "Set Error Bound",
                                        "avg_lorenzo", "quant_entropy", "predicted CR", "predicted bitrate", "p0", "P0"});
    std::string filename = inputFilePath.getValue();
    filename = filename.substr(filename.rfind('/') + 1);
    writer << std::vector<std::string>({filename,
                                        std::to_string(property->totalByteSize),
                                        std::to_string(property->numOfElem),
                                        std::to_string(property->minValue),
                                        std::to_string(property->maxValue),
                                        std::to_string(property->valueRange),
                                        std::to_string(property->avgValue),
                                        std::to_string(property->entropy),
                                        std::to_string(property->zeromean_variance),
                                        std::to_string(overhead_time),
                                        std::to_string(lorenzoResult.overhead_time),
                                        std::to_string(eb),
                                        ebArg.getValue(),
                                        std::to_string(lorenzoResult.avg_err),
                                        std::to_string(lorenzoResult.quant_entropy),
                                        std::to_string(lorenzoResult.predict_cr),
                                        std::to_string(lorenzoResult.predict_bitrate),
                                        std::to_string(lorenzoResult.p0),
                                        std::to_string(lorenzoResult.P0)});
    std::cout << ss.str() << std::endl;
    if(csvOutputPath.isSet()){
        std::ofstream fout(csvOutputPath.getValue(), std::ios::out);
        fout << ss.str() << std::endl;
    }
    return 0;
}