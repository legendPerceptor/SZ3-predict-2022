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
};

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

    double prediction = 0;
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
    prediction = prediction / pre_lossless;

//    printf("p_0, P_0: %f %f\n", p_0, P_0);
//    printf("test %d %d %d \n", quant_inds[0], quant_inds[1], quant_inds[2]);
//    printf("num, nupredicted, zero: %d %d %d\n", pre_num, pre_freq[0], pre_freq[512]);
//    printf("test %d\n", quant_inds.size());
//    printf("predicted compression bit-rate: %f %f\n", prediction, 32/prediction);
//
//    std::cout << "the prediction: " << prediction << std::endl;

    LorenzoResult result = {.avg_err = avg_err,
                            .predict_cr = 32/prediction,
                            .predict_bitrate=prediction};


    // Deal with quant

//    sum = sum / nble;
//    std::cout<<"AVG SUM:" << sum << "; avgerr" << avg_err <<std::endl;
    return result;
}

struct CompressionResult {
    double CR;
    double CPTime;
    char *cpdata;
    size_t outsize;
};

struct DecompressionResult {
    double PSNR;
    double RMSE;
    double DPTime;
    double WriteTime;
};

template<class T>
CompressionResult compress(T* data, char *cmpPath, SZ::Config conf) {

    size_t outSize;
    SZ::Timer timer(true);
    char *bytes = SZ_compress<T>(conf, data, outSize);
    double compress_time = timer.stop();

    char outputFilePath[1024];
    if (cmpPath == nullptr) {
        sprintf(outputFilePath, "tmp.sz");
    } else {
        strcpy(outputFilePath, cmpPath);
    }
    SZ::writefile(outputFilePath, bytes, outSize);

    CompressionResult result;
    result.CR = conf.num * 1.0 * sizeof(T) / outSize;
    result.CPTime=compress_time;
    result.cpdata = bytes;
    result.outsize = outSize;

//    printf("compression ratio = %.2f \n", conf.num * 1.0 * sizeof(T) / outSize);
//    printf("compression time = %f\n", compress_time);
//    printf("compressed data file = %s\n", outputFilePath);

//    delete[]data;
//    delete[]bytes;

    return result;
}

template<class T>
DecompressionResult decompress(float* ori_data, char *cmpData, size_t cmpSize, char *decPath,
                SZ::Config conf,
                int binaryOutput) {


    SZ::Timer timer(true);
    T *decData = SZ_decompress<T>(conf, cmpData, cmpSize);
    double compress_time = timer.stop();
    std::cout << "I have finished decompression" << std::endl;
    char outputFilePath[1024];
    if (decPath == nullptr) {
        sprintf(outputFilePath, "tmpsz.out");
    } else {
        strcpy(outputFilePath, decPath);
    }
    SZ::Timer timer2(true);
    if (binaryOutput == 1) {
        SZ::writefile<T>(outputFilePath, decData, conf.num);
    } else {
        SZ::writeTextFile<T>(outputFilePath, decData, conf.num);
    }
    double write_time = timer2.stop();

    DecompressionResult dpresult;
    dpresult.DPTime = compress_time;
    dpresult.WriteTime = write_time;
    SZ::verify<T>(ori_data, decData, conf.num, dpresult.PSNR, dpresult.RMSE);
//    delete[]decData;
//    printf("compression ratio = %f\n", conf.num * sizeof(T) * 1.0 / cmpSize);
//    printf("decompression time = %f seconds.\n", compress_time);
//    printf("decompressed file = %s\n", outputFilePath);
    return dpresult;
}


int main(int argc, char**argv) {
    if(argc < 2)
    {
        printf("Usage: printDataProperty [dataType] tgtFilePath]\n");
        printf("Example: printDataProperty -f testfloat_8_8_128.dat\n");
        exit(0);
    }
    TCLAP::CmdLine cmd1("SZ3 Data Property", ' ', "0.1");
    TCLAP::ValueArg<std::string> inputFilePath("f","file", "The input data source file path",false,"","string");
//    TCLAP::SwitchArg bigEndian("e", "bigendian", "Whether it's big endian", cmd1, false);
    TCLAP::SwitchArg debugArg("b", "debug", "Print additional information", cmd1, false);
    TCLAP::ValueArg<std::string> dimensionArg("d", "dimension", "the dimention of data", false, "", "string");
    TCLAP::ValueArg<std::string> ebmodeArg("m", "ebmode", "the error bound mode - ABS | REL", true, "", "string");
    TCLAP::ValueArg<std::string> ebArg("e", "eb", "the error bound value", true, "", "string");
    TCLAP::ValueArg<std::string> confFilePath("c", "conf", "the config file path", true, "", "string");
    TCLAP::SwitchArg logcalculation("l", "log", "Whether use the log before anything", cmd1, false);

    cmd1.add(inputFilePath);
    cmd1.add(dimensionArg);
    cmd1.add(ebmodeArg);
    cmd1.add(ebArg);
    cmd1.add(confFilePath);
    cmd1.parse(argc, argv);
    bool debug = debugArg.getValue();
    size_t num = 0;
    bool use_log = logcalculation.getValue();
    auto data = SZ::readfile<float>(inputFilePath.getValue().c_str(), num);
//    std::cout<<"num: " << num <<std::endl;
    if(use_log) {
        for(int i=0;i<num;i++) {
            if(data[i] < 0) {
                data[i] = -log10(-data[i]);
            } else if(data[i]>0) {
                data[i] = log10(data[i]);
            }
        }
    }
//    unsigned char* data  = NULL;

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


    auto cp_result = compress<float>(data.get(), NULL, conf);
    auto ori_data = SZ::readfile<float>(inputFilePath.getValue().c_str(), num);
    DecompressionResult dp_result;
    dp_result = decompress<float>(ori_data.get(), cp_result.cpdata, cp_result.outsize, NULL, conf, 1);


    int i = 0;
    if(debug) {
        printf("The first 10 values are: \n");
        for (i = 0; i < 10; i++)
            printf("%f ", data[i]);

        printf("....\n------------------------\n");
    }

    std::stringstream ss;
    auto writer = csv::make_csv_writer(ss);
    writer << std::vector<std::string>({"filename", "size", "num", "min", "max", "valueRange","avgValue", "entropy", "zeromean_variance",
                                        "avg_lorenzo", "predicted CR", "predicted bitrate", "CPTime", "CR","DPTime","WriteTime","PSNR","RMSE"});
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
                                        std::to_string(lorenzoResult.avg_err),
                                        std::to_string(lorenzoResult.predict_cr),
                                        std::to_string(lorenzoResult.predict_bitrate),
                                        std::to_string(cp_result.CPTime),
                                        std::to_string(cp_result.CR),
                                        std::to_string(dp_result.DPTime),
                                        std::to_string(dp_result.WriteTime),
                                        std::to_string(dp_result.PSNR),
                                        std::to_string(dp_result.RMSE)});
    if(debug) {
        printProperty(property);
    }
    free(property);
    std::cout << ss.str() << std::endl;
    return 0;
}