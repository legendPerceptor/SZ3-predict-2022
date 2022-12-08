#include <mpi.h>
#include <iostream>
#include <cstdio>
#include <string>
#include <cstdlib>
#include <vector>
#include <filesystem>

#include <memory>

#include "SZ3/api/sz.hpp"
#include "SZ3/utils/Iterator.hpp"
#include "SZ3/utils/FileUtil.hpp"
#include "qcat_dataAnalysis.h"

#include <tclap/CmdLine.h>
#include "csv.hpp"

namespace fs = std::filesystem;

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

    double quant_entropy = calculateQuantizationEntroy(quant_inds, r*2, nble);
    clock_gettime(CLOCK_REALTIME, &end);
    double overhead_time = (double) (end.tv_sec - start.tv_sec) +
                           (double) (end.tv_nsec - start.tv_nsec) / (double) 1000000000;


    LorenzoResult result = {.avg_err = avg_err,
            .predict_cr = 32/prediction,
            .predict_bitrate=prediction,
            .quant_entropy = quant_entropy,
            .overhead_time = overhead_time,
            .p0 = p_0,
            .P0 = P_0};

    return result;
}

struct CompressionResult {
    double CR;
    double CPTime;
    std::unique_ptr<char> cpdata;
    size_t outsize;
};

struct DecompressionResult {
    double PSNR;
    double RMSE;
    double DPTime;
    double WriteTime;
};

template<class T>
CompressionResult compress(T* data, const char *cmpPath, SZ::Config conf) {

    size_t outSize;
    SZ::Timer timer(true);
    std::unique_ptr<char> bytes(SZ_compress<T>(conf, data, outSize));
    double compress_time = timer.stop();

    char outputFilePath[1024];
    if (cmpPath == nullptr) {
        sprintf(outputFilePath, "tmp.sz");
    } else {
        strcpy(outputFilePath, cmpPath);
    }
    SZ::writefile(outputFilePath, bytes.get(), outSize);

    CompressionResult result;
    result.CR = conf.num * 1.0 * sizeof(T) / outSize;
    result.CPTime=compress_time;
    result.cpdata = std::move(bytes);
    result.outsize = outSize;

//    printf("compression ratio = %.2f \n", conf.num * 1.0 * sizeof(T) / outSize);
//    printf("compression time = %f\n", compress_time);
//    printf("compressed data file = %s\n", outputFilePath);

//    delete[]data;
//    delete[]bytes;

    return result;
}

template<class T>
DecompressionResult decompress(float* ori_data, char *cmpData, size_t cmpSize, const char *decPath,
                               SZ::Config conf,
                               int binaryOutput) {


    SZ::Timer timer(true);
    std::unique_ptr<T> decData(SZ_decompress<T>(conf, cmpData, cmpSize));
    double compress_time = timer.stop();
    char outputFilePath[1024];
    if (decPath == nullptr) {
        sprintf(outputFilePath, "tmpsz.out");
    } else {
        strcpy(outputFilePath, decPath);
    }
    SZ::Timer timer2(true);
    if (binaryOutput == 1) {
        SZ::writefile<T>(outputFilePath, decData.get(), conf.num);
    } else {
        SZ::writeTextFile<T>(outputFilePath, decData.get(), conf.num);
    }
    double write_time = timer2.stop();

    DecompressionResult dpresult;
    dpresult.DPTime = compress_time;
    dpresult.WriteTime = write_time;
    SZ::verify<T>(ori_data, decData.get(), conf.num, dpresult.PSNR, dpresult.RMSE);
//    delete[]decData;
//    printf("compression ratio = %f\n", conf.num * sizeof(T) * 1.0 / cmpSize);
    return dpresult;
}


int main(int argc, char** argv) {
    // Initialize the MPI environment
    if(argc < 2){
        printf("Please provide information for the parallel compression\n");
        exit(0);
    }

    TCLAP::CmdLine cmd1("SZ3 Parallel Compression", ' ', "0.1");
    TCLAP::ValueArg<std::string> ebArg("e", "ebs", "the error bounds to test", true, "", "string");
    TCLAP::ValueArg<std::string> confFilePath("c", "conf", "the config file path", true, "", "config path");
    TCLAP::ValueArg<std::string> dimensionArg("d", "dimension", "the dimention of data", true, "", "dimension");
    TCLAP::ValueArg<std::string> dataFolderPath("q", "data", "The data folder", true, "", "string");
    TCLAP::ValueArg<std::string> compressedFolderPath("p", "compress", "The compressed folder", true, "", "string");
    TCLAP::ValueArg<std::string> csvFolderPath("v", "csvfolder", "The csv files folder", true, "", "csv folder");
    TCLAP::ValueArg<std::string> decompressedFolderPath("o", "decompress", "The decompressed files folder", true, "", "decompressed folder");

    cmd1.add(dimensionArg);
    cmd1.add(ebArg);
    cmd1.add(confFilePath);
    cmd1.add(compressedFolderPath);
    cmd1.add(dataFolderPath);
    cmd1.add(csvFolderPath);
    cmd1.add(decompressedFolderPath);
    cmd1.parse(argc, argv);
    std::vector<size_t> dims;
    std::string csvfolder = csvFolderPath.getValue();
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
    }
    std::string ebsString = ebArg.getValue();
    std::vector<float> ebs;
    std::vector<std::string>ebs_str;
    {
        std::stringstream ss;
        ss << ebsString;
        std::string ebstr;
        while(ss >> ebstr) {
            ebs_str.push_back(ebstr);
        }
    }
    {
        std::stringstream ss;
        ss << ebsString;
        float cur_eb;
        while(ss >> cur_eb) {
            ebs.push_back(cur_eb);
        }
    }
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
//    float eb = std::stof(ebArg.getValue());
//    conf.absErrorBound = eb;
    conf.errorBoundMode = SZ::EB_ABS;

    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int num_of_files, num_of_ebs;
    double wtime;
    std::string dirpath = dataFolderPath.getValue();
    std::vector<std::string> filenames;
    if(world_rank==0) {
        wtime = MPI_Wtime();
    }
    for (const auto & entry : fs::directory_iterator(dirpath)) {
        filenames.push_back(entry.path());
    }
    num_of_files = filenames.size();
    num_of_ebs = ebs.size();
    // 5 * 3  m * n
    // 0 1 2  3 4 5  6 7 8   9 10 11  12 13 14
    size_t num;
    LorenzoResult lorenzoResult;
    int total_workers = num_of_files * num_of_ebs;
    for(int i=world_rank; i<total_workers; i+=world_size) {
        int file_index = i / num_of_ebs;
        int eb_index = i % num_of_ebs;
        float eb = ebs[eb_index];
        std::string file_path_str = filenames[file_index];
        std::string filename = file_path_str.substr(file_path_str.rfind('/') + 1);
        auto data = SZ::readfile<float>(file_path_str.c_str(), num);
        QCAT_DataProperty* property = computeProperty(QCAT_FLOAT, data.get(), num);
        struct timespec start, end;
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
        double overhead_time = (double) (end.tv_sec - start.tv_sec) +
                               (double) (end.tv_nsec - start.tv_nsec) / (double) 1000000000;

        conf.num = num;
        conf.absErrorBound = eb;
        std::string compressed_file = compressedFolderPath.getValue() + filename + ".sz3";
        std::string csv_file = csvfolder + filename + ebs_str[eb_index] + ".csv";
        std::string decompressed_folder = decompressedFolderPath.getValue() + filename + ".dp";
        auto cp_result = compress<float>(data.get(), compressed_file.c_str() , conf);
        auto ori_data = SZ::readfile<float>(file_path_str.c_str(), num);
        auto dp_result = decompress<float>(ori_data.get(), cp_result.cpdata.get(), cp_result.outsize, decompressed_folder.c_str(), conf, 1);
        printf("My rank is %d, dealing with %s, saving to %s, compression time: %lf, compression ratio: %lf\n",
               world_rank, filename.c_str(), compressed_file.c_str(), cp_result.CPTime, cp_result.CR);
        std::stringstream ss;
        auto writer = csv::make_csv_writer(ss);
        writer << std::vector<std::string>({"filename", "size", "num", "min", "max", "valueRange","avgValue", "entropy", "zeromean_variance", "total_overhead_time", "total_overhead_percentage",
                                            "prediction_overhead_time", "prediction_overhead_percentage", "ABS Error Bound", "Set Error Bound",
                                            "avg_lorenzo", "quant_entropy", "predicted CR", "predicted bitrate", "p0", "P0", "CPTime", "CR","DPTime","WriteTime","PSNR","RMSE"});
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
                                            std::to_string(overhead_time/cp_result.CPTime),
                                            std::to_string(lorenzoResult.overhead_time),
                                            std::to_string(lorenzoResult.overhead_time / cp_result.CPTime),
                                            std::to_string(eb),
                                            ebArg.getValue(),
                                            std::to_string(lorenzoResult.avg_err),
                                            std::to_string(lorenzoResult.quant_entropy),
                                            std::to_string(lorenzoResult.predict_cr),
                                            std::to_string(lorenzoResult.predict_bitrate),
                                            std::to_string(lorenzoResult.p0),
                                            std::to_string(lorenzoResult.P0),
                                            std::to_string(cp_result.CPTime),
                                            std::to_string(cp_result.CR),
                                            std::to_string(dp_result.DPTime),
                                            std::to_string(dp_result.WriteTime),
                                            std::to_string(dp_result.PSNR),
                                            std::to_string(dp_result.RMSE)});
        std::ofstream fout(csv_file, std::ios::out);
        fout << ss.str() << std::endl;
    }


    MPI_Barrier(MPI_COMM_WORLD);
    if(world_rank==0){
        printf("Finished all tasks! Total Time: %lf\n", MPI_Wtime() - wtime);
    }
    // Get the name of the processor
    // char processor_name[MPI_MAX_PROCESSOR_NAME];
    // int name_len;
    // MPI_Get_processor_name(processor_name, &name_len);

    // // Print off a hello world message
    // printf("Hello world from processor %s, rank %d"
    //        " out of %d processors\n", processor_name, world_rank, world_size);

    // Finalize the MPI environment.
    MPI_Finalize();
}