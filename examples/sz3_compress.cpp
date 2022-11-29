//
// Created by apple on 2022/11/28.
//

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <memory>

#include "SZ3/api/sz.hpp"
#include "SZ3/utils/Iterator.hpp"
#include "SZ3/utils/FileUtil.hpp"
#include "qcat_dataAnalysis.h"

#include <tclap/CmdLine.h>
#include "csv.hpp"
#include <string>


struct CompressionResult {
    double CR;
    double CPTime;
    std::unique_ptr<char> cpdata;
    size_t outsize;
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

    printf("compression ratio = %.2f \n", conf.num * 1.0 * sizeof(T) / outSize);
    printf("compression time = %f\n", compress_time);
    printf("compressed data file = %s\n", outputFilePath);

//    delete[]data;
//    delete[]bytes;

    return result;
}

int main(int argc, char**argv) {
    if(argc < 2)
    {
        printf("Usage: printDataProperty [dataType] tgtFilePath]\n");
        printf("Example: printDataProperty -f testfloat_8_8_128.dat\n");
        exit(0);
    }
    TCLAP::CmdLine cmd1("SZ3 Data Property", ' ', "0.1");
    TCLAP::ValueArg<std::string> inputFilePath("f","file", "The input data source file path",false,"","input file");
//    TCLAP::SwitchArg bigEndian("e", "bigendian", "Whether it's big endian", cmd1, false);
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
//    LorenzoResult lorenzoResult;

//    QCAT_DataProperty* property = computeProperty(QCAT_FLOAT, data.get(), num);
    double min=data[0],max=data[0];
    for(int i=0;i<num;i++)
    {
        if(min>data[i]) min = data[i];
        if(max<data[i]) max = data[i];
    }
    float eb;
    if(ebmodeArg.getValue() == "ABS") {
        eb = std::stof(ebArg.getValue());
    } else {
        eb = (max - min) * std::stof(ebArg.getValue());
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
    }
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
    conf.num = num;

    std::string compressed_path = "tmp.sz";
    if(compressedFilePath.isSet()) {
        compressed_path = compressedFilePath.getValue();
    }
    auto cp_result = compress<float>(data.get(), compressed_path.c_str(), conf);

//    auto ori_data = SZ::readfile<float>(inputFilePath.getValue().c_str(), num);
//    DecompressionResult dp_result;
//    dp_result = decompress<float>(ori_data.get(), cp_result.cpdata.get(), cp_result.outsize, NULL, conf, 1);

    std::stringstream ss;
    auto writer = csv::make_csv_writer(ss);
    writer << std::vector<std::string>({"filename", "ABS Error Bound", "Set Error Bound", "CPTime", "CR"});
    std::string filename = inputFilePath.getValue();
    filename = filename.substr(filename.rfind('/') + 1);
    writer << std::vector<std::string>({filename,
                                        std::to_string(eb),
                                        ebArg.getValue(),
                                        std::to_string(cp_result.CPTime),
                                        std::to_string(cp_result.CR)});
    std::cout << ss.str() << std::endl;
    if(csvOutputPath.isSet()){
        std::ofstream fout(csvOutputPath.getValue(), std::ios::out);
        fout << ss.str() << std::endl;
    }
    return 0;
}