//
// Created by apple on 2022/11/28.
//

//
// Created by apple on 2022/11/28.
//

#include <cstdio>
#include <cstdlib>
#include <memory>

#include "SZ3/api/sz.hpp"
#include "SZ3/utils/Iterator.hpp"
#include "SZ3/utils/FileUtil.hpp"
#include "qcat_dataAnalysis.h"

#include <tclap/CmdLine.h>
#include "csv.hpp"
#include <string>



struct DecompressionResult {
    double DPTime;
    double WriteTime;
};

template<class T>
DecompressionResult decompress(char *cmpData, size_t cmpSize,const char *decPath,
                               SZ::Config conf,
                               int binaryOutput) {


    SZ::Timer timer(true);
    std::unique_ptr<T> decData(SZ_decompress<T>(conf, cmpData, cmpSize));
    double compress_time = timer.stop();
    std::cout << "Decompression finished successfully" << std::endl;
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
    printf("decompression time = %f seconds.\n", compress_time);
    printf("decompressed file = %s\n", outputFilePath);
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
    TCLAP::ValueArg<std::string> csvOutputPath("o", "csvOutput", "The Output CSV file", false, "result.csv", "string");
    TCLAP::ValueArg<std::string> decompressedFilePath("p", "compress", "The compressed file", false, "", "string");

    cmd1.add(inputFilePath);
    cmd1.add(csvOutputPath);
    cmd1.add(decompressedFilePath);
    cmd1.parse(argc, argv);
    size_t num = 0;
    auto data = SZ::readfile<char>(inputFilePath.getValue().c_str(), num);

    // Do the decompression
    SZ::Config conf;
    std::string decompressed_path = "tmp.sz.dp";
    if(decompressedFilePath.isSet()) {
        decompressed_path = decompressedFilePath.getValue();
    }

//    auto ori_data = SZ::readfile<float>(inputFilePath.getValue().c_str(), num);
    DecompressionResult dp_result;
    dp_result = decompress<float>(data.get(), num, decompressed_path.c_str(), conf, 1);

    std::stringstream ss;
    auto writer = csv::make_csv_writer(ss);
    writer << std::vector<std::string>({"filename", "DPTime","WriteTime"});
    std::string filename = inputFilePath.getValue();
    filename = filename.substr(filename.rfind('/') + 1);
    writer << std::vector<std::string>({filename,
                                        std::to_string(dp_result.DPTime),
                                        std::to_string(dp_result.WriteTime)});
    std::cout << ss.str() << std::endl;
    if(csvOutputPath.isSet()){
        std::ofstream fout(csvOutputPath.getValue(), std::ios::out);
        fout << ss.str() << std::endl;
    }
    return 0;
}