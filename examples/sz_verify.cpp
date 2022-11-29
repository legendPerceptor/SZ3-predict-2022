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


int main(int argc, char**argv) {
    TCLAP::CmdLine cmd1("SZ3 Data Property", ' ', "0.1");
    TCLAP::ValueArg<std::string> oriFilePath("f","original", "The input data source file path",true,"","input file");
    TCLAP::ValueArg<std::string> csvOutputPath("o", "csvOutput", "The Output CSV file", true, "result.csv", "string");
    TCLAP::ValueArg<std::string> decompressedFilePath("d", "decompressed", "The decompressed file", true, "", "string");

    cmd1.add(decompressedFilePath);
    cmd1.add(csvOutputPath);
    cmd1.add(oriFilePath);
    cmd1.parse(argc, argv);
    size_t num = 0, ori_num;
    auto dp_data = SZ::readfile<float>(decompressedFilePath.getValue().c_str(), num);
    auto ori_data = SZ::readfile<float>(oriFilePath.getValue().c_str(), ori_num);
    double PSNR, RMSE;
    SZ::verify<float>(ori_data.get(), dp_data.get(), num, PSNR, RMSE);

    std::stringstream ss;
    auto writer = csv::make_csv_writer(ss);
    writer << std::vector<std::string>({"filename", "PSNR","RMSE"});
    std::string filename = oriFilePath.getValue();
    filename = filename.substr(filename.rfind('/') + 1);
    writer << std::vector<std::string>({filename,
                                        std::to_string(PSNR),
                                        std::to_string(RMSE)});
    std::cout << ss.str() << std::endl;
    if(csvOutputPath.isSet()){
        std::ofstream fout(csvOutputPath.getValue(), std::ios::out);
        fout << ss.str() << std::endl;
    }
    return 0;
}