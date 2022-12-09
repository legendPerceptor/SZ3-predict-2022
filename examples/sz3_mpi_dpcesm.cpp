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
    return dpresult;
}



int main(int argc, char** argv) {
    // Initialize the MPI environment

    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if(argc < 2){
        printf("Please provide information for the parallel compression\n");
        exit(0);
    }

    TCLAP::CmdLine cmd1("SZ3 Parallel Compression", ' ', "0.1");
    TCLAP::ValueArg<std::string> dataFolderPath("q", "data", "The data folder", false, "", "string");
    TCLAP::ValueArg<std::string> compressedFolderPath("p", "compress", "The decompressed folder", false, "", "string");


    cmd1.add(compressedFolderPath);
    cmd1.add(dataFolderPath);
    cmd1.parse(argc, argv);

    SZ::Config conf;

    double wtime;
    std::string dirpath = dataFolderPath.getValue();
    std::vector<std::string> filenames, dirs;
    if(world_rank==0) {
        wtime = MPI_Wtime();
    }
    int count = 0;
    for(const auto &entry: fs::recursive_directory_iterator(dirpath)) {
        std::string cur = entry.path();
        if(cur.find("1800x3600") != std::string::npos) {
            filenames.push_back(cur);
        } else {
            int slash1 = cur.rfind('/');
            dirs.push_back(cur.substr(slash1 + 1));
        }
        count++;
    }

    for(int i=0;i<dirs.size();i++) {
        std::string write_dir = compressedFolderPath.getValue() + dirs[i];
        if(!fs::exists(write_dir)) {
            fs::create_directory(write_dir);
        }
    }
    size_t num;
    for(int i=world_rank;i<filenames.size();i+=world_size) {
        std::string file_path_str = filenames[i];
        std::string filename = file_path_str.substr(file_path_str.rfind('/', file_path_str.rfind('/') - 1) + 1);
        auto data = SZ::readfile<char>(file_path_str.c_str(), num);

        std::string decompressed_file = compressedFolderPath.getValue() + filename + ".dp";
        auto dp_result = decompress<float>(data.get(), num, decompressed_file.c_str() , conf, 1);
        printf("My rank is %d, dealing with %s, saving to %s, decompression time: %lf, write time: %lf\n",
               world_rank, filename.c_str(), decompressed_file.c_str(), dp_result.DPTime, dp_result.WriteTime);
    }


    MPI_Barrier(MPI_COMM_WORLD);
    if(world_rank==0){
        printf("Finished all tasks! Total Time: %lf\n", MPI_Wtime() - wtime);
    }

    MPI_Finalize();
}