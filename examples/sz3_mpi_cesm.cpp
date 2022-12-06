//
// Created by apple on 2022/12/5.
//

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


//    delete[]data;
//    delete[]bytes;

    return result;
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
    TCLAP::ValueArg<std::string> ebArg("e", "eb", "the error bound value", true, "", "string");
    TCLAP::ValueArg<std::string> confFilePath("c", "conf", "the config file path", true, "", "config path");
    TCLAP::ValueArg<std::string> dataFolderPath("q", "data", "The data folder", false, "", "string");
    TCLAP::ValueArg<std::string> compressedFolderPath("p", "compress", "The compressed folder", false, "", "string");

    cmd1.add(ebArg);
    cmd1.add(confFilePath);
    cmd1.add(compressedFolderPath);
    cmd1.add(dataFolderPath);
    cmd1.parse(argc, argv);

    SZ::Config conf1, conf2;
    conf1 = SZ::Config(26, 1800, 3600);
    conf2 = SZ::Config(1800, 3600);
    if(confFilePath.isSet()) {
        conf1.loadcfg(confFilePath.getValue());
        conf2.loadcfg(confFilePath.getValue());
    }
    float eb = std::stof(ebArg.getValue());
    conf1.absErrorBound = eb;
    conf1.errorBoundMode = SZ::EB_ABS;
    conf2.absErrorBound = eb;
    conf2.errorBoundMode = SZ::EB_ABS;


    double wtime;
    std::string dirpath = dataFolderPath.getValue();
    std::vector<std::string> filenames, filenames_26, dirs;
    if(world_rank==0) {
        wtime = MPI_Wtime();
    }
    int count = 0;
    for(const auto &entry: fs::recursive_directory_iterator(dirpath)) {
        std::string cur = entry.path();
        if(cur.find("26x1800x3600") != std::string::npos) {
            filenames_26.push_back(cur);
        } else if(cur.find("26x") == std::string::npos && cur.find("1800x3600") != std::string::npos) {
            filenames.push_back(cur);
        } else {
            int slash1 = cur.rfind('/');
            dirs.push_back(cur.substr(slash1 + 1));
        }
        std::cout << cur << std::endl;
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
//        std::string dirname = file_path_str.substr(0, filename.find('/'));
//        std::string write_dir = compressedFolderPath.getValue() + dirname;
//        if(!fs::exists(write_dir)) {
//            fs::create_directory(write_dir);
//        }
        auto data = SZ::readfile<float>(file_path_str.c_str(), num);
        conf2.num = num;
        std::string compressed_file = compressedFolderPath.getValue() + filename + ".sz3";
        auto cp_result = compress<float>(data.get(), compressed_file.c_str() , conf2);
        printf("Part 1 1800x3600- My rank is %d, dealing with %s, saving to %s, compression time: %lf, compression ratio: %lf\n",
               world_rank, filename.c_str(), compressed_file.c_str(), cp_result.CPTime, cp_result.CR);
    }
    for(int i=world_rank;i<filenames_26.size();i+=world_size) {
        std::string file_path_str = filenames_26[i];
        std::string filename = file_path_str.substr(file_path_str.rfind('/', file_path_str.rfind('/') - 1) + 1);
//        std::string dirname = file_path_str.substr(0, filename.find('/'));
//        std::string write_dir = compressedFolderPath.getValue() + dirname;
//        if(!fs::exists(write_dir)) {
//            fs::create_directory(write_dir);
//        }
        auto data = SZ::readfile<float>(file_path_str.c_str(), num);
        conf1.num = num;
        std::string compressed_file = compressedFolderPath.getValue() + filename + ".sz3";
        auto cp_result = compress<float>(data.get(), compressed_file.c_str() , conf1);
        printf("Part 2 26x1800x3600- My rank is %d, dealing with %s, saving to %s, compression time: %lf, compression ratio: %lf\n",
               world_rank, filename.c_str(), compressed_file.c_str(), cp_result.CPTime, cp_result.CR);
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