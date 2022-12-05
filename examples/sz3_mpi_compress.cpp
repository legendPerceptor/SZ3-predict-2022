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
    if(argc < 2){
        printf("Please provide information for the parallel compression\n");
        exit(0);
    }

    TCLAP::CmdLine cmd1("SZ3 Parallel Compression", ' ', "0.1");
    TCLAP::ValueArg<std::string> ebArg("e", "eb", "the error bound value", true, "", "string");
    TCLAP::ValueArg<std::string> confFilePath("c", "conf", "the config file path", true, "", "config path");
    TCLAP::ValueArg<std::string> dimensionArg("d", "dimension", "the dimention of data", false, "", "dimension");
    TCLAP::ValueArg<std::string> dataFolderPath("q", "data", "The data folder", false, "", "string");
    TCLAP::ValueArg<std::string> compressedFolderPath("p", "compress", "The compressed folder", false, "", "string");

    cmd1.add(dimensionArg);
    cmd1.add(ebArg);
    cmd1.add(confFilePath);
    cmd1.add(compressedFolderPath);
    cmd1.add(dataFolderPath);
    cmd1.parse(argc, argv);
    std::vector<size_t> dims;
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
    float eb = std::stof(ebArg.getValue());
    conf.absErrorBound = eb;
    conf.errorBoundMode = SZ::EB_ABS;

    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int num_of_files;
    char filepath[256];
    double wtime;
    if(world_rank==0){
        std::string dirpath = dataFolderPath.getValue();
        std::vector<std::string> filenames;
        printf("Rank 0 Add Files\n");
        wtime = MPI_Wtime();
        for (const auto & entry : fs::directory_iterator(dirpath)) {
            filenames.push_back(entry.path());
        }
        num_of_files = filenames.size();
        // Broadcast the number of files
        MPI_Bcast(&num_of_files, 1, MPI_INT, 0, MPI_COMM_WORLD);
        // MPI_Barrier(MPI_COMM_WORLD);

        for (int i=0;i<num_of_files;i++){
            // printf("length: %d, filename: %s\n", filenames[i].size(), filenames[i].c_str());
            int dst = i % (world_size);
            strcpy(filepath, filenames[i].c_str());
            MPI_Send(filepath, strlen(filepath) + 1, MPI_CHAR, dst, 1, MPI_COMM_WORLD);
        }
    }
    MPI_Bcast(&num_of_files, 1, MPI_INT, 0, MPI_COMM_WORLD);
//    printf("Rank %d has num of files: %d\n", world_rank, num_of_files);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Status status;
    size_t num;
    for(int i=0;i<num_of_files / world_size; i++) {
        MPI_Recv(filepath, 250, MPI_CHAR, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        printf("My rank is %d, dealing with %s\n", world_rank, filepath);
        std::string file_path_str(filepath);
        std::string filename = file_path_str.substr(file_path_str.rfind('/') + 1);
        auto data = SZ::readfile<float>(file_path_str.c_str(), num);
        conf.num = num;
        std::string compressed_file = compressedFolderPath.getValue() + filename + ".sz3";
        auto cp_result = compress<float>(data.get(), compressed_file.c_str() , conf);
        printf("My rank is %d, dealing with %s, saving to %s,\n compression time: %lf, compression ratio: %lf",
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