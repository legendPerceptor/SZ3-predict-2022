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
CompressionResult compress(T* data, SZ::Config conf) {
    size_t outSize;
    SZ::Timer timer(true);
    std::unique_ptr<char> bytes(SZ_compress<T>(conf, data, outSize));
    double compress_time = timer.stop();
    CompressionResult result;
    result.CR = conf.num * 1.0 * sizeof(T) / outSize;
    result.CPTime=compress_time;
    result.cpdata = std::move(bytes);
    result.outsize = outSize;
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

    MPI_Status status;
    size_t num;

    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, compressedFolderPath.getValue().c_str(),
                  MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    std::vector<size_t> file_size_after_compression(num_of_files, 0);
    std::vector<bool> file_size_populated(num_of_files, false);
    size_t round = 0;
    for(size_t i=world_rank;i<num_of_files;i+=world_size) {
        std::string file_path_str = filenames[i];
        std::string filename = file_path_str.substr(file_path_str.rfind('/') + 1);
        auto data = SZ::readfile<float>(file_path_str.c_str(), num);
        conf.num = num;
        auto cp_result = compress<float>(data.get(), conf);
        size_t compressed_size = cp_result.outsize;
        file_size_after_compression[i] = compressed_size;
        size_t send_data[2] = {i, compressed_size};
        MPI_Bcast(send_data, 2, MPI_UNSIGNED, world_rank, MPI_COMM_WORLD);
        size_t receive_data[2];
        for(size_t j = 0; j < world_size; j++) {
            if(j == world_rank) continue;
            if(j + round * world_size > num_of_files) continue;
            MPI_Bcast(receive_data, 2, MPI_UNSIGNED, j, MPI_COMM_WORLD);
            file_size_after_compression[receive_data[0]] = receive_data[1];
            file_size_populated[receive_data[0]] = true;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        int start_location=0;
        for(size_t j=0;j<i;j++){
            if(file_size_populated[j] == false) {
                printf("Significant Error! File %d's size is unknown in rank %d! Should Exit!\n", j, world_rank);
            }
            start_location += file_size_after_compression[j];
        }
        MPI_Status mpi_status;
        MPI_File_write_at(fh, start_location, cp_result.cpdata.get(), compressed_size, MPI_BYTE, &mpi_status);

        printf("My rank is %d, dealing with No. %zu file, writing offset: %d, compressed_size: %zu, compression time: %lf, compression ratio: %lf\n",
               world_rank, i, start_location, compressed_size, cp_result.CPTime, cp_result.CR);
        cp_result.cpdata.reset(nullptr);
        round++;
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