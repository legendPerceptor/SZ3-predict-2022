#include <mpi.h>
#include <iostream>
#include <cstdio>
#include <string>
#include <cstdlib>
#include <vector>
#include <filesystem>

#include <memory>

#include "SZ3/api/sz.hpp"
#include "qcat_dataAnalysis.h"

#include <tclap/CmdLine.h>


namespace fs = std::filesystem;

template<class T>
struct DecompressionResult {
    double DPTime;
    double WriteTime;
    std::unique_ptr<T> decData;
};

template<class T>
DecompressionResult<T> decompress(char *cmpData, size_t cmpSize, SZ::Config& conf) {

    SZ::Timer timer(true);
    std::unique_ptr<T> decData(SZ_decompress<T>(conf, cmpData, cmpSize));
    double compress_time = timer.stop();
    DecompressionResult<T> dpresult;
    dpresult.DPTime = compress_time;
    dpresult.decData = std::move(decData);
    return dpresult;
}


int main(int argc, char** argv) {
    // Initialize the MPI environment
    if(argc < 2){
        printf("Please provide information for the parallel compression\n");
        exit(0);
    }

    TCLAP::CmdLine cmd1("SZ3 Parallel Compression", ' ', "0.1");
    TCLAP::ValueArg<std::string> dataFolderPath("q", "compressed", "The compressed file", false, "", "string");
    TCLAP::ValueArg<std::string> decompressedFolderPath("p", "decompress", "The decompressed folder", false, "", "string");
    TCLAP::ValueArg<std::string> filenamesPath("f", "filenames", "The filenames metadata", false, "", "string");

    cmd1.add(decompressedFolderPath);
    cmd1.add(dataFolderPath);
    cmd1.add(filenamesPath);
    cmd1.parse(argc, argv);
    SZ::Config conf;

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
    std::ifstream fin(filenamesPath, std::ios::in);
    fin >> num_of_files;
    std::string cur_filename;
    while(fin >> cur_filename) {
        filenames.push_back(cur_filename);
        if(world_rank == 0) {
            std::cout << cur_filename << std::endl;
        }
    }
    fin.close();

    MPI_Status status;
    size_t num;

    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, dirpath.c_str(),
                  MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    std::vector<size_t> file_size_after_compression(num_of_files, 0);
    for(size_t i =0;i<num_of_files;i++) {
        size_t cur_size;
        MPI_File_read_at(fh, (2+i) * sizeof(MPI_UNSIGNED), &cur_size, 1, MPI_UNSIGNED, &status);
        file_size_after_compression[i] = cur_size;

    }
    for(size_t i=world_rank;i<num_of_files;i+=world_size) {

        int start_location= sizeof(MPI_UNSIGNED) * (num_of_files + 2);
        for(size_t j=0;j<i;j++){
            start_location += file_size_after_compression[j];
        }
        std::unique_ptr<char[]> compressed_data = std::make_unique<char[]>(file_size_after_compression[i]);
        MPI_File_read_at(fh, start_location, reinterpret_cast<char *>(&compressed_data[0]), file_size_after_compression[i], MPI_BYTE, &status);
        std::string compressed_file = decompressedFolderPath.getValue() + filenames[i] + ".dp";
        DecompressionResult<float> dp_result;
        dp_result = decompress<float>(compressed_data.get(), file_size_after_compression[i], conf);
        printf("My rank is %d, dealing with %s, saving to %s, decompression time: %lf, write time: %lf\n",
               world_rank, filenames[i].c_str(), compressed_file.c_str(), dp_result.DPTime, dp_result.WriteTime);
    }


    MPI_Barrier(MPI_COMM_WORLD);
    if(world_rank==0){
        printf("Finished all tasks! Total Time: %lf\n", MPI_Wtime() - wtime);
    }

    MPI_Finalize();
}