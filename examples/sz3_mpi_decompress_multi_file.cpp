#include <mpi.h>
#include <iostream>
#include <cstdio>
#include <string>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <filesystem>

#include <memory>

#include "SZ3/api/sz.hpp"
#include "qcat_dataAnalysis.h"

#include <tclap/CmdLine.h>

#include <cstdint>
#include <climits>

#if SIZE_MAX == UCHAR_MAX
#define my_MPI_SIZE_T MPI_UNSIGNED_CHAR
#elif SIZE_MAX == USHRT_MAX
#define my_MPI_SIZE_T MPI_UNSIGNED_SHORT
#elif SIZE_MAX == UINT_MAX
#define my_MPI_SIZE_T MPI_UNSIGNED
#elif SIZE_MAX == ULONG_MAX
#define my_MPI_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
#define my_MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#else
   #error "what is happening here?"
#endif


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
    if(argc < 2){
        printf("Please provide information for the parallel compression\n");
        exit(0);
    }

    TCLAP::CmdLine cmd1("SZ3 Parallel Compression", ' ', "0.1");
    TCLAP::ValueArg<std::string> dataFolderPath("q", "compressed", "The general compressed file", false, "", "string");
    TCLAP::ValueArg<std::string> decompressedFolderPath("p", "decompress", "The decompressed folder", false, "", "string");
    TCLAP::ValueArg<std::string> filenamesPath("f", "filenames", "The filenames metadata", false, "", "string");

    cmd1.add(decompressedFolderPath);
    cmd1.add(dataFolderPath);
    cmd1.add(filenamesPath);
    cmd1.parse(argc, argv);
    SZ::Config conf;
    std::string general_compressed_path = dataFolderPath.getValue();
    int dot_pos = general_compressed_path.rfind('.');
    if(dot_pos == std::string::npos) {
        std::cerr << "The general compressed file's format is not correct!" << std::endl;
        exit(1);
    }
    std::string compressedFilePrefix = general_compressed_path.substr(0, dot_pos);
    std::string compressedFileExt = general_compressed_path.substr(dot_pos);
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    size_t num_of_files;
    double wtime;

    std::vector<std::string> filenames;
    if(world_rank==0) {
        wtime = MPI_Wtime();
    }
    std::ifstream fin(filenamesPath, std::ios::in);
    fin >> num_of_files;
    size_t num_in_a_group;
    fin >> num_in_a_group;
    std::string cur_filename;
    while(fin >> cur_filename) {
        filenames.push_back(cur_filename);
        if(world_rank == 0) {
            std::cout << cur_filename << std::endl;
        }
    }
    fin.close();
    size_t num_of_complete_groups = num_of_files / num_in_a_group;
    MPI_Status status;
    size_t num;

    MPI_File fh;

    std::vector<size_t> file_size_after_compression(num_of_files, 0);
    for(size_t j=0; j < num_of_complete_groups; j++) {
        std::string cur_file_path = compressedFilePrefix + std::to_string(j) + compressedFileExt;
        MPI_File_open(MPI_COMM_WORLD, cur_file_path.c_str(),
                MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
        for (size_t i = 0; i < num_in_a_group; i++) {
            size_t cur_size;
            MPI_File_read_at(fh, sizeof(MPI_OFFSET) + (1 + i) * sizeof(my_MPI_SIZE_T), &cur_size, 1, my_MPI_SIZE_T, &status);
            file_size_after_compression[j * num_in_a_group + i] = cur_size;
        }
        MPI_File_close(&fh);
    }
    size_t remaining_files = num_of_files - num_of_complete_groups * num_in_a_group;
    if(remaining_files > 0) {
        std::string cur_file_path = compressedFilePrefix + std::to_string(num_of_complete_groups) + compressedFileExt;
        MPI_File_open(MPI_COMM_WORLD, cur_file_path.c_str(),
                      MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
        for(size_t i =0;i<remaining_files;i++) {
            size_t cur_size;
            MPI_File_read_at(fh, sizeof(MPI_OFFSET) + (1 + i) * sizeof(my_MPI_SIZE_T), &cur_size, 1, my_MPI_SIZE_T, &status);
            file_size_after_compression[num_of_complete_groups * num_in_a_group + i] = cur_size;
        }
        MPI_File_close(&fh);
    }
//    MPI_Barrier(MPI_COMM_WORLD);
    bool final_round = false;
    size_t round, my_round = 0;
    for(size_t i=world_rank;i<num_of_files;i+=world_size) {
        if(num_of_files - my_round * world_size < world_size) {
            final_round = true;
        }
        MPI_Offset start_location= sizeof(my_MPI_SIZE_T) * (num_in_a_group + 1) + sizeof(MPI_OFFSET);
        round = i / num_in_a_group;
        std::string cur_file_path = compressedFilePrefix + std::to_string(round) + compressedFileExt;
        MPI_File_open(MPI_COMM_WORLD, cur_file_path.c_str(),
                      MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
        for(size_t j=0;j < i % num_in_a_group;j++){
            start_location += file_size_after_compression[round * num_in_a_group + j];
        }
        // std::unique_ptr<char[]> compressed_data = std::make_unique<char[]>(file_size_after_compression[i]);
        char *compressed_data = new char[file_size_after_compression[i]];
        // MPI_File_read_at(fh, start_location, reinterpret_cast<char *>(&compressed_data[0]), file_size_after_compression[i], MPI_BYTE, &status);
        MPI_File_read_at(fh, start_location, compressed_data, file_size_after_compression[i], MPI_SIGNED_CHAR, &status);
        MPI_File_close(&fh);
        std::string decompressed_file = decompressedFolderPath.getValue() + filenames[i] + ".dp";
        DecompressionResult dp_result;
        dp_result = decompress<float>(compressed_data, file_size_after_compression[i], decompressed_file.c_str(), conf, 1);
        printf("My rank is %d, dealing with %s, saving to %s, decompression time: %lf, write time: %lf\n",
               world_rank, filenames[i].c_str(), decompressed_file.c_str(), dp_result.DPTime, dp_result.WriteTime);
        delete[] compressed_data;
        if(!final_round) {my_round++;}
    }

//    if(!final_round) {
//        round = num_of_complete_groups + 1;
//        std::string cur_file_path = compressedFilePrefix + std::to_string(round) + compressedFileExt;
//        MPI_File_open(MPI_COMM_WORLD, cur_file_path.c_str(),
//                      MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
//        MPI_File_close(&fh);
//    }
    printf("rank %d has finished all tasks\n", world_rank);

    MPI_Barrier(MPI_COMM_WORLD);
    if(world_rank==0){
        printf("Finished all tasks! Total Time: %lf\n", MPI_Wtime() - wtime);
    }

    MPI_Finalize();
}