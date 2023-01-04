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
    TCLAP::ValueArg<std::string> dimensionArg("d", "dimension", "the dimention of data", true, "", "dimension");
    TCLAP::ValueArg<std::string> dataFolderPath("q", "data", "The data folder", true, "", "string");
    TCLAP::ValueArg<std::string> compressedFilePath("p", "compress", "The compressed file", true, "", "string");
    TCLAP::ValueArg<int> optimalSizePerfileArg("s", "size", "The optimal size per file in GB", false, 1, "int");

    cmd1.add(dimensionArg);
    cmd1.add(ebArg);
    cmd1.add(confFilePath);
    cmd1.add(compressedFilePath);
    cmd1.add(dataFolderPath);
    cmd1.add(optimalSizePerfileArg);
    cmd1.parse(argc, argv);
    std::vector<size_t> dims;
    long optimal_size = optimalSizePerfileArg.getValue() * 1000000000;
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

    std::string compressedFile = compressedFilePath.getValue();
    int dot_location = compressedFile.rfind('.');
    if(dot_location == std::string::npos) {
        std::cerr << "The compressed file format is not correct!" << std::endl;
        exit(1);
    }
    std::string compressedFilePrefix = compressedFile.substr(0, dot_location);
    std::string compressedFileExt = compressedFile.substr(dot_location);

    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    size_t num_of_files;
    double wtime;
    std::string dirpath = dataFolderPath.getValue();
    std::vector<std::string> file_paths, filenames;
    if(world_rank==0) {
        wtime = MPI_Wtime();
    }
    for (const auto & entry : fs::directory_iterator(dirpath)) {
        std::string cur_path = entry.path();
        std::string filename = cur_path.substr(cur_path.rfind('/') + 1);
        filenames.push_back(filename);
        file_paths.push_back(cur_path);
    }
    std::sort(filenames.begin(), filenames.end());
    std::sort(file_paths.begin(), file_paths.end());
    num_of_files = file_paths.size();

    MPI_Status status;
    size_t num;

    MPI_File fh;

//    size_t cur_file_index = 0;
    size_t round = 0;

    std::vector<size_t> file_size_after_compression(num_of_files, 0);
    std::vector<bool> file_size_populated(num_of_files, false);

    MPI_Status mpi_status;
    bool final_round = false;
    for(size_t i=world_rank;i<num_of_files;i+=world_size) {
        if(num_of_files - round * world_size < world_size) {
            final_round = true;
        }
        std::string cur_compressed_file = compressedFilePrefix + std::to_string(round) + compressedFileExt;
        MPI_File_open(MPI_COMM_WORLD, cur_compressed_file.c_str(),
                      MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
        
        if(world_rank == 0) {
            if(i + world_size < num_of_files) {
                MPI_File_write_at(fh, 0, &world_size, 1, MPI_UNSIGNED, &mpi_status);
            } else {
                size_t this_round_file_num = num_of_files - i;
                MPI_File_write_at(fh, 0, &this_round_file_num, 1, MPI_UNSIGNED, &mpi_status);
            }
        }
        std::string file_path_str = file_paths[i];
        std::string filename = file_path_str.substr(file_path_str.rfind('/') + 1);
        auto data = SZ::readfile<float>(file_path_str.c_str(), num);
        conf.num = num;
        auto cp_result = compress<float>(data.get(), conf);
        size_t compressed_size = cp_result.outsize;
        file_size_after_compression[i] = compressed_size;
        file_size_populated[i] = true;
        MPI_Offset size_location = sizeof(MPI_OFFSET) + sizeof(MPI_UNSIGNED) * (1 + i % world_size);
        printf("rank %u, file %u, round: %u, size location: %llu\n", world_rank, i, round, size_location);
        MPI_File_write_at(fh, size_location, &compressed_size, 1, MPI_UNSIGNED, &mpi_status);
        size_t send_data[2] = {i, compressed_size};
        MPI_Bcast(send_data, 2, MPI_UNSIGNED, world_rank, MPI_COMM_WORLD);
        size_t receive_data[2];
        size_t loop_size = world_size;
        if(final_round) {
            // loop_size = num_of_files - round * world_size;
            loop_size = world_rank;
        }
        for(size_t j = 0; j < loop_size; j++) {
            if(j == world_rank) continue;
            // if(j + round * world_size >= num_of_files) continue;
            MPI_Bcast(receive_data, 2, MPI_UNSIGNED, j, MPI_COMM_WORLD);
            file_size_after_compression[receive_data[0]] = receive_data[1];
            file_size_populated[receive_data[0]] = true;
        }
        // printf("Rank %d, file %u, reach the barrier and is waiting\n", world_rank, i);
        // MPI_Barrier(MPI_COMM_WORLD);
        MPI_Offset start_location= sizeof(MPI_UNSIGNED) * (world_size + 1) + sizeof(MPI_OFFSET);
        for(size_t j=0;j < world_rank;j++){
            if(file_size_populated[round * world_size + j] == false) {
                printf("Significant Error! File %d's size is unknown in rank %d! Should Exit!\n", j, world_rank);
                exit(1);
            }
            start_location += file_size_after_compression[round * world_size + j];
        }

        MPI_File_write_at(fh, start_location, cp_result.cpdata.get(), compressed_size, MPI_BYTE, &mpi_status);
        if(i == num_of_files - 1 || i == (round+1) * world_size - 1) {
            MPI_Offset final_location = start_location + compressed_size;
            MPI_File_write_at(fh, sizeof(MPI_UNSIGNED), &final_location, 1, MPI_OFFSET, &mpi_status);
            printf("round %u is finishing the final file, total size is %llu\n", round, final_location);
        }
        MPI_File_close(&fh);
        printf("My rank is %d, round %u, dealing with No. %u file, writing offset: %llu, compressed_size: %zu, compression time: %lf, compression ratio: %lf\n",
               world_rank, round, i, start_location, compressed_size, cp_result.CPTime, cp_result.CR);
        cp_result.cpdata.reset(nullptr);
        if(!final_round) round++;
    }
    printf("My rank is %d, current round: %d, final round: %d I finished all tasks assigned to me!\n", world_rank, round, final_round);
    size_t receive_data[2];
    if(!final_round) {
        std::string cur_compressed_file = compressedFilePrefix + std::to_string(round) + compressedFileExt;
        MPI_File_open(MPI_COMM_WORLD, cur_compressed_file.c_str(),
                        MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
        MPI_File_close(&fh);
        for (size_t j = 0; j < num_of_files - round * world_size; j++) {
            MPI_Bcast(receive_data, 2, MPI_UNSIGNED, j, MPI_COMM_WORLD);
        }
    } else {
        for (size_t j = world_rank+1; j < num_of_files - round * world_size; j++) {
            MPI_Bcast(receive_data, 2, MPI_UNSIGNED, j, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if(world_rank==0) {
        std::string compressed_file = compressedFilePath.getValue();
        std::string compressed_folder = compressed_file.substr(0, compressed_file.rfind('/') + 1);
        std::ofstream fout(compressed_folder + "filenames.txt", std::ios::out);
        fout << filenames.size() << std::endl;
        fout << world_size << std::endl;
        for(int i = 0;i<filenames.size();i++) {
            fout << filenames[i] << std::endl;
        }
        printf("Finished all tasks! Total Time: %lf\n", MPI_Wtime() - wtime);
    }
    // MPI_Barrier(MPI_COMM_WORLD);
    // Finalize the MPI environment.
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}