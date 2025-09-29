#ifndef PTX_DATA_READER_HPP_
#define PTX_DATA_READER_HPP_

#include <filesystem>
#include <iostream>
#include <fstream>

#if defined(_WIN32)
#include <windows.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#else
#include <unistd.h>
#endif


// ptx (optixir) のバイナリデータを読む関数
static std::vector<char> readData(std::string const &fileName)
{
    std::filesystem::path exe_path;
    char path_buffer[MAX_PATH] = {};
#if defined(_WIN32)
    if(GetModuleFileNameA(NULL, path_buffer, MAX_PATH) == 0){
        std::cerr << "ERROR: GetModuleFileNameA failed" << std::endl;
        return std::vector<char>();
    }
    exe_path = std::filesystem::path(path_buffer);
#else
    ssize_t count = readlink("/proc/self/exe", exe_path, PATH_MAX);
    if(count == -1) {
        std::cerr << "ERROR: readlink() failed" << std::endl;
        return std::vector<char>();
    }
#endif
    std::filesystem::path ptx_dir = exe_path.parent_path().parent_path() / "ptxes" / fileName;
    std::ifstream inputData(ptx_dir, std::ios::binary);

    std::cout << "PTX Path:" << ptx_dir << std::endl; 

    if(inputData.fail())
    {
        std::cerr << "ERROR: readData() failed to open file " << fileName << std::endl;
        return std::vector<char>();
    }

    std::vector<char> data(std::istreambuf_iterator<char>(inputData), {});

    if(inputData.fail())
    {
        std::cerr << "ERROR: readData() failed to read file " << fileName << std::endl;
        return std::vector<char>();
    }

    return data;
}


#endif // PTX_DATA_READER_HPP_