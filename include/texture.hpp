#ifndef TEXTURE_HPP_
#define TEXTURE_HPP_

#include <iostream>
#include <filesystem>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#else
#include <unistd.h>
#endif

#include <cuda_runtime.h>
#include <DirectXTex.h>
#include <vector>


struct Texture{
    ~Texture()
    {
        if(pixel){
                delete[] pixel;
        }
    }

    uint32_t    *pixel {nullptr};
    int2        resolution = make_int2(-1, -1);
    bool        isDiffuseTexture       {false};
};

struct LoadedTexture {
    uint32_t width = 0;
    uint32_t height = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    std::vector<uint32_t> pixels;

    template <typename T>
    const T* dataAs() const {
        return reinterpret_cast<const T*>(pixels.data());
    }

    template <typename T>
    T* dataAs() const {
        return reinterpret_cast<T*>(pixels.data());
    }
};

bool textureLoader(
    const std::string &imgFilePath,
    LoadedTexture& outTex,
    DXGI_FORMAT targetFormat 
);

void convertTextureSpecularToRoughnessMetallic(
    const LoadedTexture& srcSpecular,
    const bool isMetal,
    const float glossiness,
    LoadedTexture& dstARMTexture
);

std::string getFileExtension(const std::string& filePath);

std::wstring toWString(const std::string & str);


#endif // TEXTURE_HPP_