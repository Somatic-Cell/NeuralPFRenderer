#ifndef CUDA_TEXTURE_H_
#define CUDA_TEXTURE_H_

#include "optix8.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <cstring>
#include <utility>
#include <assert.h>

template<typename T>
struct TextureTraits;

template<>
struct TextureTraits<float> {
    static cudaChannelFormatDesc channelDesc() {
        return cudaCreateChannelDesc<float>();
    }
    static constexpr cudaTextureReadMode readMode = cudaReadModeElementType;
    static constexpr bool supportLinearFiltering = true;
};



template<typename T>
class CUDATexture2D {
public:
    CUDATexture2D() = default;
    
    // コピーの禁止
    CUDATexture2D(const CUDATexture2D&) = delete;
    CUDATexture2D& operator=(const CUDATexture2D&) = delete;

    // move
    CUDATexture2D(CUDATexture2D&& other) noexcept
        : m_array(other.m_array), m_tex(other.m_tex),
          m_width(other.m_width), m_height(other.m_height)
    {
        other.m_array   = nullptr;
        other.m_tex     = 0;
        other.m_width   = 0;
        other.m_height  = 0;
    }

    CUDATexture2D& operator=(CUDATexture2D&& other) noexcept
    {
        if(this != &other){
            free();
            m_array     = other.m_array;
            m_tex       = other.m_tex;
            m_width     = other.m_width;
            m_height    = other.m_height;

            other.m_array   = nullptr;
            other.m_tex     = 0;
            other.m_width   = 0;
            other.m_height  = 0;
        }
        return *this;
    }

    ~CUDATexture2D() { free(); }

    void free()
    {
        if(m_tex){
            cudaDestroyTextureObject(m_tex);
            m_tex = 0;
        }
        if(m_array){
            cudaFreeArray(m_array);
            m_array = nullptr;
        }
        m_width = 0;
        m_height = 0;
    }

    void createFromHost(const T* src, uint32_t width, uint32_t height)
    {
        free();

        m_width     = width;
        m_height    = height;

        cudaChannelFormatDesc desc = TextureTraits<T>::channelDesc();
        CUDA_CHECK(cudaMallocArray(&m_array, &desc, width, height));

        CUDA_CHECK(cudaMemcpy2DToArray(
            m_array,
            0, 0,
            src,
            sizeof(T) * width,
            sizeof(T) * width,
            height,
            cudaMemcpyHostToDevice
        ));

        cudaResourceDesc resDesc{};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = m_array;

        cudaTextureDesc texDesc{};
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode =
            TextureTraits<T>::supportLinearFiltering ? cudaFilterModeLinear : cudaFilterModePoint;
        texDesc.readMode = TextureTraits<T>::readMode;
        texDesc.normalizedCoords = 1;

        CUDA_CHECK(cudaCreateTextureObject(&m_tex, &resDesc, &texDesc, nullptr));
    }

    cudaTextureObject_t texture() const { return m_tex; }
    uint32_t width() const { return m_width; }
    uint32_t height() const { return m_height; }


private:
    cudaArray_t m_array         = nullptr;
    cudaTextureObject_t m_tex   = 0;
    uint32_t m_width            = 0;
    uint32_t m_height           = 0;
};


template<typename T>
class CUDATexture3D {
public:
    CUDATexture3D() = default;
    
    // コピーの禁止
    CUDATexture3D(const CUDATexture3D&) = delete;
    CUDATexture3D& operator=(const CUDATexture3D&) = delete;

    // move
    CUDATexture3D(CUDATexture3D&& other) noexcept
        : m_array(other.m_array), m_tex(other.m_tex),
          m_width(other.m_width), m_height(other.m_height), m_depth(other.m_depth)
    {
        other.m_array   = nullptr;
        other.m_tex     = 0;
        other.m_width   = 0;
        other.m_height  = 0;
        other.m_depth   = 0;
    }

    CUDATexture3D& operator=(CUDATexture3D&& other) noexcept
    {
        if(this != &other){
            free();
            m_array     = other.m_array;
            m_tex       = other.m_tex;
            m_width     = other.m_width;
            m_height    = other.m_height;
            m_depth     = other.m_depth;

            other.m_array   = nullptr;
            other.m_tex     = 0;
            other.m_width   = 0;
            other.m_height  = 0;
            other.m_depth   = 0;
        }
        return *this;
    }

    ~CUDATexture3D() { free(); }

    void free()
    {
        if(m_tex){
            cudaDestroyTextureObject(m_tex);
            m_tex = 0;
        }
        if(m_array){
            cudaFreeArray(m_array);
            m_array = nullptr;
        }
        m_width     = 0;
        m_height    = 0;
        m_depth     = 0;
    }

    void createFromHost(const T* src, uint32_t width, uint32_t height, uint32_t depth)
    {
        free();

        m_width     = width;
        m_height    = height;
        m_depth     = depth;

        cudaExtent extent = make_cudaExtent(width, height, depth);
        cudaChannelFormatDesc desc = TextureTraits<T>::channelDesc();
        CUDA_CHECK(cudaMalloc3DArray(&m_array, &desc, extent));


        cudaMemcpy3DParms copyParams{};
        copyParams.srcPtr = make_cudaPitchedPtr(
            const_cast<T*>(src),
            sizeof(T) * width,
            width,
            height
        );
        copyParams.dstArray = m_array;
        copyParams.extent   = extent;
        copyParams.kind     = cudaMemcpyHostToDevice;
        CUDA_CHECK(cudaMemcpy3D(&copyParams));

        cudaResourceDesc resDesc{};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = m_array;

        cudaTextureDesc texDesc{};
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.addressMode[2] = cudaAddressModeClamp;
        texDesc.filterMode =
            TextureTraits<T>::supportLinearFiltering ? cudaFilterModeLinear : cudaFilterModePoint;
        texDesc.readMode = TextureTraits<T>::readMode;
        texDesc.normalizedCoords = 1;

        CUDA_CHECK(cudaCreateTextureObject(&m_tex, &resDesc, &texDesc, nullptr));
    }

    cudaTextureObject_t texture() const { return m_tex; }
    uint32_t width() const { return m_width; }
    uint32_t height() const { return m_height; }
    uint32_t depth() const { return m_depth; }


private:
    cudaArray_t m_array         = nullptr;
    cudaTextureObject_t m_tex   = 0;
    uint32_t m_width            = 0;
    uint32_t m_height           = 0;
    uint32_t m_depth            = 0;
};

static cudaTextureObject_t createLinearFloatTexture1D(CUdeviceptr dptr, size_t sizeInBytes)
{
    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = reinterpret_cast<void*>(static_cast<uintptr_t>(dptr));
    resDesc.res.linear.desc = cudaCreateChannelDesc<float>();
    resDesc.res.linear.sizeInBytes = sizeInBytes;

    cudaTextureDesc texDesc{};
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t tex = 0;
    CUDA_CHECK(cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr));
    return tex;
}

#endif // CUDA_TEXTURE_H_