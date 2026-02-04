#ifndef CUDA_BUFFER_H_
#define CUDA_BUFFER_H_

#include "optix8.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <assert.h>

class CUDABuffer {
public:
    CUDABuffer() = default;

    // コピーの禁止
    CUDABuffer(const CUDABuffer&) = delete;
    CUDABuffer& operator=(const CUDABuffer&) = delete;
    
    // move
    CUDABuffer(CUDABuffer&& other) noexcept
        : sizeInBytes(other.sizeInBytes), d_ptr(other.d_ptr)
    {
        other.sizeInBytes = 0;
        other.d_ptr = nullptr;
    }

    CUDABuffer& operator=(CUDABuffer&& other) noexcept
    {
        if(this != &other){
            free();
            sizeInBytes = other.sizeInBytes;
            d_ptr       = other.d_ptr;
            other.sizeInBytes = 0;
            other.d_ptr = nullptr;
        }
        return *this;
    }
    
    ~CUDABuffer()
    {
        free();
    }

    inline CUdeviceptr getDevicePointer() const 
    {
        return (CUdeviceptr)d_ptr;
    }

    void resize(const size_t size)
    {
        if (d_ptr){
            free();
        }
        alloc(size);
    }

    void alloc(const size_t size)
    {
        assert(d_ptr == nullptr);
        this->sizeInBytes = size;
        CUDA_CHECK(cudaMalloc((void**)&d_ptr, sizeInBytes));
    }

    void free()
    {
        cudaFree(d_ptr);
        d_ptr = nullptr;
        sizeInBytes = 0;
    }

    template<typename T>
    void upload(const T *t, size_t count)
    {
        assert(d_ptr != nullptr);
        assert(sizeInBytes == count*sizeof(T));
        CUDA_CHECK(cudaMemcpy(d_ptr, (void *)t, count*sizeof(T), cudaMemcpyHostToDevice));
    }

    template<typename T>
    void allocAndUpload(const std::vector<T> &vt)
    {
        alloc(vt.size()*sizeof(T));
        upload((const T*)vt.data(), vt.size());
    }

    template<typename T>
    void download(T * t, size_t count)
    {
        assert(d_ptr != nullptr);
        assert(sizeInBytes == count*sizeof(T));
        CUDA_CHECK(cudaMemcpy((void *)t, d_ptr, count*sizeof(T), cudaMemcpyDeviceToHost));
    }

    inline size_t getSizeInBytes() const 
    {
        return sizeInBytes;
    }

protected:


    size_t sizeInBytes  {0};
    void *d_ptr {nullptr};
};


#endif //CUDA_BUFFER_H_