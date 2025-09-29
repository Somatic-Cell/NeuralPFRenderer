#ifndef OPTIX_8_HPP_
#define OPTIX_8_HPP_


#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <sstream>
#include <stdexcept>


#define OPTIX_CHECK( call ) \
{ \
    OptixResult res = call; \
    if(res != OPTIX_SUCCESS){ \
        fprintf(stderr, "Optix call (%s) failed with code %d (line %d)\n", #call, res, __LINE__); \
        exit(2); \
    } \
}

// #define CUDA_CHECK_NOEXEPT(call) \
// { \
//     cuda##call; \
// }

// ## はトークン連結演算子といい，マクロで，文字列を連結することができる．
#define CUDA_CHECK(call) \
 { \
    cudaError_t rc = call; \
    if(rc != cudaSuccess) { \
        std::stringstream txt; \
        cudaError_t err = rc; \
        txt << "CUDA Error " << cudaGetErrorName(err) \
            << "( " << cudaGetErrorString(err) << ")"; \
        throw std::runtime_error(txt.str()); \
    } \
 }

 #define CUDA_SYNC_CHECK() \
 { \
    cudaDeviceSynchronize(); \
    cudaError_t error = cudaGetLastError(); \
    if(error != cudaSuccess) { \
        fprintf( stderr, "error (%s: line %d): %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(2); \
    } \
 }

#define CUDA_DRIVER_CHECK(call) \
do {                                                                 \
    CUresult err = call;                                             \
    if (err != CUDA_SUCCESS) {                                       \
        const char *msg = nullptr;                                   \
        cuGetErrorString(err, &msg);                                 \
        std::cerr << "CUDA Driver Error: " << msg                    \
                    << " at " << __FILE__ << ":" << __LINE__ << "\n";\
        exit(1);                                                     \
    }                                                                \
} while (0)

#endif // OPTIX_8_HPP_