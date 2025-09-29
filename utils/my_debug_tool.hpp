#ifndef MY_DEBUG_TOOLS_HPP_
#define MY_DEBUG_TOOLS_HPP_

#include <stdio.h>
#include <iostream>
#include <stdexcept>
#include <memory>
#include <assert.h>
#include <string>
#include <math.h>
#ifdef __CUDA_ARCH__
#  include <math_constants.h>
#else
#  include <cmath>
#endif
#include <algorithm>
#ifdef __GNUC__
#  include <sys/time.h>
#  include <stdint.h>
#endif
#include <stdexcept>

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <Windows.h>
    #ifdef min
        #undef min
    #endif
    #ifdef max
        #undef max
    #endif
#endif

#if defined(_MSC_VER)
#  define MTK_DLL_EXPORT __declspec(dllexport)
#  define MTK_DLL_IMPORT __declspec(dllimport)
#elif defined(__clang__) || defined(__GNUC__)
#  define MTK_DLL_EXPORT __attribute__((visibility("default")))
#  define MTK_DLL_IMPORT __attribute__((visibility("default")))
#else
#  define MTK_DLL_EXPORT
#  define MTK_DLL_IMPORT
#endif

#ifndef PRINT
#define PRINT(var) std::cout << #var << "=" << var << std::endl;
#define PING std::cout << __FILE__ << "::" << __LINE__ << ": " << __FUNCTION__ << std::endl;
#endif // PRINT

#define MTK_TERMINAL_RED "\033[1;31m"
#define MTK_TERMINAL_GREEN "\033[1;32m"
#define MTK_TERMINAL_YELLOW "\033[1;33m"
#define MTK_TERMINAL_BLUE "\033[1;34m"
#define MTK_TERMINAL_RESET "\033[0m"
#define MTK_TERMINAL_DEFAULT MTK_TERMINAL_RESET
#define MTK_TERMINAL_BOLD "\033[1;1m"


#endif // !MY_DEBUG_TOOLS_HPP_