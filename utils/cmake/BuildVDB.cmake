# =========================================
# File Name : BuildVDB.cmake
# Encoding  : UTF-8
# =========================================

# オプション
option(WITH_OPENVDB "Enable CPU OpenVDB support" ON)
set(DEPS_MODE "auto" CACHE STRING "Depends mode: auto, system, or submodule")
set_property(CACHE DEPS_MODE PROPERTY STRINGS auto system submodule)

# NanoVDB
set(_NANOVDB_DIR ${CMAKE_SOURCE_DIR}/ext/openvdb/nanovdb)
if(NOT EXISTS "${_NANOVDB_DIR}")
    message(FATAL_ERROR "NanoVDB headers not found at ${_NANOVDB_DIR}. Add OpenVDB as submodule.")
endif()

add_library(nanovdb_headers INTERFACE)
target_include_directories(nanovdb_headers INTERFACE
    ${_NANOVDB_DIR}
    ${_NANOVDB_DIR}/nanovdb
)
target_compile_definitions(nanovdb_headers INTERFACE NANOVDB_USE_CUDA)

# エクスポートのための変数
set(VDB_TARGETS nanovdb_headers)

# OpenVDB
if(WITH_OPENVDB)
    set(_OPENVDB_DIR ${CMAKE_SOURCE_DIR}/ext/openvdb)

    if(DEPS_MODE STREQUAL "system" OR (DEPS_MODE STREQUAL "auto" AND NOT EXISTS "${_OPENVDB_DIR}/CMakeLists.txt"))
        find_package(OPENVDB CONFIG REQUIRED)
        set(_OpenVDBTarget OpenVDB::openvdb)
    else()
        find_package(TBB QUIET)
        find_package(ZLIB REQUIRED)
        find_package(OpenEXR QUIET)
        find_package(Imath QUIET)
        find_package(Blosc QUIET)
        find_package(Boost QUIET COMPONENTS iostreams)

        # OpenVDB の不要なコンポーネントを off にする
        set(OPENVDB_BUILD_PYTHON    OFF CACHE BOOL "" FORCE)
        set(OPENVDB_BUILD_UNITTESTS OFF CACHE BOOL "" FORCE)
        set(OPENVDB_BUILD_BINARIES  OFF CACHE BOOL "" FORCE)
        set(OPENVDB_BUILD_DOCS      OFF CACHE BOOL "" FORCE)
        set(OPENVDB_BUILD_RPATH     OFF CACHE BOOL "" FORCE)
        set(OPENVDB_BUILD_BLOSC     OFF CACHE BOOL "" FORCE)
        set(OPENVDB_BUILD_SHARED    ON  CACHE BOOL "" FORCE)

        set(OPENVDB_ENABLE_UNINSTALL OFF CACHE BOOL "" FORCE)

        add_subdirectory(${_OPENVDB_DIR} ${CMAKE_BINARY_DIR}/_deps/openvdb)

        if(TARGET OpenVDB::openvdb)
          set(_OpenVDBTarget OpenVDB::openvdb)
        elseif(TARGET openvdb)
          set(_OpenVDBTarget openvdb)
        else()
          message(FATAL_ERROR "OpenVDB target not found after add_subdirectory.")
        endif()
    endif()

    # 呼び出し側で簡単に使えるよう返す
    set(OPENVDB_TARGET ${_OpenVDBTarget})
    set(TARGETS ${VDB_TARGETS} ${_OpenVDBTarget})
    add_definitions(-DOPENVDB_AVAILABLE=1)
else()
    add_definitions(-DOPENVDB_AVAILABLE=0)
endif()