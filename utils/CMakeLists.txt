# =========================================
# File Name : CMakeLists.txt
# Encoding  : UTF-8
# =========================================

add_library(utils INTERFACE)

target_include_directories(
    utils 
    INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}
)
