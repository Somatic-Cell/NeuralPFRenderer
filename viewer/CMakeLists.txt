set(project_SRC
    glfw_window.cpp
)

set(project_HEADERS
    glfw_window.hpp
)

add_library(viewer STATIC
    ${project_SRC}
    ${project_HEADERS}
)

target_include_directories(viewer
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)