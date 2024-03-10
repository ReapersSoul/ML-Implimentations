project(MLLib)

set(CMAKE_CXX_STANDARD 20)

add_executable(${PROJECT_NAME} Src/main.cpp Src/MLLib.cpp Headers/MLLib.hpp)

find_package(OpenCL REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL::OpenCL)