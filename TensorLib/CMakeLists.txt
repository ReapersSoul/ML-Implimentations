project(TensorLib)

set(CMAKE_CXX_STANDARD 20)

add_executable(${PROJECT_NAME} main.cpp TensorLib.cpp TensorLib.hpp)

find_package(OpenCL REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL::OpenCL)