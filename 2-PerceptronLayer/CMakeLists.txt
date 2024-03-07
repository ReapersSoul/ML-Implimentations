project(PerceptronLayer)

set(CMAKE_CXX_STANDARD 20)

add_executable(PerceptronLayer "main.cpp" "PerceptronLayer.cpp" "PerceptronLayer.hpp")

find_package(SDL2 CONFIG REQUIRED)
target_include_directories(PerceptronLayer PRIVATE ${SDL2_INCLUDE_DIRS})
target_link_libraries(PerceptronLayer
	PRIVATE
	$<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
	$<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
)

find_package(SDL2_ttf CONFIG REQUIRED)
target_include_directories(PerceptronLayer PRIVATE ${SDL2_TTF_INCLUDE_DIRS})
target_link_libraries(PerceptronLayer PRIVATE $<IF:$<TARGET_EXISTS:SDL2_ttf::SDL2_ttf>,SDL2_ttf::SDL2_ttf,SDL2_ttf::SDL2_ttf-static>)

find_package(OpenCL REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL::OpenCL)