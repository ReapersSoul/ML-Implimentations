project(DeepNeuralNetwork)

set(CMAKE_CXX_STANDARD 20)

add_executable(DeepNeuralNetwork main.cpp DeepNeuralNetwork.cpp DeepNeuralNetwork.hpp)

find_package(SDL2 CONFIG REQUIRED)
target_include_directories(DeepNeuralNetwork PRIVATE ${SDL2_INCLUDE_DIRS})
target_link_libraries(DeepNeuralNetwork
	PRIVATE
	$<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
	$<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
)

find_package(SDL2_ttf CONFIG REQUIRED)
target_include_directories(DeepNeuralNetwork PRIVATE ${SDL2_TTF_INCLUDE_DIRS})
target_link_libraries(DeepNeuralNetwork PRIVATE $<IF:$<TARGET_EXISTS:SDL2_ttf::SDL2_ttf>,SDL2_ttf::SDL2_ttf,SDL2_ttf::SDL2_ttf-static>)