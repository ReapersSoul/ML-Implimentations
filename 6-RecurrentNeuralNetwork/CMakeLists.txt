project(RecurrentNeuralNetwork)

set(CMAKE_CXX_STANDARD 20)

add_executable(RecurrentNeuralNetwork main.cpp RecurrentNeuralNetwork.cpp RecurrentNeuralNetwork.hpp)

find_package(SDL2 CONFIG REQUIRED)
target_include_directories(RecurrentNeuralNetwork PRIVATE ${SDL2_INCLUDE_DIRS})
target_link_libraries(RecurrentNeuralNetwork
	PRIVATE
	$<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
	$<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
)

find_package(SDL2_ttf CONFIG REQUIRED)
target_include_directories(RecurrentNeuralNetwork PRIVATE ${SDL2_TTF_INCLUDE_DIRS})
target_link_libraries(RecurrentNeuralNetwork PRIVATE $<IF:$<TARGET_EXISTS:SDL2_ttf::SDL2_ttf>,SDL2_ttf::SDL2_ttf,SDL2_ttf::SDL2_ttf-static>)