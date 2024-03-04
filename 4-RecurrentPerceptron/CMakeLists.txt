project(RecurrentPerceptron)

set(CMAKE_CXX_STANDARD 20)

add_executable(RecurrentPerceptron main.cpp RecurrentPerceptron.cpp RecurrentPerceptron.hpp)

find_package(SDL2 CONFIG REQUIRED)
target_include_directories(RecurrentPerceptron PRIVATE ${SDL2_INCLUDE_DIRS})
target_link_libraries(RecurrentPerceptron
	PRIVATE
	$<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
	$<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
)

find_package(SDL2_ttf CONFIG REQUIRED)
target_include_directories(RecurrentPerceptron PRIVATE ${SDL2_TTF_INCLUDE_DIRS})
target_link_libraries(RecurrentPerceptron PRIVATE $<IF:$<TARGET_EXISTS:SDL2_ttf::SDL2_ttf>,SDL2_ttf::SDL2_ttf,SDL2_ttf::SDL2_ttf-static>)