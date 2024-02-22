# CMakeList.txt : CMake project for WebGPU-Dawn, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.13)
project(app)
set(CMAKE_CXX_STANDARD 20)
set(EMSCRIPTEN_ROOT "C:/SourceCode/Tools/emsdk/upstream/emscripten/cache/sysroot")
message("EMSCRIPTEN_ROOT=${EMSCRIPTEN_ROOT}")

# Add source to this project's executable.
add_executable (app "main.cpp" "src/videoframe/WebGPURFRender.cpp" "src/videoframe/WebGPURFRender.h")
target_include_directories(app PRIVATE "src/videoframe/")
target_include_directories(app PRIVATE ${EMSCRIPTEN_ROOT})
include_directories("src/videoframe/")

if(EMSCRIPTEN)
	set_target_properties(app PROPERTIES SUFFIX ".html")
	target_link_options(app PRIVATE "-sUSE_WEBGPU=1" "-sUSE_GLFW=3" "-sEXPORTED_FUNCTIONS=['_main', '_ReceiveBuffer', '_ReceiveYUVBuffer', '_StartRendering', '_malloc', '_free']" "-sEXPORTED_RUNTIME_METHODS=['ccall', 'cwrap']")
else()
	set(DAWN_FETCH_DEPENDENCIES ON)
	add_subdirectory("dawn" EXCLUDE_FROM_ALL)
	target_link_libraries(app PRIVATE webgpu_dawn webgpu_cpp webgpu_glfw)
endif()



# TODO: Add tests and install targets if needed.
