cmake_minimum_required(VERSION 3.24)
project(LBM_Dyfuzja CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Ścieżki względne (idź poziom wyżej, potem do test na clionie)
set(FREEGLUT_DIR "${CMAKE_SOURCE_DIR}/libs/freeglut")
set(GLEW_DIR "${CMAKE_SOURCE_DIR}/libs/glew-2.2.0")

# Include directories
include_directories(${FREEGLUT_DIR}/include)
include_directories(${GLEW_DIR}/include)

# Link directories
link_directories(${FREEGLUT_DIR}/lib/x64)
link_directories(${GLEW_DIR}/lib/Release/x64)

add_executable(LBM_Dyfuzja main.cu)

set_target_properties(LBM_Dyfuzja PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

# Linkowanie bibliotek
target_link_libraries(LBM_Dyfuzja
        freeglut
        glew32
        opengl32)

# Automatyczne kopiowanie DLL
add_custom_command(TARGET LBM_Dyfuzja POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${FREEGLUT_DIR}/bin/x64/freeglut.dll"
        $<TARGET_FILE_DIR:LBM_Dyfuzja>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${GLEW_DIR}/bin/Release/x64/glew32.dll"
        $<TARGET_FILE_DIR:LBM_Dyfuzja>
)