cmake_minimum_required(VERSION 3.20)
project(Sdl3ComputeDct)
set(CMAKE_EXPORT_COMPILE_COMMANDS true)

find_package(imgui CONFIG REQUIRED)
find_package(SDL3 REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(Stb REQUIRED)

option(SHADERCROSS_PATH "Path to SDL_shadercross executable")

if (MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Werror)
endif()

if (WIN32)
    set(DXC_PATH dxc)
elseif (APPLE)
    set(SHADERCROSS_PATH "/Users/matheus/Applications/SDL_shadercross/build/shadercross")
else()
    if (NOT SHADERCROSS_PATH)
        message(FATAL_ERROR "Please set SHADERCROSS_PATH when invoking CMake.")
    endif()
endif()

# Vertex shader
if (WIN32)
    add_custom_command(
        DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/Src/vs.hlsl
        OUTPUT
            ${CMAKE_BINARY_DIR}/vs.dxil
        COMMAND
            ${DXC_PATH}
            -T vs_6_0
            -E VSMain
            ${CMAKE_CURRENT_SOURCE_DIR}/Src/vs.hlsl
            -Fo ${CMAKE_BINARY_DIR}/vs.dxil
            -Zi
            -Qembed_debug
            -O3
        VERBATIM
    )
elseif(APPLE)
    add_custom_command(
        DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/Src/vs.hlsl
        OUTPUT
            ${CMAKE_BINARY_DIR}/vs.metal
        COMMAND
            ${SHADERCROSS_PATH}
            -o ${CMAKE_BINARY_DIR}/vs.metal
            ${CMAKE_CURRENT_SOURCE_DIR}/Src/vs.hlsl
            -s HLSL
            -d MSL
            -t vertex
            -e VSMain
            --msl-version 1.2.0
            # -g
        VERBATIM
    )
    add_custom_command(
        DEPENDS
            ${CMAKE_BINARY_DIR}/vs.metal
        OUTPUT
            ${CMAKE_BINARY_DIR}/vs.mtlir
        COMMAND
            xcrun -sdk macosx
            metal
            -o vs.mtlir
            -c ${CMAKE_BINARY_DIR}/vs.metal
            -frecord-sources
            -gline-tables-only
        VERBATIM
    )
    add_custom_command(
        DEPENDS
            ${CMAKE_BINARY_DIR}/vs.mtlir
        OUTPUT
            ${CMAKE_BINARY_DIR}/vs.metallib
        COMMAND
            xcrun -sdk macosx
            metallib
            -o ${CMAKE_BINARY_DIR}/vs.metallib
            ${CMAKE_BINARY_DIR}/vs.mtlir
        VERBATIM
    )
endif()

# Fragment shader
if (WIN32)
    add_custom_command(
        DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/Src/fs.hlsl
        OUTPUT
            ${CMAKE_BINARY_DIR}/fs.dxil
        COMMAND
            ${DXC_PATH}
            -T ps_6_0
            -E FSMain
            ${CMAKE_CURRENT_SOURCE_DIR}/Src/fs.hlsl
            -Fo ${CMAKE_BINARY_DIR}/fs.dxil
            -Zi
            -Qembed_debug
            -O3
        VERBATIM
    )
elseif(APPLE)
    add_custom_command(
        DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/Src/fs.hlsl
        OUTPUT
            ${CMAKE_BINARY_DIR}/fs.metal
        COMMAND
            ${SHADERCROSS_PATH}
            -o ${CMAKE_BINARY_DIR}/fs.metal
            ${CMAKE_CURRENT_SOURCE_DIR}/Src/fs.hlsl
            -s HLSL
            -d MSL
            -t fragment
            -e FSMain
            --msl-version 1.2.0
            # -g
        VERBATIM
    )
    add_custom_command(
        DEPENDS
            ${CMAKE_BINARY_DIR}/fs.metal
        OUTPUT
            ${CMAKE_BINARY_DIR}/fs.mtlir
        COMMAND
            xcrun -sdk macosx
            metal
            -o fs.mtlir
            -c ${CMAKE_BINARY_DIR}/fs.metal
            -frecord-sources
            -gline-tables-only
        VERBATIM
    )
    add_custom_command(
        DEPENDS
            ${CMAKE_BINARY_DIR}/fs.mtlir
        OUTPUT
            ${CMAKE_BINARY_DIR}/fs.metallib
        COMMAND
            xcrun -sdk macosx
            metallib
            -o ${CMAKE_BINARY_DIR}/fs.metallib
            ${CMAKE_BINARY_DIR}/fs.mtlir
        VERBATIM
    )
endif()

# Compute Shader
if (WIN32)
    add_custom_command(
        DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/Src/cs.hlsl
        OUTPUT
            ${CMAKE_BINARY_DIR}/cs.dxil
            # ${CMAKE_BINARY_DIR}/cs.depfile
            # ${CMAKE_BINARY_DIR}/cs.rootsig.json
        COMMAND
            ${DXC_PATH}
            -enable-16bit-types
            -T cs_6_2 // Support for true fp16
            -E CSMain
            ${CMAKE_CURRENT_SOURCE_DIR}/Src/cs.hlsl
            -Fo ${CMAKE_BINARY_DIR}/cs.dxil
            # -Frs ${CMAKE_BINARY_DIR}/cs.rootsig.json
            -Zi
            -Qembed_debug
            -O3
            -ffinite-math-only
        #     -MF ${CMAKE_BINARY_DIR}/cs.depfile
        # DEPFILE
        #     ${CMAKE_BINARY_DIR}/cs.depfile
        VERBATIM
    )
elseif(APPLE)
    add_custom_command(
        DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/Src/cs.hlsl
        OUTPUT
            ${CMAKE_BINARY_DIR}/cs.metal
        COMMAND
            ${SHADERCROSS_PATH}
            -o ${CMAKE_BINARY_DIR}/cs.metal
            ${CMAKE_CURRENT_SOURCE_DIR}/Src/cs.hlsl
            -s HLSL
            -d MSL
            -t compute
            -e CSMain
            --msl-version 1.2.0
            # -g
        VERBATIM
    )
    add_custom_command(
        DEPENDS
            ${CMAKE_BINARY_DIR}/cs.metal
        OUTPUT
            ${CMAKE_BINARY_DIR}/cs.mtlir
        COMMAND
            xcrun -sdk macosx
            metal
            -o cs.mtlir
            -c ${CMAKE_BINARY_DIR}/cs.metal
            -frecord-sources
            -gline-tables-only
        VERBATIM
    )
    add_custom_command(
        DEPENDS
            ${CMAKE_BINARY_DIR}/cs.mtlir
        OUTPUT
            ${CMAKE_BINARY_DIR}/cs.metallib
        COMMAND
            xcrun -sdk macosx
            metallib
            -o ${CMAKE_BINARY_DIR}/cs.metallib
            ${CMAKE_BINARY_DIR}/cs.mtlir
        VERBATIM
    )
endif()

if (APPLE)
    add_custom_target(Shaders
        DEPENDS
            ${CMAKE_BINARY_DIR}/vs.metallib
            ${CMAKE_BINARY_DIR}/fs.metallib
            ${CMAKE_BINARY_DIR}/cs.metallib
    )
elseif(WIN32)
    add_custom_target(Shaders
        DEPENDS
            ${CMAKE_BINARY_DIR}/vs.dxil
            ${CMAKE_BINARY_DIR}/fs.dxil
            ${CMAKE_BINARY_DIR}/cs.dxil
    )
endif()

add_executable(ComputeDct
    src/Main.cpp
)
target_compile_features(ComputeDct
    PUBLIC
        cxx_std_17
)
target_include_directories(ComputeDct
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/Src
        ${Stb_INCLUDE_DIR}
)
target_compile_options(ComputeDct
    PRIVATE
        -fsanitize=address,undefined
)
target_link_libraries(ComputeDct
    PRIVATE
        imgui::imgui
        SDL3::SDL3
        spdlog::spdlog
)
target_link_options(ComputeDct
    PRIVATE
        -fsanitize=address,undefined
)
add_dependencies(ComputeDct
    Shaders
)

if (MSVC)
    target_compile_definitions(ComputeDct
        PRIVATE
            _CRT_SECURE_NO_WARNINGS
    )
endif()
