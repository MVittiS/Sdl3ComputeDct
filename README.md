# SDL 3 Compute DCT

This repository contains a simple application using SDL_gpu that applies a JPEG-style DCT quantization effect to images from a camera feed in real-time, using compute shaders to let you adjust the quantization factors. It's a simple demonstration of:

* Reading image data from a camera using the [SDL3 Camera API](https://wiki.libsdl.org/SDL3/CategoryCamera), including waiting for permissions on systems like macOS
* Using HLSL shaders with [SDL_shadercross](https://github.com/libsdl-org/SDL_shadercross) on non-Windows systems
* Using `groupshared` memory and barriers to perform DCT, quantization, and IDCT in a single compute dispatch
* Applying [imgui](https://github.com/ocornut/imgui)'s SDL3 + SDL_gpu backend to a simple interactive app

## Limitations

* The code assumes that the camera feed is in the NV12_YUV format (as reported by SDL). If that's not the case, the UI will display a warning message.
* The code assumes that your camera's image is 16:9. Using any other aspect ratio (like 4:3) will apply the DCT effect correctly, but the output texture will be rendered stretched.
* The window is always 1280x720, non-resizable. Most webcams I came across output in this resolution.

## Build Prerequisites

Make sure you have the following installed in your machine:

* CMake (through Kitware's installer, `winget` on Windows, `brew` on macOS, or your package manager on Linux)
* A C++17 compiler
  * On Windows, get [Visual Studio](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community)
  * On macOS, get [Xcode](https://apps.apple.com/us/app/xcode/id497799835)
  * On Linux, get a compiler toolchain like `gcc` or `clang`
* [vcpkg](https://github.com/microsoft/vcpkg), or install `imgui`, `sdl3`, and `spdlog` by compiling their source or through your package managers of choice
* For non-Windows systems, you need `SDL_shadercross` already built and installed somewhere

## Building

```bash
# If using Ninja and vcpkg on Windows
cmake \
    -S . \
    -B Build \
    -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE=(...)/vcpkg/scripts/buildsystems/vcpkg.cmake
ninja

# If using Xcode and vcpkg on macOS
cmake \
    -S . \
    -B Build \
    -G Xcode \
    -DSHADERCROSS_PATH=(...)/SDL_shadercross/build/shadercross \
    -DCMAKE_TOOLCHAIN_FILE=(...)/vcpkg/scripts/buildsystems/vcpkg.cmake
open Build/Sdl3ComputeDct.xcodeproj

# If you installed the dependencies with a package manager, on Linux
$> cmake ..
$> cmake --build .
```

Make sure to run the executable from the built directory so that it picks up the compiled shader files (`.dxil` on Windows, `.metallib` on macOS, and `.spirv` on Linux).
