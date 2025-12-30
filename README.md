# Sdl3ComputeDct

This is an SDL3 GPU sample using a compute shader to apply a DCT quantization effect to an image coming from a video source.

## How to build

Make sure you have the following installed in your machine:

- CMake (through Kitware's installer, `winget` on Windows, or `brew` on macOS)
- [vcpkg](https://github.com/microsoft/vcpkg), or install `sdl3` and `spdlog` through your package managers of choice

To configure the project:

```bash
$> mkdir build && cd build

# If using Ninja and vcpkg
$> cmake .. -G Ninja -DCMAKE_TOOLCHAIN_FILE=/(...)/vcpkg/scripts/buildsystems/vcpkg.cmake
$> ninja

# If you installed sdl3 and spdlog with another package manager
$> cmake ..
$> cmake --build .
```

Make sure to run the executable from the built directory so that it picks up the shader files.
