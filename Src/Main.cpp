#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_sdlgpu3.h>

#include <SDL3/SDL_camera.h>
#include <SDL3/SDL_error.h>
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_gpu.h>
#include <SDL3/SDL_init.h>
#include <SDL3/SDL_oldnames.h>
#include <SDL3/SDL_stdinc.h>
#include <SDL3/SDL_timer.h>
#include <SDL3/SDL_video.h>

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <random>
#include <spdlog/spdlog.h>
#include <vector>

struct ConstantBufferData {
    Uint32 frameWidth;
    Uint32 frameHeight;
    Uint32 rowByteStride;
    Uint32 uvByteOffset;
    Uint32 padding[60];
    float quantTable[8][8];
    float quantTableInv[8][8];
};
static_assert((sizeof(ConstantBufferData) % 256 == 0), "ConstantBufferData needs to be sized a multiple of 256 bytes. D3D requires that.");

int main(int argc, char** args) {
    const bool debugMode = true;
    const char* preferredGpu = nullptr;

    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_CAMERA);

    SDL_GPUDevice* gpu = SDL_CreateGPUDevice(
#if defined(__APPLE__)
        SDL_GPU_SHADERFORMAT_METALLIB,
#elif defined(_WIN32)
        SDL_GPU_SHADERFORMAT_DXIL,
#endif
        debugMode,
        preferredGpu
    );
    if (gpu == nullptr) {
        spdlog::error("Could not create GPU.");
    }

    spdlog::info("Created GPU with driver {}", SDL_GetGPUDeviceDriver(gpu));

    SDL_Camera* webcam = [&] {
        int cameraCount = 0;
        SDL_CameraID* cameras = SDL_GetCameras(&cameraCount);
        if (cameras == nullptr) {
            spdlog::error("No cameras attached to this system. Error: {}", SDL_GetError());
            exit(-1);
        }
        for (int idx = 0; idx < cameraCount; ++idx) {
            SDL_Camera* tryCamera = SDL_OpenCamera(cameras[idx], nullptr);
            if (tryCamera != nullptr) {
                SDL_free(cameras);
                return tryCamera;
            }
        }

        spdlog::error("Could not open any cameras out of {} options.", cameraCount);
        exit(-1);
    }();

    {
        SDL_Event cameraEvent;
        cameraEvent.type = SDL_EVENT_LAST;
        int permission = 0;

        while (permission == 0) {
            permission = SDL_GetCameraPermissionState(webcam);
            if (permission == 1) {
                spdlog::info("Camera access granted.");
                break;
            }
            else if (permission == -1) {
                spdlog::error("User denied camera access.");
                exit(-1);
            }
            SDL_Delay(200);
        }
    }

    SDL_CameraSpec webcamFormat = {};
    if (!SDL_GetCameraFormat(webcam, &webcamFormat)) {
        spdlog::error("Could not get camera format.");
        exit(1);
    }
    else {
        spdlog::info("Camera spec:\n"
            "- Format: {:x}\n"
            "- Colorspace: {:x}\n"
            "- Width: {}\n"
            "- Height: {}\n"
            "- Framerate: {}/{} ({})"
            , Uint64(webcamFormat.format)
            , Uint64(webcamFormat.colorspace)
            , webcamFormat.width
            , webcamFormat.height
            , webcamFormat.framerate_numerator
            , webcamFormat.framerate_denominator
            , float(webcamFormat.framerate_numerator) / float(webcamFormat.framerate_denominator)
        );
    }
    // Frame is 1 plane of Y in full res, and one interleaved U+V plane in half-res (width * helf-height)
    const Uint32 webcamYuvFrameSizeBytes = (3 * webcamFormat.width * webcamFormat.height) / 2;
    ConstantBufferData cbufData;
    for (int idx = 0; idx < 60; ++idx) {
        cbufData.padding[idx] = idx;
    }
    for (int row = 0; row < 8; ++row) {
        for (int col = 0; col < 8; ++col) {
            const float quantVal = float((2 * row + 1) * (2 * col + 1)) / 255.0f;
            cbufData.quantTable[row][col] = quantVal;
            cbufData.quantTableInv[row][col] = 1.0f / quantVal;
        }
    }
    cbufData.frameWidth = webcamFormat.width;
    cbufData.frameHeight = webcamFormat.height;
    cbufData.rowByteStride = webcamFormat.width;
    cbufData.uvByteOffset = webcamFormat.width * webcamFormat.height;

    // Now, create window, swapchain texture, and pipelines.
    SDL_Window* window = SDL_CreateWindow("FriedCamera", 1280, 720, /*SDL_WINDOW_HIGH_PIXEL_DENSITY*/ 0);
    SDL_ClaimWindowForGPUDevice(gpu, window);
    SDL_SetGPUSwapchainParameters(gpu, window, SDL_GPU_SWAPCHAINCOMPOSITION_SDR, SDL_GPU_PRESENTMODE_VSYNC);


    // Setup Dear ImGui context - most of the code is straight from https://github.com/ocornut/imgui/pull/8163/files#diff-3ef28c917731f41f2381f195496078a9eb430fe357c9ef11cfb9226024282777
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Platform/Renderer backends
    ImGui_ImplSDL3_InitForOther(window);
    ImGui_ImplSDLGPU3_InitInfo init_info = {};
    init_info.Device = gpu;
    init_info.ColorTargetFormat = SDL_GetGPUSwapchainTextureFormat(gpu, window);
    init_info.MSAASamples = SDL_GPU_SAMPLECOUNT_1;
    ImGui_ImplSDLGPU3_Init(&init_info);

    const auto shaderFormats = SDL_GetGPUShaderFormats(gpu);
#if defined(__APPLE__)
    if (!(shaderFormats & SDL_GPU_SHADERFORMAT_METALLIB)) {
        spdlog::error("This GPU doesn't support Metal.");
    }
#elif defined(_WIN32)
    if (!(shaderFormats & SDL_GPU_SHADERFORMAT_DXIL)) {
        spdlog::error("This GPU doesn't support DXIL.");
    }
#endif

    SDL_GPUShader* vertexShader = [&] {
        size_t shaderSize;
#if defined(__APPLE__)
        const auto* shaderCode = static_cast<Uint8*>(SDL_LoadFile("vs.metallib", &shaderSize));
#elif defined(_WIN32)
        const auto* shaderCode = static_cast<Uint8*>(SDL_LoadFile("vs.dxil", &shaderSize));
#endif
        if (shaderCode == nullptr) {
            spdlog::error("Vertex shader could not be found!!");
            exit(-1);
        }

        SDL_GPUShaderCreateInfo vertexShaderCreateInfo{0};
        vertexShaderCreateInfo.code = shaderCode;
        vertexShaderCreateInfo.code_size = shaderSize;
        vertexShaderCreateInfo.entrypoint = "VSMain";
#if defined(__APPLE__)
        vertexShaderCreateInfo.format = SDL_GPU_SHADERFORMAT_METALLIB;
#elif defined(_WIN32)
        vertexShaderCreateInfo.format = SDL_GPU_SHADERFORMAT_DXIL;
#endif
        vertexShaderCreateInfo.stage = SDL_GPU_SHADERSTAGE_VERTEX;
        vertexShaderCreateInfo.num_storage_buffers = 0;
        vertexShaderCreateInfo.num_uniform_buffers = 0;

        SDL_GPUShader* vertexShader = SDL_CreateGPUShader(gpu, &vertexShaderCreateInfo);
        if (vertexShader == nullptr) {
            spdlog::error("Failed to create vertex shader!");
            exit(-1);
        }
        else {
            spdlog::info("Vertex shader created.");
        }

        return vertexShader;
    }();

    SDL_GPUShader* fragShader = [&] {
        size_t shaderSize;
#if defined(__APPLE__)
        const auto* shaderCode = static_cast<Uint8*>(SDL_LoadFile("fs.metallib", &shaderSize));
#elif defined(_WIN32)
        const auto* shaderCode = static_cast<Uint8*>(SDL_LoadFile("fs.dxil", &shaderSize));
#endif
        if (shaderCode == nullptr) {
            spdlog::error("Fragment shader could not be found!!");
            exit(-1);
        }

        SDL_GPUShaderCreateInfo fragShaderCreateInfo{0};
        fragShaderCreateInfo.code = shaderCode;
        fragShaderCreateInfo.code_size = shaderSize;
        fragShaderCreateInfo.entrypoint = "FSMain";
#if defined(__APPLE__)
        fragShaderCreateInfo.format = SDL_GPU_SHADERFORMAT_METALLIB;
#elif defined(_WIN32)
        fragShaderCreateInfo.format = SDL_GPU_SHADERFORMAT_DXIL;
#endif
        fragShaderCreateInfo.stage = SDL_GPU_SHADERSTAGE_FRAGMENT;
        // Yes, we need a sampler object for the frag shader.
        fragShaderCreateInfo.num_samplers = 1;

        SDL_GPUShader* fragShader = SDL_CreateGPUShader(gpu, &fragShaderCreateInfo);
        if (fragShader == nullptr) {
            spdlog::error("Failed to create fragment shader!");
            exit(-1);
        }
        else {
            spdlog::info("Fragment shader created.");
        }

        return fragShader;
    }();


    const auto graphicsPipelineInfo = [&] {
        SDL_GPUGraphicsPipelineCreateInfo graphicsPipelineInfo{0};
        graphicsPipelineInfo.vertex_shader = vertexShader;
        graphicsPipelineInfo.fragment_shader = fragShader;
        graphicsPipelineInfo.depth_stencil_state = [&] {
            SDL_GPUDepthStencilState dsState{};
            dsState.enable_depth_test = false;
            dsState.enable_depth_write = false;
            dsState.compare_op = SDL_GPU_COMPAREOP_NEVER;
            return dsState;
        }();
        graphicsPipelineInfo.primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLESTRIP;
        graphicsPipelineInfo.rasterizer_state = [&] {
            SDL_GPURasterizerState rasterState{};
             rasterState.cull_mode = SDL_GPU_CULLMODE_BACK;
            rasterState.fill_mode = SDL_GPU_FILLMODE_FILL;
            rasterState.front_face = SDL_GPU_FRONTFACE_CLOCKWISE;
            return rasterState;
        }();
        graphicsPipelineInfo.target_info = [&] {
            static const auto tgtDesc = [&gpu, &window] {
                SDL_GPUColorTargetDescription tgtDesc;
                tgtDesc.format = SDL_GetGPUSwapchainTextureFormat(gpu, window);
				tgtDesc.blend_state = [] {
                    SDL_GPUColorTargetBlendState blendState{};
                    blendState.enable_blend = false;
                    return blendState;
                }();
                return tgtDesc;
            }();
            SDL_GPUGraphicsPipelineTargetInfo tgtInfo{0};
            tgtInfo.num_color_targets = 1;
            tgtInfo.color_target_descriptions = &tgtDesc;
            tgtInfo.has_depth_stencil_target = false;
            return tgtInfo;
        }();

        graphicsPipelineInfo.vertex_input_state.num_vertex_buffers = 0;
        graphicsPipelineInfo.vertex_input_state.num_vertex_attributes = 0;

        graphicsPipelineInfo.vertex_input_state.vertex_buffer_descriptions = nullptr;
        graphicsPipelineInfo.vertex_input_state.vertex_attributes = nullptr;

        return graphicsPipelineInfo;
    }();

    SDL_GPUGraphicsPipeline* gfxPipe = SDL_CreateGPUGraphicsPipeline(gpu, &graphicsPipelineInfo);

    if (gfxPipe == nullptr) {
        spdlog::error("Failed to create graphics pipeline!");
        return -1;
    }
    else {
        spdlog::info("Graphics pipeline created.");
    }

    SDL_GPUComputePipeline* computePipe = [&] {
        size_t shaderSize;
    #if defined(__APPLE__)
        void* shaderCode = SDL_LoadFile("cs.metallib", &shaderSize);
    #elif defined(_WIN32)
        void* shaderCode = SDL_LoadFile("cs.dxil", &shaderSize);
    #endif

        SDL_GPUComputePipelineCreateInfo computePipeInfo = {0};
        computePipeInfo.code = reinterpret_cast<Uint8*>(shaderCode);
        computePipeInfo.code_size = shaderSize;
    #if defined(__APPLE__)
        computePipeInfo.entrypoint = "CSMain";
        computePipeInfo.format = SDL_GPU_SHADERFORMAT_METALLIB;
    #elif defined(_WIN32)
        computePipeInfo.entrypoint = "CSMain";
        computePipeInfo.format = SDL_GPU_SHADERFORMAT_DXIL;
    #endif
        computePipeInfo.num_readonly_storage_textures = 0;
        computePipeInfo.num_readwrite_storage_textures = 1;
        computePipeInfo.num_readonly_storage_buffers = 1;
        computePipeInfo.num_readwrite_storage_buffers = 0;
        computePipeInfo.num_samplers = 0;
        computePipeInfo.num_uniform_buffers = 1;
        computePipeInfo.threadcount_x = 8;
        computePipeInfo.threadcount_y = 8;
        computePipeInfo.threadcount_z = 1;

        SDL_GPUComputePipeline* computePipe = SDL_CreateGPUComputePipeline(gpu, &computePipeInfo);
        
        if (computePipe == nullptr) {
            spdlog::error("Failed to create compute pipeline!");
        }
        else {
            spdlog::info("Compute pipeline created.");
        }

        return computePipe;
    }();

    SDL_GPUBuffer* gpuCameraFrame = [&] {
        SDL_GPUBufferCreateInfo gpuCameraFrameBufferInfo;
        gpuCameraFrameBufferInfo.usage = SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ;
        gpuCameraFrameBufferInfo.size = webcamYuvFrameSizeBytes;
        return SDL_CreateGPUBuffer(gpu, &gpuCameraFrameBufferInfo);
    }();
    if (gpuCameraFrame == nullptr) {
        spdlog::error("Could not create GPU camera frame. Are we out of VRAM?");
        exit(-1);
    }
    SDL_SetGPUBufferName(gpu, gpuCameraFrame, "GPU Camera Frame");

    SDL_GPUTexture *texFried;
    // Create RGB texture
    {
        SDL_GPUTextureCreateInfo texCreateInfo;
        texCreateInfo.type = SDL_GPU_TEXTURETYPE_2D;
        texCreateInfo.format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;
//        texCreateInfo.format = SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT;
        texCreateInfo.width = webcamFormat.width;
        texCreateInfo.height = webcamFormat.height;
        texCreateInfo.layer_count_or_depth = 1;
        texCreateInfo.num_levels = 1;
        texCreateInfo.sample_count = SDL_GPU_SAMPLECOUNT_1;
        texCreateInfo.usage = 0
            | SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ
            | SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE
        ;
        texFried = SDL_CreateGPUTexture(gpu, &texCreateInfo);
    }

    SDL_GPUSampler* sampler = [&]{
        SDL_GPUSamplerCreateInfo samplerInfo = {};
        samplerInfo.min_filter = SDL_GPU_FILTER_LINEAR;
        samplerInfo.mag_filter = SDL_GPU_FILTER_LINEAR;
        samplerInfo.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_LINEAR;
        samplerInfo.address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_MIRRORED_REPEAT;
        samplerInfo.address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_MIRRORED_REPEAT;
        samplerInfo.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_MIRRORED_REPEAT;
        samplerInfo.mip_lod_bias = .0f;
        samplerInfo.max_anisotropy = 8.f;
        samplerInfo.compare_op = SDL_GPU_COMPAREOP_INVALID;
        samplerInfo.min_lod = .0f;
        samplerInfo.max_lod = .0f;
        samplerInfo.enable_anisotropy = true;
        samplerInfo.enable_compare = false;

        SDL_GPUSampler* sampler = SDL_CreateGPUSampler(gpu, &samplerInfo);

        if (!sampler) {
            spdlog::error("Could not create sampler object!");
            exit(-1);
        }
        return sampler;
    }();

    SDL_SetGPUTextureName(gpu, texFried, "Output RGB (fried) Texture");

    SDL_GPUTransferBuffer* txBuffer = [&] {
        SDL_GPUTransferBufferCreateInfo txBufferInfo;
            txBufferInfo.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
            txBufferInfo.size = webcamYuvFrameSizeBytes;
            txBufferInfo.props = 0;
        return SDL_CreateGPUTransferBuffer(gpu, &txBufferInfo);
    }();
    if (txBuffer == nullptr) {
        spdlog::error("Could not create image upload buffer! Error: {}", SDL_GetError());
        exit(-1);
    }
    
#if 0 // Debug: capture images in advance so that we can close the camera when not in use.
    FILE* cameraOut = fopen("camera.raw", "wb");
    if (!cameraOut) {
        spdlog::error("Could not open 'camera.raw' file.");
        exit(-1);
    }
    std::vector<std::byte> cameraMem(webcamYuvFrameSizeBytes);
    for (int frame = 0; frame < 100; ++frame) {
        Uint64 frameTimestamp;
        SDL_Surface* cpuCameraSurface = SDL_AcquireCameraFrame(webcam, &frameTimestamp);
        if (cpuCameraSurface == nullptr) {
            SDL_Delay(5);
            continue;
        }
        
        {
            auto* txPointer = static_cast<Uint8*>(SDL_MapGPUTransferBuffer(gpu, txBuffer, false));
            std::copy_n(static_cast<const Uint8*>(cpuCameraSurface->pixels), webcamYuvFrameSizeBytes, txPointer);
            std::copy_n(static_cast<const std::byte*>(cpuCameraSurface->pixels), webcamYuvFrameSizeBytes, cameraMem.begin());
            SDL_UnmapGPUTransferBuffer(gpu, txBuffer);

        }
        SDL_ReleaseCameraFrame(webcam, cpuCameraSurface);
    }
    if (cameraOut) {
        fwrite(cameraMem.data(), 1, cameraMem.size(), cameraOut);
        fclose(cameraOut);
        cameraOut = nullptr;
    }
    SDL_CloseCamera(webcam);
#endif
    

    bool shouldExit = false;
    SDL_GPUFence* frameFence = nullptr;
    while (!shouldExit) {
        SDL_Event events;
        while(SDL_PollEvent(&events)) {
            ImGui_ImplSDL3_ProcessEvent(&events);
            switch (events.type) {
                case SDL_EVENT_QUIT:
                    shouldExit = true;
                    break;
                default:
                    break;
            }
        }

        if (frameFence) {
            SDL_WaitForGPUFences(gpu, true, &frameFence, 1);
            SDL_ReleaseGPUFence(gpu, frameFence);
            frameFence = nullptr;
        }
#if 1
        [[maybe_unused]] Uint64 frameTimestamp;
        SDL_Surface* cpuCameraSurface = SDL_AcquireCameraFrame(webcam, &frameTimestamp);
        if (cpuCameraSurface == nullptr) {
            SDL_Delay(5);
            continue;
        }

        {
            auto* txPointer = static_cast<Uint8*>(SDL_MapGPUTransferBuffer(gpu, txBuffer, false));
            std::copy_n(static_cast<const Uint8*>(cpuCameraSurface->pixels), webcamYuvFrameSizeBytes, txPointer);
            SDL_UnmapGPUTransferBuffer(gpu, txBuffer);
        }
        SDL_ReleaseCameraFrame(webcam, cpuCameraSurface);
#endif
        // Start the Dear ImGui frame
        ImGui_ImplSDLGPU3_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        static float crunchBase = 3.f;
        static float crunchX = 5.f;
        static float crunchY = 5.f;

        ImGui::SliderFloat("Crunch Base Factor", &crunchBase, 1, 128, "%.2f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Crunch Horizontal Factor", &crunchX, 0.1, 128, "%.2f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Crunch Vertical Factor", &crunchY, 0.1, 128, "%.2f", ImGuiSliderFlags_Logarithmic);

        for (int row = 0; row < 8; ++row) {
            for (int col = 0; col < 8; ++col) {
                const float quantVal = float((crunchY * row + crunchBase) * (crunchX * col + crunchBase)) / 255.0f;
                cbufData.quantTable[row][col] = quantVal;
                cbufData.quantTableInv[row][col] = 1.0f / quantVal;
            }
        }

        ImGui::Render();
        ImDrawData* imGuiDrawData = ImGui::GetDrawData();

        // Acquire swapchain
        SDL_GPUTexture* swapchainTexture;
        Uint32 swapchainWidth, swapchainHeight;
        SDL_GPUCommandBuffer* frameCmdBuf = SDL_AcquireGPUCommandBuffer(gpu); {
            SDL_WaitAndAcquireGPUSwapchainTexture(frameCmdBuf, window, &swapchainTexture, &swapchainWidth, &swapchainHeight);

            SDL_GPUCopyPass* uploadPass = SDL_BeginGPUCopyPass(frameCmdBuf); {
                SDL_GPUTransferBufferLocation cpuBufferLoc;
                    cpuBufferLoc.offset = 0;
                    cpuBufferLoc.transfer_buffer = txBuffer;
                SDL_GPUBufferRegion gpuBufferLoc;
                gpuBufferLoc.buffer = gpuCameraFrame;
                gpuBufferLoc.offset = 0;
                gpuBufferLoc.size = webcamYuvFrameSizeBytes;
                SDL_UploadToGPUBuffer(uploadPass, &cpuBufferLoc, &gpuBufferLoc, false);
            } SDL_EndGPUCopyPass(uploadPass);

            SDL_GPUStorageTextureReadWriteBinding outputTextureBinding = {0};
                outputTextureBinding.texture = texFried;

            static constexpr Uint32 numWriteTextures = 1;
            static constexpr Uint32 numWriteBuffers = 0;
            SDL_GPUComputePass* computePass = SDL_BeginGPUComputePass(frameCmdBuf, &outputTextureBinding, numWriteTextures, nullptr, numWriteBuffers);
            {
                SDL_BindGPUComputePipeline(computePass, computePipe);
                static constexpr Uint32 firstSlot = 0;
                static constexpr Uint32 numReadBuffers = 1;
                SDL_BindGPUComputeStorageBuffers(computePass, firstSlot, &gpuCameraFrame, numReadBuffers);
                static constexpr Uint32 constantBufferSlot = 0;
                SDL_PushGPUComputeUniformData(frameCmdBuf, constantBufferSlot, &cbufData, sizeof(ConstantBufferData));

                const Uint32 numBlockX = webcamFormat.width / 16;
                const Uint32 numBlockY = webcamFormat.height / 16;
                static constexpr Uint32 numBlockZ = 1;
                SDL_DispatchGPUCompute(computePass
                    , numBlockX
                    , numBlockY
                    , numBlockZ
                );
            } SDL_EndGPUComputePass(computePass);

            static constexpr Uint32 numColorTargets = 1;
            const SDL_GPUColorTargetInfo rtInfo = [&] {
                SDL_GPUColorTargetInfo rtInfo = {0};
                rtInfo.load_op = SDL_GPU_LOADOP_CLEAR;
                rtInfo.store_op = SDL_GPU_STOREOP_STORE;
                rtInfo.texture = swapchainTexture;
                return rtInfo;
            }();
            const SDL_GPUDepthStencilTargetInfo* dsInfo = nullptr;

            const SDL_GPUTextureSamplerBinding samplerBinding = [&] {
                SDL_GPUTextureSamplerBinding samplerBinding;
                samplerBinding.sampler = sampler;
                samplerBinding.texture = texFried;

                return samplerBinding;
            }();
            
            Imgui_ImplSDLGPU3_PrepareDrawData(imGuiDrawData, frameCmdBuf);

            SDL_GPURenderPass* gfxPass = SDL_BeginGPURenderPass(frameCmdBuf, &rtInfo, numColorTargets, dsInfo); {
                SDL_BindGPUGraphicsPipeline(gfxPass, gfxPipe);

                static constexpr Uint32 samplerSlot = 0;
                static constexpr Uint32 numSamplers = 1;
                SDL_BindGPUFragmentSamplers(gfxPass, samplerSlot, &samplerBinding, numSamplers);

                static constexpr Uint32 numVerts = 4;
                static constexpr Uint32 numInstances = 1;
                static constexpr Uint32 firstVert = 0;
                static constexpr Uint32 firstInstance = 0;
                SDL_DrawGPUPrimitives(gfxPass, numVerts, numInstances, firstVert, firstInstance);

                // Finally, render ImGui.
                ImGui_ImplSDLGPU3_RenderDrawData(imGuiDrawData, frameCmdBuf, gfxPass);
            } SDL_EndGPURenderPass(gfxPass);
        } frameFence = SDL_SubmitGPUCommandBufferAndAcquireFence(frameCmdBuf);
    }

    SDL_ReleaseGPUComputePipeline(gpu, computePipe);
    SDL_ReleaseGPUGraphicsPipeline(gpu, gfxPipe);

    SDL_ReleaseGPUTransferBuffer(gpu, txBuffer);
    SDL_ReleaseGPUBuffer(gpu, gpuCameraFrame);
    SDL_ReleaseGPUTexture(gpu, texFried);

    // SDL_free(shaderCode);
    SDL_DestroyGPUDevice(gpu);
    SDL_Quit();
    return 0;
}
