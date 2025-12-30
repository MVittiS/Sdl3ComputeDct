struct ProcessingParams {
    uint frameWidth;
    uint frameHeight;
    uint quantMatrix[8][8];
};

ByteAddressBuffer inputRawYuvFrame      : register(t0, space0);
RWTexture2D<float3> outputTexture       : register(u0, space1);
ConstantBuffer<ProcessingParams> params : register(b0, space2);

groupshared float y[16][16];
groupshared float u[8][8];
groupshared float v[8][8];

groupshared float dctY[16][16];
groupshared float dctU[8][8];
groupshared float dctV[8][8];

float QuantizeFloat(float x, uint quantFactor) {
    const float quantX = round(x / (float)quantFactor);
    return (quantX * quantFactor);
}

[numthreads(8, 8, 1)]
void CSMain(uint3 globalId : SV_DispatchThreadId, uint3 localId : SV_GroupThreadID) {
    // if (any(globalId.xy >= uint2(params.frameWidth, params.frameHeight))) {
    //     return;
    // }

    const float invSqrt8 = 1.0f/sqrt(8.0f);
    const float invSqrt4 = 0.5;

    const float halfDctCoeffs[8][8] = {
        {invSqrt8, invSqrt8, invSqrt8, invSqrt8, invSqrt8, invSqrt8, invSqrt8, invSqrt8},
        {1./invSqrt4, 0.92388/invSqrt4, 0.707107/invSqrt4, 0.382683/invSqrt4, 0./invSqrt4, -0.382683/invSqrt4, -0.707107/invSqrt4, -0.92388/invSqrt4},
        {1./invSqrt4, 0.707107/invSqrt4, 0./invSqrt4, -0.707107/invSqrt4, -1./invSqrt4, -0.707107/invSqrt4, 0./invSqrt4, 0.707107/invSqrt4},
        {1./invSqrt4, 0.382683/invSqrt4, -0.707107/invSqrt4, -0.92388/invSqrt4, 0./invSqrt4, 0.92388/invSqrt4, 0.707107/invSqrt4, -0.382683/invSqrt4},
        {1./invSqrt4, 0./invSqrt4, -1./invSqrt4, 0./invSqrt4, 1./invSqrt4, 0./invSqrt4, -1./invSqrt4, 0./invSqrt4},
        {1./invSqrt4, -0.382683/invSqrt4, -0.707107/invSqrt4, 0.92388/invSqrt4, 0./invSqrt4, -0.92388/invSqrt4, 0.707107/invSqrt4, 0.382683/invSqrt4},
        {1./invSqrt4, -0.707107/invSqrt4, 0./invSqrt4, 0.707107/invSqrt4, -1./invSqrt4, 0.707107/invSqrt4, 0./invSqrt4, -0.707107/invSqrt4},
        {1./invSqrt4, -0.92388/invSqrt4, 0.707107/invSqrt4, -0.382683/invSqrt4, 0./invSqrt4, 0.382683/invSqrt4, -0.707107/invSqrt4, 0.92388/invSqrt4}
    };

    const uint2 x0y0 = uint2(0, 0);
    const uint2 x1y0 = uint2(1, 0);
    const uint2 x0y1 = uint2(0, 1);
    const uint2 x1y1 = uint2(1, 1);

    // According to https://stackoverflow.com/a/20438735,
    // HLSL matrices are column-major. I hope this works.
    const float3x3 yuvMat = {
        0.299, 0.587, 0.114,
        -0.14713, -0.28886, 0.436,
        0.615, -0.515, -0.1
    };

    // Stage 1 - loading shared memory with 4Y, 1U, and 1V tiles
    const uint linearYIndex = dot(globalId.xy * 2, uint2(1, params.frameWidth)) / 4; // ByteAddressBuffer reads uint32
    const uint linearUVOffset = params.frameWidth * params.frameHeight / 4;
    const uint linearUVIndex = linearUVOffset + dot(globalId.xy, uint2(2, params.frameWidth / 2)) / 4;

    y[2 * localId.y + 0][2 * localId.x + 0] = (inputRawYuvFrame.Load(linearYIndex) >> 0) & 0xFFu;
    y[2 * localId.y + 0][2 * localId.x + 1] = (inputRawYuvFrame.Load(linearYIndex) >> 8) & 0xFFu;
    y[2 * localId.y + 1][2 * localId.x + 0] = (inputRawYuvFrame.Load(linearYIndex + params.frameWidth / 4) >> 0) & 0xFFu;
    y[2 * localId.y + 1][2 * localId.x + 1] = (inputRawYuvFrame.Load(linearYIndex + params.frameWidth / 4) >> 8) & 0xFFu;

    u[localId.y][localId.x] = (inputRawYuvFrame.Load(linearUVIndex) >> 0) & 0xFFu;
    v[localId.y][localId.x] = (inputRawYuvFrame.Load(linearUVIndex) >> 8) & 0xFFu;

    GroupMemoryBarrierWithGroupSync();

    // Stage 2 - DCT and destructive quantization

    dctY[2 * localId.y + 0][2 * localId.x + 0] = .0f;
    dctY[2 * localId.y + 0][2 * localId.x + 1] = .0f;
    dctY[2 * localId.y + 1][2 * localId.x + 0] = .0f;
    dctY[2 * localId.y + 1][2 * localId.x + 1] = .0f;

    dctU[localId.y][localId.x] = .0f;
    dctV[localId.y][localId.x] = .0f;

    float4 localDctY = .0f;
    float localDctU = .0f;
    float localDctV = .0f;

    for (int row = 0; row != 8; ++row) {
        for (int col = 0; col != 8; ++col) {
            const float coeff = halfDctCoeffs[localId.y][row] * halfDctCoeffs[localId.x][col];
            localDctY[0] += y[row + 0][col + 0] * coeff;
            localDctY[1] += y[row + 0][col + 8] * coeff;
            localDctY[2] += y[row + 8][col + 0] * coeff;
            localDctY[3] += y[row + 8][col + 8] * coeff;
            localDctU += u[row][col] * coeff;
            localDctV += v[row][col] * coeff;
        }
    }

    const uint quantMatOffset = dot(localId.xy, uint2(1, 8));
    localDctY[0] = QuantizeFloat(localDctY[0], params.quantMatrix[localId.y][localId.x]);
    localDctY[1] = QuantizeFloat(localDctY[1], params.quantMatrix[localId.y][localId.x]);
    localDctY[2] = QuantizeFloat(localDctY[2], params.quantMatrix[localId.y][localId.x]);
    localDctY[3] = QuantizeFloat(localDctY[3], params.quantMatrix[localId.y][localId.x]);
    localDctU = QuantizeFloat(localDctU, params.quantMatrix[localId.y][localId.x]);
    localDctV = QuantizeFloat(localDctV, params.quantMatrix[localId.y][localId.x]);

    dctY[2 * localId.y + 0][2 * localId.x + 0] = localDctY[0];
    dctY[2 * localId.y + 0][2 * localId.x + 1] = localDctY[1];
    dctY[2 * localId.y + 1][2 * localId.x + 0] = localDctY[2];
    dctY[2 * localId.y + 1][2 * localId.x + 1] = localDctY[3];

    dctU[localId.y][localId.x] = localDctU;
    dctV[localId.y][localId.x] = localDctV;

    GroupMemoryBarrierWithGroupSync();

    // Stage 3 - IDCT and write to texture
    float4 localY = .0f;
    float localU = .0f;
    float localV = .0f;

    for (int row = 0; row != 8; ++row) {
        for (int col = 0; col != 8; ++col) {
            const float coeff = halfDctCoeffs[localId.y][row] * halfDctCoeffs[localId.x][col];
            localY[0] += dctY[row + 0][col + 0] * coeff;
            localY[1] += dctY[row + 0][col + 8] * coeff;
            localY[2] += dctY[row + 8][col + 0] * coeff;
            localY[3] += dctY[row + 8][col + 8] * coeff;
            localU += dctU[row][col] * coeff;
            localV += dctV[row][col] * coeff;
        }
    }

    const float normFactor = 1.0f/255.0f;

    // Now, let each thread write to the texture.
#if 0
    outputTexture[(2 * globalId.xy) + x0y0] = float3(localY[0], localU, localV) * normFactor;
    outputTexture[(2 * globalId.xy) + x1y0] = float3(localY[1], localU, localV) * normFactor;
    outputTexture[(2 * globalId.xy) + x0y1] = float3(localY[2], localU, localV) * normFactor;
    outputTexture[(2 * globalId.xy) + x1y1] = float3(localY[3], localU, localV) * normFactor;
#else
    outputTexture[(2 * globalId.xy) + x0y0] = float3(y[2 * localId.y + 0][2 * localId.x + 0], u[localId.y][localId.x], v[localId.y][localId.x]) * normFactor;
    outputTexture[(2 * globalId.xy) + x1y0] = float3(y[2 * localId.y + 0][2 * localId.x + 1], u[localId.y][localId.x], v[localId.y][localId.x]) * normFactor;
    outputTexture[(2 * globalId.xy) + x0y1] = float3(y[2 * localId.y + 1][2 * localId.x + 0], u[localId.y][localId.x], v[localId.y][localId.x]) * normFactor;
    outputTexture[(2 * globalId.xy) + x1y1] = float3(y[2 * localId.y + 1][2 * localId.x + 1], u[localId.y][localId.x], v[localId.y][localId.x]) * normFactor;
#endif
}
