struct ProcessingParams {
    uint frameWidth;
    uint frameHeight;
    uint rowByteStride;
    uint uvByteOffset;

    // Needs padding due to D3D's annoying constant buffer alignment rules.
    // Without this, the last few elements of the struct are dropped.
    uint padding[60];

    float quantTable[8][8];
    float quantTableInv[8][8];
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

float QuantizeFloat(float x, float quantFactor, float invQuantFactor) {
    const float quantX = round(x * invQuantFactor);
    return (quantX * quantFactor);
}

int4 Uint32ToUVInt(uint x) {
    int4 bytes;
    bytes[0] = int((x >>  0) & 0xFF) - 0x80;
    bytes[1] = int((x >>  8) & 0xFF) - 0x80;
    bytes[2] = int((x >> 16) & 0xFF) - 0x80;
    bytes[3] = int((x >> 24) & 0xFF) - 0x80;

    return bytes;
}

[numthreads(8, 8, 1)]
void CSMain(uint3 globalId : SV_DispatchThreadId
    , uint3 localId : SV_GroupThreadID
    , uint3 blockId : SV_GroupId
) {
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

    // Stage 1 - loading shared memory with 4Y, 1U, and 1V tiles
    // DEBUG: load only Y values. Each thread loads 4 values, so first 4 threads read entire 16-element row,
    //  with 16 rows being read each by 4 threads. These values are all put in shared memory.
    
    /* For 1920x1080, we should load:
    /        0    1    2    3    4    5    6    7
    /   0    0    4    8   12 1920 1924 1928 1932
    /   1 3840 3844 ...
    /   2
    /   3
    /   4
    /   5
    /   6
    /   7
    */
    const bool isOddRow = (localId.x >= 4);
    const uint rowOffset = ((localId.x % 4) * 4)
        + (isOddRow ? params.rowByteStride : 0)
        + (blockId.x * 16)
    ;
    const uint colOffset = globalId.y * params.rowByteStride * 2;
    const uint yValues = inputRawYuvFrame.Load(rowOffset + colOffset);

    const bool isLocalOddRow = (localId.x >= 4);
    const uint localRowToStore = 2 * localId.y + uint(isLocalOddRow);
    const uint localColToStore = 4 * (localId.x - (isLocalOddRow ? 4 : 0));

    y[localRowToStore][localColToStore + 0] = ((yValues >>  0) & 0xFF) * (1.0f / 255.0f);
    y[localRowToStore][localColToStore + 1] = ((yValues >>  8) & 0xFF) * (1.0f / 255.0f);
    y[localRowToStore][localColToStore + 2] = ((yValues >> 16) & 0xFF) * (1.0f / 255.0f);
    y[localRowToStore][localColToStore + 3] = ((yValues >> 24) & 0xFF) * (1.0f / 255.0f);

    // For UV components, we only use half the threads.
    if (localId.x < 4) {
        /* For 1920x1080, we should load:
        /        0    1    2    3    4    5    6    7    8    9   ...   959
        /   0    0    4    8   12    -    -    -    -   16   20
        /   1 1920 1924 ...
        /   2
        /   3
        /   4
        /   5
        /   6
        /   7
        */

        const uint uvByteCol = (blockId.x * 16) + (localId.x * 4);
        const uint uvByteRow = globalId.y * params.rowByteStride;
        const int4 uvSamples = Uint32ToUVInt(inputRawYuvFrame.Load(params.uvByteOffset + uvByteCol + uvByteRow));

        u[localId.y][2 * localId.x + 0] = uvSamples[0] * (1.0f / 128.0f);
        v[localId.y][2 * localId.x + 0] = uvSamples[1] * (1.0f / 128.0f);
        u[localId.y][2 * localId.x + 1] = uvSamples[2] * (1.0f / 128.0f);
        v[localId.y][2 * localId.x + 1] = uvSamples[3] * (1.0f / 128.0f);
    }

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
    localDctY[0] = QuantizeFloat(localDctY[0], params.quantTable[localId.y][localId.x], params.quantTableInv[localId.y][localId.x]);
    localDctY[1] = QuantizeFloat(localDctY[1], params.quantTable[localId.y][localId.x], params.quantTableInv[localId.y][localId.x]);
    localDctY[2] = QuantizeFloat(localDctY[2], params.quantTable[localId.y][localId.x], params.quantTableInv[localId.y][localId.x]);
    localDctY[3] = QuantizeFloat(localDctY[3], params.quantTable[localId.y][localId.x], params.quantTableInv[localId.y][localId.x]);
    localDctU = QuantizeFloat(localDctU, params.quantTable[localId.y][localId.x], params.quantTableInv[localId.y][localId.x]);
    localDctV = QuantizeFloat(localDctV, params.quantTable[localId.y][localId.x], params.quantTableInv[localId.y][localId.x]);

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
    // From https://paulbourke.net/dataformats/nv12/
//    r = y + 1.402 * v;
//    g = y - 0.34414 * u - 0.71414 * v;
//    b = y + 1.772 * u;
    
    // Elements are stored col by col, then row after row.
    // Just like what you'd expect visually, huh.
    const float3x3 yuvToRgb = float3x3 (
        1.0,    0.0,      1.402,
        1.0,   -0.34414, -0.71414,
        1.0,    1.772,    0.0
    );

    const float3 zeros = float3(0, 0, 0);
    const float3 ones = float3(1, 1, 1);

    const float3 cy0x0 = float3(y[2 * localId.y + 0][2 * localId.x + 0], u[localId.y][localId.x], v[localId.y][localId.x]);
    const float3 cy0x1 = float3(y[2 * localId.y + 0][2 * localId.x + 1], u[localId.y][localId.x], v[localId.y][localId.x]);
    const float3 cy1x0 = float3(y[2 * localId.y + 1][2 * localId.x + 0], u[localId.y][localId.x], v[localId.y][localId.x]);
    const float3 cy1x1 = float3(y[2 * localId.y + 1][2 * localId.x + 1], u[localId.y][localId.x], v[localId.y][localId.x]);

    outputTexture[(2 * globalId.xy) + x0y0] = clamp(mul(yuvToRgb, cy0x0), zeros, ones);
    outputTexture[(2 * globalId.xy) + x1y0] = clamp(mul(yuvToRgb, cy0x1), zeros, ones);
    outputTexture[(2 * globalId.xy) + x0y1] = clamp(mul(yuvToRgb, cy1x0), zeros, ones);
    outputTexture[(2 * globalId.xy) + x1y1] = clamp(mul(yuvToRgb, cy1x1), zeros, ones);
    
#endif
}
