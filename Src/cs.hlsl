struct ProcessingParams {
    uint frameWidth;
    uint frameHeight;
    uint rowByteStride;
    uint uvByteOffset;

    // Needs padding due to D3D's annoying constant buffer alignment rules.
    // Without this, the last few elements of the struct are dropped.
    uint4 padding[15];
    
    // D3D is also annoying with arrays; you need to use <type>4 for
    //  constant buffers, or they get promoted automatically. Ugh...
    float4 quantTable[8][2];
    float4 quantTableInv[8][2];
};

ByteAddressBuffer inputRawYuvFrame      : register(t0, space0);
RWTexture2D<float4> outputTexture       : register(u0, space1);
ConstantBuffer<ProcessingParams> params : register(b0, space2);

groupshared half y[16][16];    // 512B
groupshared half u[8][8];      // 128B
groupshared half v[8][8];      // 128B

groupshared half dctY[16][16]; // 512B
groupshared half dctU[8][8];   // 128B
groupshared half dctV[8][8];   // 128B

// Total shared memory per threadgroup: 1.5KiB

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

    const double pi = 3.14159265359;
    const float dctCoeffs[8][8] = {
        {invSqrt8, invSqrt8, invSqrt8, invSqrt8, invSqrt8, invSqrt8, invSqrt8, invSqrt8},
        {invSqrt4*cos(1 * 1 * pi / 16.0), invSqrt4*cos(1 * 3 * pi / 16.0), invSqrt4*cos(1 * 5 * pi / 16.0), invSqrt4*cos(1 * 7 * pi / 16.0), invSqrt4*cos(1 * 9 * pi / 16.0), invSqrt4*cos(1 * 11 * pi / 16.0), invSqrt4*cos(1 * 13 * pi / 16.0), invSqrt4*cos(1 * 15 * pi / 16.0)},
        {invSqrt4*cos(2 * 1 * pi / 16.0), invSqrt4*cos(2 * 3 * pi / 16.0), invSqrt4*cos(2 * 5 * pi / 16.0), invSqrt4*cos(2 * 7 * pi / 16.0), invSqrt4*cos(2 * 9 * pi / 16.0), invSqrt4*cos(2 * 11 * pi / 16.0), invSqrt4*cos(2 * 13 * pi / 16.0), invSqrt4*cos(2 * 15 * pi / 16.0)},
        {invSqrt4*cos(3 * 1 * pi / 16.0), invSqrt4*cos(3 * 3 * pi / 16.0), invSqrt4*cos(3 * 5 * pi / 16.0), invSqrt4*cos(3 * 7 * pi / 16.0), invSqrt4*cos(3 * 9 * pi / 16.0), invSqrt4*cos(3 * 11 * pi / 16.0), invSqrt4*cos(3 * 13 * pi / 16.0), invSqrt4*cos(3 * 15 * pi / 16.0)},
        {invSqrt4*cos(4 * 1 * pi / 16.0), invSqrt4*cos(4 * 3 * pi / 16.0), invSqrt4*cos(4 * 5 * pi / 16.0), invSqrt4*cos(4 * 7 * pi / 16.0), invSqrt4*cos(4 * 9 * pi / 16.0), invSqrt4*cos(4 * 11 * pi / 16.0), invSqrt4*cos(4 * 13 * pi / 16.0), invSqrt4*cos(4 * 15 * pi / 16.0)},
        {invSqrt4*cos(5 * 1 * pi / 16.0), invSqrt4*cos(5 * 3 * pi / 16.0), invSqrt4*cos(5 * 5 * pi / 16.0), invSqrt4*cos(5 * 7 * pi / 16.0), invSqrt4*cos(5 * 9 * pi / 16.0), invSqrt4*cos(5 * 11 * pi / 16.0), invSqrt4*cos(5 * 13 * pi / 16.0), invSqrt4*cos(5 * 15 * pi / 16.0)},
        {invSqrt4*cos(6 * 1 * pi / 16.0), invSqrt4*cos(6 * 3 * pi / 16.0), invSqrt4*cos(6 * 5 * pi / 16.0), invSqrt4*cos(6 * 7 * pi / 16.0), invSqrt4*cos(6 * 9 * pi / 16.0), invSqrt4*cos(6 * 11 * pi / 16.0), invSqrt4*cos(6 * 13 * pi / 16.0), invSqrt4*cos(6 * 15 * pi / 16.0)},
        {invSqrt4*cos(7 * 1 * pi / 16.0), invSqrt4*cos(7 * 3 * pi / 16.0), invSqrt4*cos(7 * 5 * pi / 16.0), invSqrt4*cos(7 * 7 * pi / 16.0), invSqrt4*cos(7 * 9 * pi / 16.0), invSqrt4*cos(7 * 11 * pi / 16.0), invSqrt4*cos(7 * 13 * pi / 16.0), invSqrt4*cos(7 * 15 * pi / 16.0)},
    };

    const uint2 x0y0 = uint2(0, 0);
    const uint2 x1y0 = uint2(1, 0);
    const uint2 x0y1 = uint2(0, 1);
    const uint2 x1y1 = uint2(1, 1);

    // Stage 1 - loading shared memory with 4Y, 1U, and 1V tiles
    /* For 1920x1080, we should load:
    /        0    1    2    3    4    5    6    7
    /   0    0    4    8   12 1920 1924 1928 1932
    /   1 3840 3844 ...
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

    dctY[localId.y + 0][localId.x + 0] = .0f;
    dctY[localId.y + 0][localId.x + 8] = .0f;
    dctY[localId.y + 8][localId.x + 0] = .0f;
    dctY[localId.y + 8][localId.x + 8] = .0f;

    dctU[localId.y][localId.x] = .0f;
    dctV[localId.y][localId.x] = .0f;

    float4 localDctY = .0f;
    float localDctU = .0f;
    float localDctV = .0f;

    for (int row = 0; row != 8; ++row) {
        const float rowCoeff = dctCoeffs[localId.y][row];
        for (int col = 0; col != 8; ++col) {
            const float coeff = rowCoeff * dctCoeffs[localId.x][col];
            localDctY[0] += y[row + 0][col + 0] * coeff;
            localDctY[1] += y[row + 0][col + 8] * coeff;
            localDctY[2] += y[row + 8][col + 0] * coeff;
            localDctY[3] += y[row + 8][col + 8] * coeff;
            localDctU += u[row][col] * coeff;
            localDctV += v[row][col] * coeff;
        }
    }

    const float localQuant    = params.quantTable   [localId.y][localId.x / 4][localId.x % 4];
    const float localQuantInv = params.quantTableInv[localId.y][localId.x / 4][localId.x % 4];
    localDctY[0] = QuantizeFloat(localDctY[0], localQuant, localQuantInv);
    localDctY[1] = QuantizeFloat(localDctY[1], localQuant, localQuantInv);
    localDctY[2] = QuantizeFloat(localDctY[2], localQuant, localQuantInv);
    localDctY[3] = QuantizeFloat(localDctY[3], localQuant, localQuantInv);
    localDctU = QuantizeFloat(localDctU, localQuant, localQuantInv);
    localDctV = QuantizeFloat(localDctV, localQuant, localQuantInv);

    dctY[localId.y + 0][localId.x + 0] = localDctY[0];
    dctY[localId.y + 0][localId.x + 8] = localDctY[1];
    dctY[localId.y + 8][localId.x + 0] = localDctY[2];
    dctY[localId.y + 8][localId.x + 8] = localDctY[3];

    dctU[localId.y][localId.x] = localDctU;
    dctV[localId.y][localId.x] = localDctV;

    GroupMemoryBarrierWithGroupSync();

    // Stage 3 - IDCT and write to texture
    float4 localY = .0f;
    float localU = .0f;
    float localV = .0f;

    for (int row = 0; row != 8; ++row) {
        const float rowCoeff = dctCoeffs[row][localId.y];
        for (int col = 0; col != 8; ++col) {
            const float coeff = rowCoeff * dctCoeffs[col][localId.x];
            localY[0] += dctY[row + 0][col + 0] * coeff;
            localY[1] += dctY[row + 0][col + 8] * coeff;
            localY[2] += dctY[row + 8][col + 0] * coeff;
            localY[3] += dctY[row + 8][col + 8] * coeff;
            localU += dctU[row][col] * coeff;
            localV += dctV[row][col] * coeff;
        }
    }
    
    u[localId.y][localId.x] = localU;
    v[localId.y][localId.x] = localV;

    GroupMemoryBarrierWithGroupSync();

    // Now, let each thread write to the texture.
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

    const float3 cy0x0 = float3(localY[0], u[(0 + localId.y) / 2][(0 + localId.x) / 2], v[(0 + localId.y) / 2][(0 + localId.x) / 2]);
    const float3 cy0x1 = float3(localY[1], u[(0 + localId.y) / 2][(8 + localId.x) / 2], v[(0 + localId.y) / 2][(8 + localId.x) / 2]);
    const float3 cy1x0 = float3(localY[2], u[(8 + localId.y) / 2][(0 + localId.x) / 2], v[(8 + localId.y) / 2][(0 + localId.x) / 2]);
    const float3 cy1x1 = float3(localY[3], u[(8 + localId.y) / 2][(8 + localId.x) / 2], v[(8 + localId.y) / 2][(8 + localId.x) / 2]);

    outputTexture[(16 * blockId.xy) + localId.xy + (8 * x0y0)] = float4(clamp(mul(yuvToRgb, cy0x0), zeros, ones), 1.0);
    outputTexture[(16 * blockId.xy) + localId.xy + (8 * x1y0)] = float4(clamp(mul(yuvToRgb, cy0x1), zeros, ones), 1.0);
    outputTexture[(16 * blockId.xy) + localId.xy + (8 * x0y1)] = float4(clamp(mul(yuvToRgb, cy1x0), zeros, ones), 1.0);
    outputTexture[(16 * blockId.xy) + localId.xy + (8 * x1y1)] = float4(clamp(mul(yuvToRgb, cy1x1), zeros, ones), 1.0);
}
