struct VertexOut {
    float4 pos : SV_POSITION;
    float2 tc : TEXCOORD0;
};

VertexOut VSMain(uint vid : SV_VERTEXID) {
    const float4x4 verts = float4x4(
        -1, -1, 0, 1,
        -1, 1, 0, 1,
        1, -1, 0, 1,
        1, 1, 0, 1
    );
    const float4x2 texCoords = float4x2(
        0, 0,
        0, 1,
        1, 0,
        1, 1
    );

    VertexOut vOut;
    vOut.pos = verts[vid];
    vOut.tc = texCoords[vid];
    return vOut;
}