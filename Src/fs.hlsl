struct VertexOut {
    float4 pos : SV_POSITION;
    float2 tc : TEXCOORD0;
};

Texture2D image   : register(t0, space2);
SamplerState samp : register(s0, space2);

float4 FSMain(VertexOut vOut) : SV_Target0 {
    const float3 sampleTex = image.Sample(samp, vOut.tc).rgb;
    return float4(sampleTex, 1.0);
}