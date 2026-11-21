// ggml backend d3d12: the F32 bias ADD after a projection,
//   Y[c * y_stride + i] = X[c * x_stride + i] + B[i],  i < n.
// One thread per element; SV_GroupID.y is the activation column. A single
// correctly rounded fp32 add (D3D11.3 spec §3.1.3.1), compiled with
// -denorm preserve, so the result equals ggml-cpu's bit for bit. Y may alias X
// (an in-place allocation): each thread reads its element before writing it.
// Root signature (src/bridge/ggml_d3d12.cpp): b0 constants, t0 the bias in
// D3D12_Weights, u0 the output, u1 the input activations.

#define NUM_THREADS 256 // kD3d12AddThreads

cbuffer Params : register(b0) {
    uint n;        // elements per column (ne0)
    uint x_stride; // floats between input columns (nb01 / 4)
    uint y_stride; // floats between output columns (nb1 / 4)
    uint pad0;
    uint pad1;
    uint pad2;
    uint pad3;
    uint pad4;
};

ByteAddressBuffer B : register(t0);
RWByteAddressBuffer X : register(u1);
RWStructuredBuffer<float> Y : register(u0);

[numthreads(NUM_THREADS, 1, 1)]
void CSMain(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    const uint i = gid.x * NUM_THREADS + tid;
    if (i >= n)
        return;
    const float x = asfloat(X.Load((gid.y * x_stride + i) * 4u));
    Y[gid.y * y_stride + i] = x + asfloat(B.Load(i * 4u));
}
