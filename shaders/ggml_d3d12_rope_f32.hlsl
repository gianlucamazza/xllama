// ggml backend d3d12: ROPE on an F32 activation [ne0, rows, tokens], modes NORMAL
// (pairs 2p, 2p+1) and NEOX (pairs p, p + n_dims/2). The cos/sin table is built on
// the CPU with the ggml-cpu formulas (one float2 per rotated pair per token), so
// the shader only multiplies: no GPU sin/cos/pow. Elements at and past n_dims are
// copied. Each thread owns one pair and reads both inputs before writing, so
// Y may alias X (an in-place allocation).
// Products and the sum are `precise`: no fused multiply-add, the same two
// roundings per product as ggml-cpu compiled without FMA contraction.
// Root signature (src/bridge/ggml_d3d12.cpp): b0 constants, u0 output Y,
// u1 input X, u2 the table. Dispatch (ceil(max(half, tail) / NUM_THREADS), rows, tokens).

#define NUM_THREADS 128 // kD3d12RopeThreads

cbuffer Params : register(b0) {
    uint ne0;        // elements per row
    uint n_dims;     // rotated elements (even)
    uint mode;       // 0 NORMAL, 2 NEOX
    uint x_row;      // floats between rows (heads) of X
    uint x_tok;      // floats between tokens of X
    uint y_row;      // floats between rows of Y
    uint y_tok;      // floats between tokens of Y
    uint pad0;
};

RWStructuredBuffer<float> Y : register(u0);
RWByteAddressBuffer X : register(u1);
RWByteAddressBuffer T : register(u2); // [token][n_dims] = (cos, sin) per pair

[numthreads(NUM_THREADS, 1, 1)]
void CSMain(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    const uint j = gid.x * NUM_THREADS + tid;
    const uint row = gid.y;
    const uint tok = gid.z;
    const uint xb = tok * x_tok + row * x_row;
    const uint yb = tok * y_tok + row * y_row;
    const uint half_dims = n_dims >> 1;
    if (j < half_dims) {
        const uint t = (tok * n_dims + 2u * j) * 4u;
        const float c = asfloat(T.Load(t));
        const float s = asfloat(T.Load(t + 4u));
        const uint i0 = mode == 0u ? 2u * j : j;
        const uint i1 = mode == 0u ? 2u * j + 1u : j + half_dims;
        const float x0 = asfloat(X.Load((xb + i0) * 4u));
        const float x1 = asfloat(X.Load((xb + i1) * 4u));
        precise float a = x0 * c;
        precise float b = x1 * s;
        precise float d = x0 * s;
        precise float e = x1 * c;
        precise float r0 = a - b;
        precise float r1 = d + e;
        Y[yb + i0] = r0;
        Y[yb + i1] = r1;
    }
    const uint i = n_dims + j;
    if (i < ne0)
        Y[yb + i] = asfloat(X.Load((xb + i) * 4u));
}
