// ggml backend d3d12 (docs/gguf-gpu-decode.md) — shared by ggml_d3d12_mmv_*.hlsl.
// Root signature (src/bridge/ggml_d3d12.cpp): 8 root constants at b0, root SRV
// t0 (weights, DEFAULT heap), root UAVs u0 (output) and u1 (activations). X and
// Y both live in the CPU-visible D3D12_Host buffer, which stays in
// UNORDERED_ACCESS, so X is read through a UAV too. Every root descriptor
// points at its tensor, so offsets below are tensor-relative.
//
// Thread layout (as the H6.3 `rows` kernel): 64 threads, NUM_ROWS rows per
// group; ix = tid/16 picks one of 4 chunks in flight (stride 4 over the
// 256-element chunks of K), itid = tid%16 owns 16 weights of that chunk.
// SV_GroupID.y is the activation column (prefill), y = 0 for decode.

#define NUM_ROWS 4

cbuffer Params : register(b0) {
    uint n;           // output rows (ne01)
    uint k_dim;       // input width (ne00), multiple of 256
    uint nchunk;      // k_dim / 256
    uint w_row_bytes; // bytes per weight row (nb01)
    uint x_stride;    // floats between activation columns (nb11 / 4)
    uint y_stride;    // floats between output columns (nb1 / 4)
    uint pad0;
    uint pad1;
};

ByteAddressBuffer W : register(t0);
RWByteAddressBuffer X : register(u1);
RWStructuredBuffer<float> Y : register(u0);

groupshared float red[NUM_ROWS][64];

// Q4_0 (18 B) and Q6_K (210 B) blocks sit on 2-byte boundaries. ByteAddressBuffer
// loads need 4-byte alignment, so read the aligned dwords and shift.
uint ld32(uint a) {
    const uint b = a & ~3u;
    const uint lo = W.Load(b);
    if ((a & 3u) == 0u)
        return lo;
    return (lo >> 16) | (W.Load(b + 4u) << 16);
}

uint ld16(uint a) {
    const uint v = W.Load(a & ~3u);
    return (a & 2u) != 0u ? (v >> 16) : (v & 0xffffu);
}

float4 xload(uint col, uint elem) {
    return asfloat(X.Load4((col * x_stride + elem) * 4u));
}

// Sum each row's 64 partials and store; rows past n are skipped, never early-exit.
void reduce_store(float acc[NUM_ROWS], uint tid, uint row0, uint col) {
    [unroll]
    for (uint r = 0; r < NUM_ROWS; ++r)
        red[r][tid] = acc[r];
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint stride = 32u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            [unroll]
            for (uint r = 0; r < NUM_ROWS; ++r)
                red[r][tid] += red[r][tid + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid < NUM_ROWS && row0 + tid < n)
        Y[col * y_stride + row0 + tid] = red[tid][0];
}
