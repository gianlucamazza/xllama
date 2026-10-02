// ggml backend d3d12 (docs/gguf-gpu-decode.md) — shared by ggml_d3d12_mmv_*.hlsl.
// Root signature (src/bridge/ggml_d3d12.cpp): 8 root constants at b0, root SRV
// t0 (weights, DEFAULT heap), root UAVs u0 (output) and u1 (activations). X is
// not the f32 tensor: the backend quantizes it on the CPU with ggml's own
// from_float into the weight's vec_dot type (q8_0 for Q4_0, q8_K for Q4_K /
// Q6_K) — one row of ggml blocks per column, x_stride bytes apart — so the
// kernels sum integers exactly like the CPU's vec_dot (#312). X and Y live in
// CPU-visible buffers that stay in UNORDERED_ACCESS, so X is read through a UAV.
//
// Thread layout (the H6.3 `rows` kernel, widened): NUM_THREADS threads,
// NUM_ROWS rows per group; ix = tid/16 picks one of IN_FLIGHT chunks (stride
// IN_FLIGHT over the 256-element chunks of K), itid = tid%16 owns 16 weights
// of that chunk. NUM_THREADS is 64 or 128 (two blobs per type); the backend
// picks per matmul with d3d12_mm_threads(K): long K wants 8 chunks in flight
// (D2a run 1: ffn_down N=2048 under 100 GB/s at 64), short K starves 128
// (run 2: lm_head K=1024 fell from 110 to 64 GB/s).
// SV_GroupID.y is the activation column (prefill), y = 0 for decode.

#define NUM_ROWS 4
#ifndef NUM_THREADS
#define NUM_THREADS 64
#endif
#define IN_FLIGHT (NUM_THREADS / 16)

cbuffer Params : register(b0) {
    uint n;           // output rows (ne01)
    uint k_dim;       // input width (ne00), multiple of 256
    uint nchunk;      // k_dim / 256
    uint w_row_bytes; // bytes per weight row (nb01)
    uint x_stride;    // bytes between quantized activation columns
    uint y_stride;    // floats between output columns (nb1 / 4)
    uint pad0;
    uint pad1;
};

ByteAddressBuffer W : register(t0);
RWByteAddressBuffer X : register(u1);
RWStructuredBuffer<float> Y : register(u0);

groupshared float red[NUM_ROWS][NUM_THREADS];

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

// X reads: block_q8_0 is 34 B, so q8_0 columns sit on 2-byte boundaries too.
uint xld32(uint a) {
    const uint b = a & ~3u;
    const uint lo = X.Load(b);
    if ((a & 3u) == 0u)
        return lo;
    return (lo >> 16) | (X.Load(b + 4u) << 16);
}

uint xld16(uint a) {
    const uint v = X.Load(a & ~3u);
    return (a & 2u) != 0u ? (v >> 16) : (v & 0xffffu);
}

// Packed signed int8 dot products (SM 6.4). Weight lanes (nibbles 0..15, Q6_K
// values 0..63) are valid non-negative int8.
int dot4(uint a, uint b) {
    return dot4add_i8packed(a, b, 0);
}

int sum4(uint b) {
    return dot4add_i8packed(b, 0x01010101u, 0);
}

// Sum each row's 64 partials and store; rows past n are skipped, never early-exit.
void reduce_store(float acc[NUM_ROWS], uint tid, uint row0, uint col) {
    [unroll]
    for (uint r = 0; r < NUM_ROWS; ++r)
        red[r][tid] = acc[r];
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint stride = NUM_THREADS / 2u; stride > 0u; stride >>= 1u) {
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
