// Phase 15 H6.3 (#228): multi-row Q4_K GEMV — X in registers, reused across rows.
// Layout matches ggml block_q4_K (144 B). Lane mapping mirrors
// gpugemv_rows_lane() in include/xllama/gpugemv.h (llama.cpp Vulkan
// mul_mat_vec_q4_k shape):
//
//   tid 0..63: ix = tid / 16 picks one of 4 blocks in flight (stride 4 on nb),
//   itid = tid % 16 → il = itid / 4 (64-element group), ir = itid % 4 (8 qs bytes).
//   Each thread covers 8 lo nibbles (sub-block 2*il) and 8 hi nibbles (2*il+1).
//
// No LDS transpose and no barriers in the main loop; one LDS tree per row at
// the end. Rows past n clamp their loads and skip the store (no early return).
//
// y[row] = sum_j W[row,j] * x[j]

#ifndef NUM_ROWS
#define NUM_ROWS 4
#endif

cbuffer Params : register(b0) {
    uint n;
    uint k_dim;
    uint nb; // k_dim / 256
    uint pad0;
};

ByteAddressBuffer W : register(t0);
ByteAddressBuffer X : register(t1);
RWStructuredBuffer<float> Y : register(u0);

groupshared float red[NUM_ROWS][64];

uint scale_byte(uint i, uint4 hdr) {
    uint w = (i < 4u) ? hdr.y : ((i < 8u) ? hdr.z : hdr.w);
    return (w >> ((i & 3u) * 8u)) & 0xffu;
}

void get_scale_min_k4(uint j, uint4 hdr, out uint d, out uint m) {
    if (j < 4u) {
        d = scale_byte(j, hdr) & 63u;
        m = scale_byte(j + 4u, hdr) & 63u;
    } else {
        uint a = scale_byte(j + 4u, hdr);
        uint b = scale_byte(j - 4u, hdr);
        uint c = scale_byte(j, hdr);
        d = (a & 0xFu) | ((b >> 6) << 4);
        m = (a >> 4) | ((c >> 6) << 4);
    }
}

float4 nib4(uint word, uint shift) {
    return float4((float)((word >> (shift + 0u)) & 0xFu), (float)((word >> (shift + 8u)) & 0xFu),
                  (float)((word >> (shift + 16u)) & 0xFu), (float)((word >> (shift + 24u)) & 0xFu));
}

[numthreads(64, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint row0 = gid.x * NUM_ROWS;
    const uint ix = tid >> 4;
    const uint itid = tid & 15u;
    const uint il = itid >> 2;
    const uint ir = itid & 3u;
    const uint qs_byte = 16u + il * 32u + ir * 8u; // offset inside the 144 B block
    const uint e_lo = il * 64u + ir * 8u;          // first lo element in the block
    const uint e_hi = e_lo + 32u;                  // first hi element in the block

    float acc[NUM_ROWS];
    [unroll]
    for (uint r = 0; r < NUM_ROWS; ++r)
        acc[r] = 0.0;

    for (uint blk = ix; blk < nb; blk += 4u) {
        const uint xb = (blk * 256u) * 4u;
        const float4 xl0 = asfloat(X.Load4(xb + (e_lo + 0u) * 4u));
        const float4 xl1 = asfloat(X.Load4(xb + (e_lo + 4u) * 4u));
        const float4 xh0 = asfloat(X.Load4(xb + (e_hi + 0u) * 4u));
        const float4 xh1 = asfloat(X.Load4(xb + (e_hi + 4u) * 4u));
        const float sxl = dot(xl0 + xl1, float4(1, 1, 1, 1));
        const float sxh = dot(xh0 + xh1, float4(1, 1, 1, 1));

        [unroll]
        for (uint r = 0; r < NUM_ROWS; ++r) {
            const uint row = min(row0 + r, n - 1u);
            const uint bb = (row * nb + blk) * 144u;
            const uint4 hdr = W.Load4(bb);
            const uint2 q = W.Load2(bb + qs_byte);

            const float d = f16tof32(hdr.x & 0xffffu);
            const float minv = f16tof32(hdr.x >> 16);
            uint sc, m;
            get_scale_min_k4(2u * il, hdr, sc, m);
            const float d1 = d * (float)sc;
            const float m1 = minv * (float)m;
            get_scale_min_k4(2u * il + 1u, hdr, sc, m);
            const float d2 = d * (float)sc;
            const float m2 = minv * (float)m;

            const float lo = dot(nib4(q.x, 0u), xl0) + dot(nib4(q.y, 0u), xl1);
            const float hi = dot(nib4(q.x, 4u), xh0) + dot(nib4(q.y, 4u), xh1);
            acc[r] += d1 * lo - m1 * sxl + d2 * hi - m2 * sxh;
        }
    }

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
        Y[row0 + tid] = red[tid][0];
}
