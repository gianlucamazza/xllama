// ggml backend d3d12: Q4_K x f32 matmul (decode and per-column prefill).
// Algebra and lane mapping of shaders/gpugemv_q4k_rows.hlsl (H6.3, 143 GB/s),
// plus tensor strides and the column index. block_q4_K = 144 B, 4-byte aligned.

#include "ggml_d3d12_common.hlsli"

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
    const uint col = gid.y;
    const uint ix = tid >> 4;
    const uint itid = tid & 15u;
    const uint il = itid >> 2;
    const uint ir = itid & 3u;
    const uint qs_byte = 16u + il * 32u + ir * 8u;
    const uint e_lo = il * 64u + ir * 8u;
    const uint e_hi = e_lo + 32u;

    float acc[NUM_ROWS];
    [unroll]
    for (uint r = 0; r < NUM_ROWS; ++r)
        acc[r] = 0.0;

    for (uint blk = ix; blk < nchunk; blk += 4u) {
        const uint xe = blk * 256u;
        const float4 xl0 = xload(col, xe + e_lo);
        const float4 xl1 = xload(col, xe + e_lo + 4u);
        const float4 xh0 = xload(col, xe + e_hi);
        const float4 xh1 = xload(col, xe + e_hi + 4u);
        const float sxl = dot(xl0 + xl1, float4(1, 1, 1, 1));
        const float sxh = dot(xh0 + xh1, float4(1, 1, 1, 1));

        [unroll]
        for (uint r = 0; r < NUM_ROWS; ++r) {
            const uint row = min(row0 + r, n - 1u);
            const uint bb = row * w_row_bytes + blk * 144u;
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
    reduce_store(acc, tid, row0, col);
}
