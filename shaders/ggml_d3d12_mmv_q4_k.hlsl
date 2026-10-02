// ggml backend d3d12: Q4_K x q8_K matmul (decode and per-column prefill).
// Lane mapping of shaders/gpugemv_q4k_rows.hlsl (H6.3) plus tensor strides and
// the column index. block_q4_K = 144 B; block_q8_K = { float d; int8 qs[256];
// int16 bsums[16]; } = 292 B, both 4-byte aligned. Per super-block, like
// ggml_vec_dot_q4_K_q8_K: (d * d_y) * sum(sc * q4 * q8) - (dmin * d_y) * sum(m * q8).

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

[numthreads(NUM_THREADS, 1, 1)]
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

    for (uint blk = ix; blk < nchunk; blk += IN_FLIGHT) {
        const uint xb = col * x_stride + blk * 292u;
        const float dy = asfloat(X.Load(xb));
        const uint2 xl = X.Load2(xb + 4u + e_lo);
        const uint2 xh = X.Load2(xb + 4u + e_hi);
        const int sl = sum4(xl.x) + sum4(xl.y);
        const int sh = sum4(xh.x) + sum4(xh.y);

        [unroll]
        for (uint r = 0; r < NUM_ROWS; ++r) {
            const uint row = min(row0 + r, n - 1u);
            const uint bb = row * w_row_bytes + blk * 144u;
            const uint4 hdr = W.Load4(bb);
            const uint2 q = W.Load2(bb + qs_byte);

            const float d = f16tof32(hdr.x & 0xffffu);
            const float dmin = f16tof32(hdr.x >> 16);
            uint sc1, m1, sc2, m2;
            get_scale_min_k4(2u * il, hdr, sc1, m1);
            get_scale_min_k4(2u * il + 1u, hdr, sc2, m2);

            const int lo = dot4(q.x & 0x0F0F0F0Fu, xl.x) + dot4(q.y & 0x0F0F0F0Fu, xl.y);
            const int hi = dot4((q.x >> 4) & 0x0F0F0F0Fu, xh.x) + dot4((q.y >> 4) & 0x0F0F0F0Fu, xh.y);
            acc[r] += (d * dy) * (float)((int)sc1 * lo + (int)sc2 * hi) -
                      (dmin * dy) * (float)((int)m1 * sl + (int)m2 * sh);
        }
    }
    reduce_store(acc, tid, row0, col);
}
