// ggml backend d3d12: Q6_K x q8_K matmul (decode and per-column prefill).
// block_q6_K = { uint8 ql[128]; uint8 qh[64]; int8 scales[16]; half d; } = 210 B.
// dequantize_row_q6_K: per 128-element half v (ql += 64v, qh += 32v, sc += 8v),
// for l in 0..31 with is = l/16:
//   y[l]    = d*sc[is+0] * ((ql[l]    & 0xF) | ((qh[l]>>0)&3)<<4) - 32)
//   y[l+32] = d*sc[is+2] * ((ql[l+32] & 0xF) | ((qh[l]>>2)&3)<<4) - 32)
//   y[l+64] = d*sc[is+4] * ((ql[l]    >> 4 ) | ((qh[l]>>4)&3)<<4) - 32)
//   y[l+96] = d*sc[is+6] * ((ql[l+32] >> 4 ) | ((qh[l]>>6)&3)<<4) - 32)
// Thread itid: v = itid/8, l0 = 4*(itid%8), l = l0..l0+3 (Vulkan mul_mat_vec_q6_k).
// Per super-block, like ggml_vec_dot_q6_K_q8_K: (d * d_y) * sum(sc * (q6 - 32) * q8);
// q6 - 32 is applied as dot(q6, q8) - 32 * sum(q8), exact in integers.

#include "ggml_d3d12_common.hlsli"

// Four 6-bit values (0..63) packed one per byte.
uint q6pack(uint ql, uint qh, uint ql_shift, uint qh_shift) {
    return ((ql >> ql_shift) & 0x0F0F0F0Fu) | (((qh >> qh_shift) & 0x03030303u) << 4);
}

int sbyte(uint word, uint i) {
    return asint(((word >> (8u * i)) & 0xffu) << 24) >> 24;
}

[numthreads(NUM_THREADS, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint row0 = gid.x * NUM_ROWS;
    const uint col = gid.y;
    const uint ix = tid >> 4;
    const uint itid = tid & 15u;
    const uint v = itid >> 3;
    const uint l0 = 4u * (itid & 7u);
    const uint is = l0 >> 4;
    const uint e = 128u * v + l0;

    float acc[NUM_ROWS];
    [unroll]
    for (uint r = 0; r < NUM_ROWS; ++r)
        acc[r] = 0.0;

    for (uint blk = ix; blk < nchunk; blk += IN_FLIGHT) {
        const uint xb = col * x_stride + blk * 292u;
        const float dy = asfloat(X.Load(xb));
        const uint x1 = X.Load(xb + 4u + e);
        const uint x2 = X.Load(xb + 4u + e + 32u);
        const uint x3 = X.Load(xb + 4u + e + 64u);
        const uint x4 = X.Load(xb + 4u + e + 96u);
        const int s1 = sum4(x1), s2 = sum4(x2), s3 = sum4(x3), s4 = sum4(x4);

        [unroll]
        for (uint r = 0; r < NUM_ROWS; ++r) {
            const uint row = min(row0 + r, n - 1u);
            const uint bb = row * w_row_bytes + blk * 210u;
            const uint ql_a = ld32(bb + 64u * v + l0);
            const uint ql_b = ld32(bb + 64u * v + 32u + l0);
            const uint qh = ld32(bb + 128u + 32u * v + l0);
            const uint sc0 = ld32(bb + 192u + 8u * v);      // sc[8v+0..3]
            const uint sc1 = ld32(bb + 192u + 8u * v + 4u); // sc[8v+4..7]
            const float d = f16tof32(ld16(bb + 208u));
            const int t = sbyte(sc0, is) * (dot4(q6pack(ql_a, qh, 0u, 0u), x1) - 32 * s1) +
                          sbyte(sc0, is + 2u) * (dot4(q6pack(ql_b, qh, 0u, 2u), x2) - 32 * s2) +
                          sbyte(sc1, is) * (dot4(q6pack(ql_a, qh, 4u, 4u), x3) - 32 * s3) +
                          sbyte(sc1, is + 2u) * (dot4(q6pack(ql_b, qh, 4u, 6u), x4) - 32 * s4);
            acc[r] += (d * dy) * (float)t;
        }
    }
    reduce_store(acc, tid, row0, col);
}
