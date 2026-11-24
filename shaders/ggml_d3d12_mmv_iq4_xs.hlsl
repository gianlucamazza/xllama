// ggml backend d3d12: IQ4_XS x q8_K matmul (decode and per-column prefill).
// block_iq4_xs = 136 B: half d, uint16 scales_h, uint8 scales_l[4], qs[128].
// 8 groups of 32; each qs byte holds two 4-bit kvalues_iq4nl indices.
// Per group: (d * d_y) * (ls - 32) * sum(kvalues[nibble] * q8), as
// ggml_vec_dot_iq4_xs_q8_K.
// kvalues live in groupshared: 16-entry codebook lookups are compute, not
// bandwidth, and a CB/imm array gather is slower than LDS on this path.

#include "ggml_d3d12_common.hlsli"

static const int kIq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

groupshared int gIq4nl[16];

uint pack4nl(uint qs, uint shift) {
    const int a = gIq4nl[(qs >> (0u + shift)) & 0xfu];
    const int b = gIq4nl[(qs >> (8u + shift)) & 0xfu];
    const int c = gIq4nl[(qs >> (16u + shift)) & 0xfu];
    const int d = gIq4nl[(qs >> (24u + shift)) & 0xfu];
    return (a & 0xff) | ((b & 0xff) << 8) | ((c & 0xff) << 16) | ((d & 0xff) << 24);
}

[numthreads(NUM_THREADS, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    if (tid < 16u)
        gIq4nl[tid] = kIq4nl[tid];
    GroupMemoryBarrierWithGroupSync();

    const uint row0 = gid.x * NUM_ROWS;
    const uint col = gid.y;
    const uint ix = tid >> 4;
    const uint itid = tid & 15u;
    const uint g = itid >> 1;
    const uint half = itid & 1u;
    const uint shift = 4u * half;
    const uint e = g * 32u + half * 16u;

    float acc[NUM_ROWS];
    [unroll]
    for (uint r = 0; r < NUM_ROWS; ++r)
        acc[r] = 0.0;

    for (uint blk = ix; blk < nchunk; blk += IN_FLIGHT) {
        const uint xb = col * x_stride + blk * 292u;
        const float dy = asfloat(X.Load(xb));
        const uint4 xq = X.Load4(xb + 4u + e);

        [unroll]
        for (uint r = 0; r < NUM_ROWS; ++r) {
            const uint row = min(row0 + r, n - 1u);
            const uint bb = row * w_row_bytes + blk * 136u;
            const uint2 hdr = W.Load2(bb);
            const uint4 qs = W.Load4(bb + 8u + g * 16u);
            const float d = f16tof32(hdr.x & 0xffffu);
            const uint scales_h = hdr.x >> 16;
            const uint sl_byte = (hdr.y >> (8u * (g >> 1))) & 0xffu;
            const int ls = (int)((sl_byte >> (4u * (g & 1u))) & 0xfu) |
                           (int)(((scales_h >> (2u * g)) & 3u) << 4);
            const int sumi = dot4(pack4nl(qs.x, shift), xq.x) + dot4(pack4nl(qs.y, shift), xq.y) +
                             dot4(pack4nl(qs.z, shift), xq.z) + dot4(pack4nl(qs.w, shift), xq.w);
            acc[r] += (d * dy) * (float)((ls - 32) * sumi);
        }
    }
    reduce_store(acc, tid, row0, col);
}
