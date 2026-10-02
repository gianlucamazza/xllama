// ggml backend d3d12: Q4_0 x q8_0 matmul (decode and per-column prefill).
// block_q4_0 = { half d; uint8 qs[16]; } = 18 B, value = (nibble - 8) * d;
// low nibbles are elements 0..15, high nibbles 16..31 (dequantize_row_q4_0).
// block_q8_0 = { half d; int8 qs[32]; } = 34 B. One 256-element chunk is 8
// blocks; thread itid takes block b = itid/2 and half h = itid%2: qs[8h..8h+7]
// → elements 8h+i (lo) and 16+8h+i (hi). Per block, like
// ggml_vec_dot_q4_0_q8_0: (d_w * d_x) * sum((nibble - 8) * q8).

#include "ggml_d3d12_common.hlsli"

[numthreads(NUM_THREADS, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint row0 = gid.x * NUM_ROWS;
    const uint col = gid.y;
    const uint ix = tid >> 4;
    const uint itid = tid & 15u;
    const uint b = itid >> 1;
    const uint h = itid & 1u;

    float acc[NUM_ROWS];
    [unroll]
    for (uint r = 0; r < NUM_ROWS; ++r)
        acc[r] = 0.0;

    for (uint blk = ix; blk < nchunk; blk += IN_FLIGHT) {
        const uint xb = col * x_stride + (blk * 8u + b) * 34u;
        const float dx = f16tof32(xld16(xb));
        const uint xl0 = xld32(xb + 2u + 8u * h);
        const uint xl1 = xld32(xb + 6u + 8u * h);
        const uint xh0 = xld32(xb + 18u + 8u * h);
        const uint xh1 = xld32(xb + 22u + 8u * h);
        const int sq = sum4(xl0) + sum4(xl1) + sum4(xh0) + sum4(xh1);

        [unroll]
        for (uint r = 0; r < NUM_ROWS; ++r) {
            const uint row = min(row0 + r, n - 1u);
            const uint bb = row * w_row_bytes + (blk * 8u + b) * 18u;
            const float d = f16tof32(ld16(bb));
            const uint q0 = ld32(bb + 2u + 8u * h);
            const uint q1 = ld32(bb + 6u + 8u * h);
            const int lo = dot4(q0 & 0x0F0F0F0Fu, xl0) + dot4(q1 & 0x0F0F0F0Fu, xl1);
            const int hi = dot4((q0 >> 4) & 0x0F0F0F0Fu, xh0) + dot4((q1 >> 4) & 0x0F0F0F0Fu, xh1);
            acc[r] += (d * dx) * (float)(lo + hi - 8 * sq);
        }
    }
    reduce_store(acc, tid, row0, col);
}
