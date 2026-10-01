// ggml backend d3d12: Q4_0 x f32 matmul (decode and per-column prefill).
// block_q4_0 = { half d; uint8 qs[16]; } = 18 B, value = (nibble - 8) * d;
// low nibbles are elements 0..15, high nibbles 16..31 (dequantize_row_q4_0).
// One 256-element chunk is 8 blocks; thread itid takes block b = itid/2 and
// half h = itid%2: qs[8h..8h+7] → elements 8h+i (lo) and 16+8h+i (hi).

#include "ggml_d3d12_common.hlsli"

float4 nib4(uint word, uint shift) {
    return float4((float)((word >> (shift + 0u)) & 0xFu), (float)((word >> (shift + 8u)) & 0xFu),
                  (float)((word >> (shift + 16u)) & 0xFu), (float)((word >> (shift + 24u)) & 0xFu));
}

[numthreads(NUM_THREADS, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint row0 = gid.x * NUM_ROWS;
    const uint col = gid.y;
    const uint ix = tid >> 4;
    const uint itid = tid & 15u;
    const uint b = itid >> 1;
    const uint h = itid & 1u;
    const uint e_lo = 32u * b + 8u * h;
    const uint e_hi = e_lo + 16u;

    float acc[NUM_ROWS];
    [unroll]
    for (uint r = 0; r < NUM_ROWS; ++r)
        acc[r] = 0.0;

    for (uint blk = ix; blk < nchunk; blk += IN_FLIGHT) {
        const uint xe = blk * 256u;
        const float4 xl0 = xload(col, xe + e_lo);
        const float4 xl1 = xload(col, xe + e_lo + 4u);
        const float4 xh0 = xload(col, xe + e_hi);
        const float4 xh1 = xload(col, xe + e_hi + 4u);
        const float sx = dot(xl0 + xl1 + xh0 + xh1, float4(1, 1, 1, 1));

        [unroll]
        for (uint r = 0; r < NUM_ROWS; ++r) {
            const uint row = min(row0 + r, n - 1u);
            const uint bb = row * w_row_bytes + (blk * 8u + b) * 18u;
            const float d = f16tof32(ld16(bb));
            const uint q0 = ld32(bb + 2u + 8u * h);
            const uint q1 = ld32(bb + 6u + 8u * h);
            const float dotq = dot(nib4(q0, 0u), xl0) + dot(nib4(q1, 0u), xl1) +
                               dot(nib4(q0, 4u), xh0) + dot(nib4(q1, 4u), xh1);
            acc[r] += d * (dotq - 8.0 * sx);
        }
    }
    reduce_store(acc, tid, row0, col);
}
