// ggml backend d3d12: GET_ROWS from a Q6_K weight (a tied token_embd that also
// serves as the lm_head lives once, in D3D12_Weights — #309). One group per
// (256-element chunk, id); thread tid = 32v + l writes elements
// 128v + l + {0, 32, 64, 96} exactly as dequantize_row_q6_K's inner loop:
//   y = d * sc[is + 2t] * (q6 - 32), is = l / 16 (float ops in ggml's order).
// The ids (int32) sit at the activation slot u1; x_stride is unused.

#include "ggml_d3d12_common.hlsli"

uint ld8(uint a) {
    return (W.Load(a & ~3u) >> ((a & 3u) * 8u)) & 0xffu;
}

float scale8(uint a) {
    return (float)(asint(ld8(a) << 24) >> 24);
}

[numthreads(64, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint v = tid >> 5;
    const uint l = tid & 31u;
    const uint is = l >> 4;
    const uint row = X.Load(gid.y * 4u);
    const uint bb = row * w_row_bytes + gid.x * 210u;
    const float d = f16tof32(ld16(bb + 208u));
    const uint qla = ld8(bb + 64u * v + l);
    const uint qlb = ld8(bb + 64u * v + l + 32u);
    const uint qh = ld8(bb + 128u + 32u * v + l);
    const uint sc = bb + 192u + 8u * v + is;
    const int q1 = (int)((qla & 0xFu) | (((qh >> 0) & 3u) << 4)) - 32;
    const int q2 = (int)((qlb & 0xFu) | (((qh >> 2) & 3u) << 4)) - 32;
    const int q3 = (int)((qla >> 4) | (((qh >> 4) & 3u) << 4)) - 32;
    const int q4 = (int)((qlb >> 4) | (((qh >> 6) & 3u) << 4)) - 32;
    const uint o = gid.y * y_stride + gid.x * 256u + 128u * v + l;
    Y[o] = d * scale8(sc) * (float)q1;
    Y[o + 32u] = d * scale8(sc + 2u) * (float)q2;
    Y[o + 64u] = d * scale8(sc + 4u) * (float)q3;
    Y[o + 96u] = d * scale8(sc + 6u) * (float)q4;
}
