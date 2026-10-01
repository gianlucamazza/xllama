// Phase 15 H6.3 (#228): multi-row Q4_K × q8 GEMV with packed int8 dot (SM 6.4).
// Same lane mapping as gpugemv_q4k_rows.hlsl. X is pre-quantized on the host
// (gpugemv_quantize_x_q8): int8 per element, one fp32 scale per 32 elements —
// the same activation-quant idea llama.cpp's CPU vec_dot_q4_K_q8_K uses.
//
// X raw buffer: bytes [0, k) = int8 qx; fp32 dx[k/32] at byte offset k.
// Nibbles are 0..15, so they are valid signed int8 lanes for dot4add_i8packed.
//
// y[row] = sum_s dx[s] * (d*sc_s * sum(q4*qx) - dmin*m_s * sum(qx))

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

[numthreads(64, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint row0 = gid.x * NUM_ROWS;
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

    for (uint blk = ix; blk < nb; blk += 4u) {
        const uint xb = blk * 256u;
        const uint2 xl = X.Load2(xb + e_lo);
        const uint2 xh = X.Load2(xb + e_hi);
        const float dxl = asfloat(X.Load(k_dim + (blk * 8u + 2u * il) * 4u));
        const float dxh = asfloat(X.Load(k_dim + (blk * 8u + 2u * il + 1u) * 4u));
        const float sxl = (float)dot4add_i8packed(xl.y, 0x01010101u, dot4add_i8packed(xl.x, 0x01010101u, 0));
        const float sxh = (float)dot4add_i8packed(xh.y, 0x01010101u, dot4add_i8packed(xh.x, 0x01010101u, 0));

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

            const int lo = dot4add_i8packed(q.y & 0x0F0F0F0Fu, xl.y,
                                            dot4add_i8packed(q.x & 0x0F0F0F0Fu, xl.x, 0));
            const int hi = dot4add_i8packed((q.y >> 4) & 0x0F0F0F0Fu, xh.y,
                                            dot4add_i8packed((q.x >> 4) & 0x0F0F0F0Fu, xh.x, 0));
            acc[r] += dxl * (d1 * (float)lo - m1 * sxl) + dxh * (d2 * (float)hi - m2 * sxh);
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
