// Phase 15 H6.2 (#228): wave32 Q4_K GEMV — one wave = one row, LDS-red.
// Layout matches ggml block_q4_K (144 B). Dequant algebra matches
// shaders/gpugemv_q4k.hlsl (4×64 lo-then-hi). Stream X; no 256-wide tile.
//
// LDS transpose: write gs[pass*nload + lane], read gs[lane*9 + q].
// Compact pitch = nload*16. Idle lanes (lane >= nload) do not write gs;
// they still take both barriers and must not return before them.
//
// y[row] = sum_j W[row,j] * x[j]

cbuffer Params : register(b0) {
    uint n;
    uint k_dim;
    uint nb; // k_dim / 256
    uint pad0;
};

ByteAddressBuffer W : register(t0);
ByteAddressBuffer X : register(t1);
RWStructuredBuffer<float> Y : register(u0);

groupshared uint4 gs[288]; // 9 * 32; 288 × 16 B = 4608 B
groupshared float red[32];

void get_scale_min_k4(uint j, uint s[12], out uint d, out uint m) {
    if (j < 4) {
        d = s[j] & 63u;
        m = s[j + 4] & 63u;
    } else {
        d = (s[j + 4] & 0xFu) | ((s[j - 4] >> 6) << 4);
        m = (s[j + 4] >> 4) | ((s[j - 0] >> 6) << 4);
    }
}

[numthreads(32, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID, uint gix : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint row = gid.x; // == dtid.x with 32 threads
    const uint lane = gix;  // 0..31
    // All-in or all-out for the group: do not return after the first barrier.
    if (row >= n)
        return;

    float acc = 0.0;
    for (uint block_base = 0; block_base < nb; block_base += 32u) {
        const uint nload = min(32u, nb - block_base);
        const uint chunk_byte = (row * nb + block_base) * 144u;
        const uint pitch = nload * 16u;
        [unroll]
        for (uint pass = 0; pass < 9u; ++pass) {
            uint4 v = uint4(0, 0, 0, 0);
            if (lane < nload)
                v = W.Load4(chunk_byte + pass * pitch + lane * 16u);
            if (lane < nload)
                gs[pass * nload + lane] = v; // idle lanes do NOT write
        }
        GroupMemoryBarrierWithGroupSync();
        if (lane < nload) {
            uint4 b[9];
            [unroll]
            for (uint q = 0; q < 9u; ++q)
                b[q] = gs[lane * 9u + q]; // 9 consecutive uint4 = one block_q4_K
            const uint d_dmin = b[0].x;   // packed fp16 d (lo16) + dmin (hi16)
            const uint scales01 = b[0].y;
            const uint scales23 = b[0].z;
            const uint scales45 = b[0].w;
            uint s[12];
            s[0] = scales01 & 0xffu;
            s[1] = (scales01 >> 8) & 0xffu;
            s[2] = (scales01 >> 16) & 0xffu;
            s[3] = (scales01 >> 24) & 0xffu;
            s[4] = scales23 & 0xffu;
            s[5] = (scales23 >> 8) & 0xffu;
            s[6] = (scales23 >> 16) & 0xffu;
            s[7] = (scales23 >> 24) & 0xffu;
            s[8] = scales45 & 0xffu;
            s[9] = (scales45 >> 8) & 0xffu;
            s[10] = (scales45 >> 16) & 0xffu;
            s[11] = (scales45 >> 24) & 0xffu;

            const float d = f16tof32(d_dmin & 0xffffu);
            const float minv = f16tof32(d_dmin >> 16);
            const uint x_base_bytes = (block_base + lane) * 256u * 4u;
            uint q_off = 0;
            uint is = 0;
            for (uint grp = 0; grp < 4u; ++grp) {
                uint sc, m;
                get_scale_min_k4(is + 0, s, sc, m);
                const float d1 = d * (float)sc;
                const float m1 = minv * (float)m;
                get_scale_min_k4(is + 1, s, sc, m);
                const float d2 = d * (float)sc;
                const float m2 = minv * (float)m;

                // lo nibbles — homogeneous 32, float4 only inside this loop
                for (uint l = 0; l < 32u; l += 4u) {
                    const uint word_idx = q_off + (l >> 2);
                    const uint4 src = b[1u + (word_idx >> 2)];
                    const uint word = src[word_idx & 3u];
                    const float4 n4 =
                        float4((float)((word >> 0) & 0xFu), (float)((word >> 8) & 0xFu),
                               (float)((word >> 16) & 0xFu), (float)((word >> 24) & 0xFu));
                    const float4 x4 = asfloat(X.Load4(x_base_bytes + (grp * 64u + l) * 4u));
                    acc += dot(d1 * n4 - m1, x4);
                }
                // hi nibbles — same qs bytes, X[grp*64+32 : 64)
                for (uint l = 0; l < 32u; l += 4u) {
                    const uint word_idx = q_off + (l >> 2);
                    const uint4 src = b[1u + (word_idx >> 2)];
                    const uint word = src[word_idx & 3u];
                    const float4 n4 =
                        float4((float)((word >> 4) & 0xFu), (float)((word >> 12) & 0xFu),
                               (float)((word >> 20) & 0xFu), (float)((word >> 28) & 0xFu));
                    const float4 x4 = asfloat(X.Load4(x_base_bytes + (grp * 64u + 32u + l) * 4u));
                    acc += dot(d2 * n4 - m2, x4);
                }
                q_off += 8u;
                is += 2u;
            }
        }
    }

    red[lane] = acc;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride)
            red[lane] += red[lane + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0u)
        Y[row] = red[0];
}
