// Phase 15 H6.1 (#228): Q4_K GEMV — dequant in register, no fp16 materialize.
// Layout matches ggml block_q4_K (144 B): half d, half dmin, scales[12], qs[128].
// One thread per output row y[n]; loops over k/256 super-blocks.
//
// y[row] = sum_j W[row,j] * x[j]

cbuffer Params : register(b0) {
    uint n;
    uint k_dim;
    uint nb; // k_dim / 256
    uint pad0;
};

struct Q4KBlock {
    uint d_dmin;      // two fp16 packed: low16=d, high16=dmin
    uint scales01;    // scales[0..3]
    uint scales23;    // scales[4..7]
    uint scales45;    // scales[8..11]
    uint qs[32];      // 128 bytes = 32 uints
};

StructuredBuffer<Q4KBlock> Weights : register(t0);
StructuredBuffer<float> X : register(t1);
RWStructuredBuffer<float> Y : register(u0);

float half_to_float(uint h) {
    uint sign = (h & 0x8000u) << 16;
    uint exp = (h >> 10) & 0x1fu;
    uint mant = h & 0x3ffu;
    uint f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            exp = 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3ffu;
            f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7f800000u | (mant << 13);
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    return asfloat(f);
}

void get_scale_min_k4(uint j, uint scales01, uint scales23, uint scales45, out uint d, out uint m) {
    // Reconstruct byte array scales[12] from packed words.
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

    if (j < 4) {
        d = s[j] & 63u;
        m = s[j + 4] & 63u;
    } else {
        d = (s[j + 4] & 0xFu) | ((s[j - 4] >> 6) << 4);
        m = (s[j + 4] >> 4) | ((s[j - 0] >> 6) << 4);
    }
}

[numthreads(64, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    uint row = dtid.x;
    if (row >= n)
        return;

    float acc = 0.0;
    for (uint bi = 0; bi < nb; ++bi) {
        Q4KBlock b = Weights[row * nb + bi];
        float d = half_to_float(b.d_dmin & 0xffffu);
        float minv = half_to_float(b.d_dmin >> 16);
        uint base = bi * 256u;
        uint is = 0;
        // 4 groups of 64 elements; qs stored as 32 uints = 128 bytes
        // qs layout: for each j in 0,64,128,192: 32 bytes (8 uints) cover 32 lo + 32 hi nibbles
        uint q_off = 0;
        for (uint j = 0; j < 256u; j += 64u) {
            uint sc, m;
            get_scale_min_k4(is + 0, b.scales01, b.scales23, b.scales45, sc, m);
            float d1 = d * (float)sc;
            float m1 = minv * (float)m;
            get_scale_min_k4(is + 1, b.scales01, b.scales23, b.scales45, sc, m);
            float d2 = d * (float)sc;
            float m2 = minv * (float)m;

            // 32 bytes of qs → 8 uints
            for (uint l = 0; l < 32u; ++l) {
                uint word = b.qs[q_off + (l >> 2)];
                uint byte = (word >> ((l & 3u) * 8u)) & 0xffu;
                float w0 = d1 * (float)(byte & 0xFu) - m1;
                acc += w0 * X[base + j + l];
            }
            for (uint l = 0; l < 32u; ++l) {
                uint word = b.qs[q_off + (l >> 2)];
                uint byte = (word >> ((l & 3u) * 8u)) & 0xffu;
                float w1 = d2 * (float)(byte >> 4) - m2;
                acc += w1 * X[base + j + 32u + l];
            }
            q_off += 8u; // 32 bytes
            is += 2u;
        }
    }
    Y[row] = acc;
}
