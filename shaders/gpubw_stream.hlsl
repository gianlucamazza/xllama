// Phase 15 W3 (#211): STREAM-style GPU read + parallel XOR reduction.
// Each thread loads one uint32 pattern word; groups reduce with groupshared XOR;
// per-group partials land in Output. CPU xors partials for the final checksum.
cbuffer Params : register(b0) {
    uint n_words;
    uint pad0;
    uint pad1;
    uint pad2;
};

StructuredBuffer<uint> Input : register(t0);
RWStructuredBuffer<uint> Output : register(u0);

groupshared uint gs[256];

[numthreads(256, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID, uint gix : SV_GroupIndex, uint3 gid : SV_GroupID) {
    uint i = dtid.x;
    uint v = (i < n_words) ? Input[i] : 0u;
    gs[gix] = v;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (gix < stride) {
            gs[gix] ^= gs[gix + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (gix == 0u) {
        Output[gid.x] = gs[0];
    }
}
