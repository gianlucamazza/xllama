// Phase 15 W3 (#211): STREAM-style GPU read + parallel XOR reduction.
// Flat word index works for multi-dimensional Dispatch (X/Y/Z) so 1 GiB
// (1 048 576 groups) stays within D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION.
//
// flat_i = dtid.x + dtid.y * (dispatch_x * threads_per_group)
//        + dtid.z * (dispatch_x * dispatch_y * threads_per_group)
// flat_group = gid.x + gid.y * dispatch_x + gid.z * dispatch_x * dispatch_y
cbuffer Params : register(b0) {
    uint n_words;
    uint dispatch_x;
    uint dispatch_y;
    uint threads_per_group; // 256
};

StructuredBuffer<uint> Input : register(t0);
RWStructuredBuffer<uint> Output : register(u0);

groupshared uint gs[256];

[numthreads(256, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID, uint gix : SV_GroupIndex, uint3 gid : SV_GroupID) {
    uint stride_y = dispatch_x * threads_per_group;
    uint stride_z = dispatch_x * dispatch_y * threads_per_group;
    uint i = dtid.x + dtid.y * stride_y + dtid.z * stride_z;
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
        uint flat_group = gid.x + gid.y * dispatch_x + gid.z * dispatch_x * dispatch_y;
        Output[flat_group] = gs[0];
    }
}
