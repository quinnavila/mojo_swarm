import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from memory import UnsafePointer
from math import ceildiv

from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor


fn boids_kernel[agents_layout: Layout](
    agents_next: LayoutTensor[mut=True, DType.float32, agents_layout],
    agents: LayoutTensor[mut=False, DType.float32, agents_layout],
    N: Int,
    R: Float32,
    ALPHA: Float32,
):

    var i = block_dim.x * block_idx.x + thread_idx.x

    if i >= N:
        return

    var px = agents[i, 0][0]
    var py = agents[i, 1][0]
    var vx = agents[i, 2][0]
    var vy = agents[i, 3][0]

    var avg_vx: Float32 = 0.0
    var avg_vy: Float32 = 0.0
    var neighbor_count: Int = 0

    for j in range(N):
        if i == j:
            continue

        var dx = px - agents[j, 0][0]
        var dy = py - agents[j, 1][0]
        var dist_sq = dx * dx + dy * dy

        if dist_sq < R * R:
            neighbor_count += 1
            avg_vx += agents[j, 2][0]
            avg_vy += agents[j, 3][0]
    
    var vx_next: Float32
    var vy_next: Float32

    if neighbor_count > 0:
        var count_f32 = Float32(neighbor_count)
        avg_vx /= count_f32
        avg_vy /= count_f32
        
        vx_next = vx + ALPHA * (avg_vx - vx)
        vy_next = vy + ALPHA * (avg_vy - vy)
    else:
        vx_next = vx
        vy_next = vy

    var px_next = px + vx_next
    var py_next = py + vy_next

    agents_next.store[1](i, 0, SIMD[DType.float32, 1](px_next))
    agents_next.store[1](i, 1, SIMD[DType.float32, 1](py_next))
    agents_next.store[1](i, 2, SIMD[DType.float32, 1](vx_next))
    agents_next.store[1](i, 3, SIMD[DType.float32, 1](vy_next))

@compiler.register("boids")
struct BoidsCustomOp:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor[rank=2],
        input: InputTensor[rank=2],
        N: Int,
        R: Float32,
        ALPHA: Float32,
        ctx: DeviceContextPtr,
    ) raises:
        
        var agents_next = output.to_layout_tensor()
        var agents = input.to_layout_tensor()
        alias agents_layout = agents.layout

        @parameter
        if target == "gpu":
            var gpu_ctx = ctx.get_device_context()
            
            var threads_per_block: Int = 256
            var blocks_per_grid = ceildiv(N, threads_per_block)

            # The compile-time parameter list now ONLY contains the layout.
            gpu_ctx.enqueue_function[
                boids_kernel[agents_layout]
            ](
                # The runtime argument list contains the tensors AND the scalars.
                agents_next,
                agents,
                N,
                R,
                ALPHA,
                grid_dim=blocks_per_grid,
                block_dim=threads_per_block,
            )
        else:
            raise Error("Unsupported target: " + target)