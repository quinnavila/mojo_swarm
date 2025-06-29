from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from layout import Layout, LayoutTensor
from math import ceildiv
from random import random_float64

alias N = 10000
alias R: Float32 = 50.0
alias ALPHA: Float32 = 0.1
alias TPB = 256

alias dtype = DType.float32
alias agents_layout = Layout.row_major(N, 4)

fn boids_kernel[layout: Layout](
    agents_next: LayoutTensor[mut=True, dtype, layout],
    agents: LayoutTensor[mut=False, dtype, layout],
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

    agents_next.store[1](i, 0, SIMD[dtype, 1](px_next))
    agents_next.store[1](i, 1, SIMD[dtype, 1](py_next))
    agents_next.store[1](i, 2, SIMD[dtype, 1](vx_next))
    agents_next.store[1](i, 3, SIMD[dtype, 1](vy_next))

def main():
    with DeviceContext() as ctx:
        print("Initializing GPU buffers...")
        agents_buffer = ctx.enqueue_create_buffer[dtype](agents_layout.size())
        agents_next_buffer = ctx.enqueue_create_buffer[dtype](agents_layout.size())

        with agents_buffer.map_to_host() as agents_host:
            var agents_tensor_host = LayoutTensor[dtype, agents_layout](agents_host)
            for i in range(N):
                agents_tensor_host[i, 0] = Float32(random_float64(0, 800))
                agents_tensor_host[i, 1] = Float32(random_float64(0, 600))
                agents_tensor_host[i, 2] = Float32(random_float64(-2, 2))
                agents_tensor_host[i, 3] = Float32(random_float64(-2, 2))

        var agents_tensor = LayoutTensor[dtype, agents_layout](agents_buffer)
        var agents_next_tensor = LayoutTensor[dtype, agents_layout](agents_next_buffer)

        var blocks_per_grid = ceildiv(N, TPB)

        print("Launching boids kernel on GPU...")
        ctx.enqueue_function[boids_kernel[agents_layout]](
            agents_next_tensor,
            agents_tensor,
            grid_dim=blocks_per_grid,
            block_dim=TPB,
        )

        ctx.synchronize()
        print("Kernel execution finished.")

        with agents_next_buffer.map_to_host() as result_host:
            var result_tensor = LayoutTensor[dtype, agents_layout](result_host)
            print("First 3 boids after one step:")
            for i in range(N):
                print(
                    "  Boid", i, ": pos=(", result_tensor[i, 0], ",", result_tensor[i, 1], 
                    ") vel=(", result_tensor[i, 2], ",", result_tensor[i, 3], ")"
                )