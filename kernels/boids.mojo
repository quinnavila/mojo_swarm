from time import perf_counter_ns
from gpu.host import DeviceContext, DeviceBuffer
from gpu.id import block_dim, block_idx, thread_idx
from layout import Layout, LayoutTensor
from math import ceildiv
from random import random_float64

alias N = 100000
alias STEPS = 200
alias R: Float32 = 50.0
alias ALPHA: Float32 = 0.1
alias TPB = 256

alias WORLD_WIDTH: Float32 = 800.0
alias WORLD_HEIGHT: Float32 = 600.0

alias dtype = DType.float32
alias agents_layout = Layout.row_major(N, 4)

fn boids_kernel[layout: Layout](
    agents_next: LayoutTensor[mut=True, dtype, layout],
    agents: LayoutTensor[mut=False, dtype, layout],
    world_w: Float32,
    world_h: Float32,
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

    var px_next = (px + vx_next + world_w) % world_w
    var py_next = (py + vy_next + world_h) % world_h

    agents_next[i, 0] = px_next
    agents_next[i, 1] = py_next
    agents_next[i, 2] = vx_next
    agents_next[i, 3] = vy_next

def main():
    with DeviceContext() as ctx:
        print("Initializing", N, "boids...")
        
        var agents_buf_A = ctx.enqueue_create_buffer[dtype](agents_layout.size())
        var agents_buf_B = ctx.enqueue_create_buffer[dtype](agents_layout.size())

        with agents_buf_A.map_to_host() as agents_host:
            var agents_tensor_host = LayoutTensor[dtype, agents_layout](agents_host.unsafe_ptr())
            for i in range(N):
                agents_tensor_host[i, 0] = Float32(random_float64(0, Float64(WORLD_WIDTH)))
                agents_tensor_host[i, 1] = Float32(random_float64(0, Float64(WORLD_HEIGHT)))
                agents_tensor_host[i, 2] = Float32(random_float64(-2, 2))
                agents_tensor_host[i, 3] = Float32(random_float64(-2, 2))

        var current_buffer = agents_buf_A
        var next_buffer = agents_buf_B

        var blocks_per_grid = ceildiv(N, TPB)
        
        print("Starting simulation for", STEPS, "steps...")
        var total_kernel_time_ns: Int = 0
        
        var sim_start_ns = perf_counter_ns()

        for _ in range(STEPS):
            var step_start_ns = perf_counter_ns()

            var current_agents = LayoutTensor[dtype, agents_layout](current_buffer)
            var next_agents = LayoutTensor[dtype, agents_layout](next_buffer)

            ctx.enqueue_function[boids_kernel[agents_layout]](
                next_agents,
                current_agents,
                WORLD_WIDTH,
                WORLD_HEIGHT,
                grid_dim=blocks_per_grid,
                block_dim=TPB,
            )
            ctx.synchronize()
            
            var step_end_ns = perf_counter_ns()
            total_kernel_time_ns += (step_end_ns - step_start_ns)

            var temp_buf = current_buffer
            current_buffer = next_buffer
            next_buffer = temp_buf

        var sim_end_ns = perf_counter_ns()
        var total_duration_s = (sim_end_ns - sim_start_ns) / 1_000_000_000
        var avg_kernel_ms = (total_kernel_time_ns / STEPS) / 1_000_000

        print("\nSimulation finished.")
        print("Total steps:", STEPS)
        print("Total simulation time:", total_duration_s, "seconds")
        print("Average time per kernel step:", avg_kernel_ms, "ms")

        print("\nFinal state of first 3 boids:")
        with current_buffer.map_to_host() as result_host:
            var result_tensor = LayoutTensor[dtype, agents_layout](result_host.unsafe_ptr())
            for i in range(3):
                print(
                    "  Boid", i, ": pos=(", result_tensor[i, 0][0], ",", result_tensor[i, 1][0], 
                    ") vel=(", result_tensor[i, 2][0], ",", result_tensor[i, 3][0], ")"
                )