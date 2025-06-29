from time import perf_counter_ns
from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from gpu import barrier
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from math import ceildiv, floor, log2, sqrt
from random import random_float64
from os.atomic import Atomic

alias N = 100000
alias STEPS = 200
alias TPB = 256
alias WORLD_WIDTH: Float32 = 800.0
alias WORLD_HEIGHT: Float32 = 600.0
alias R: Float32 = 50.0
alias CELL_SIZE: Float32 = R

alias SEPARATION_STRENGTH: Float32 = 0.05
alias COHESION_STRENGTH: Float32 = 0.001
alias ALIGNMENT_STRENGTH: Float32 = 0.03
alias MAX_SPEED: Float32 = 4.0
alias MIN_SPEED: Float32 = 1.0

alias GRID_COLS = Int(floor(WORLD_WIDTH / CELL_SIZE))
alias GRID_ROWS = Int(floor(WORLD_HEIGHT / CELL_SIZE))
alias NUM_CELLS = GRID_COLS * GRID_ROWS
alias PADDED_NUM_CELLS = 256

alias dtype = DType.float32
alias agents_layout = Layout.row_major(N, 4)
alias hash_layout = Layout.row_major(N)
alias index_layout = Layout.row_major(N)
alias histogram_layout = Layout.row_major(NUM_CELLS)
alias cell_offsets_layout = Layout.row_major(NUM_CELLS)

fn spatial_hash_kernel[
    agents_layout: Layout,
    hash_layout: Layout,
    index_layout: Layout,
](
    particle_hashes: LayoutTensor[mut=True, DType.int32, hash_layout],
    particle_indices: LayoutTensor[mut=True, DType.int32, index_layout],
    agents: LayoutTensor[mut=False, dtype, agents_layout],
):
    var i = block_dim.x * block_idx.x + thread_idx.x
    if i >= N:
        return

    var px = agents[i, 0][0]
    var py = agents[i, 1][0]
    var cell_x = Int(floor(px / CELL_SIZE))
    var cell_y = Int(floor(py / CELL_SIZE))
    
    if cell_x < 0: cell_x = 0
    if cell_x >= GRID_COLS: cell_x = GRID_COLS - 1
    if cell_y < 0: cell_y = 0
    if cell_y >= GRID_ROWS: cell_y = GRID_ROWS - 1

    var hash = cell_y * GRID_COLS + cell_x
    
    particle_hashes[i] = hash
    particle_indices[i] = i

fn histogram_kernel[
    hash_layout: Layout,
    histogram_layout: Layout,
](
    particle_hashes: LayoutTensor[mut=False, DType.int32, hash_layout],
    cell_counts: LayoutTensor[mut=True, DType.int32, histogram_layout],
):
    var i = block_dim.x * block_idx.x + thread_idx.x
    if i >= N:
        return

    var hash = particle_hashes[i][0]
    _ = Atomic[DType.int32].fetch_add(cell_counts.ptr + hash, 1)

fn parallel_prefix_sum_kernel[
    histogram_layout: Layout,
    cell_offsets_layout: Layout,
](
    cell_counts: LayoutTensor[mut=False, DType.int32, histogram_layout],
    cell_offsets: LayoutTensor[mut=True, DType.int32, cell_offsets_layout],
):
    var shared_data = tb[DType.int32]().row_major[PADDED_NUM_CELLS]().shared().alloc()
    var i = thread_idx.x

    if i < NUM_CELLS:
        shared_data[i] = cell_counts[i]
    elif i < PADDED_NUM_CELLS:
        shared_data[i] = 0
    barrier()

    var log2_padded = Int(log2(Float64(PADDED_NUM_CELLS)))
    
    for d in range(log2_padded):
        var offset = 1 << d
        var stride = 1 << (d + 1)
        if (i + 1) % stride == 0:
            shared_data[i] += shared_data[i - offset]
        barrier()

    if i == PADDED_NUM_CELLS - 1:
        shared_data[i] = 0
    barrier()

    for d in range(log2_padded - 1, -1, -1):
        var offset = 1 << d
        var stride = 1 << (d + 1)
        if (i + 1) % stride == 0:
            var temp = shared_data[i - offset]
            shared_data[i - offset] = shared_data[i]
            shared_data[i] += temp
        barrier()

    if i < NUM_CELLS:
        cell_offsets[i] = shared_data[i]

fn reorder_kernel[
    agents_layout: Layout,
    hash_layout: Layout,
    index_layout: Layout,
    cell_offsets_layout: Layout,
](
    sorted_agents: LayoutTensor[mut=True, dtype, agents_layout],
    original_agents: LayoutTensor[mut=False, dtype, agents_layout],
    particle_hashes: LayoutTensor[mut=False, DType.int32, hash_layout],
    particle_indices: LayoutTensor[mut=False, DType.int32, index_layout],
    cell_offsets_counter: LayoutTensor[mut=True, DType.int32, cell_offsets_layout],
):
    var i = block_dim.x * block_idx.x + thread_idx.x
    if i >= N:
        return

    var original_idx = Int(particle_indices[i][0])
    var hash = Int(particle_hashes[i][0])
    
    var sorted_idx = Int(Atomic[DType.int32].fetch_add(cell_offsets_counter.ptr + hash, 1))

    sorted_agents[sorted_idx, 0] = original_agents[original_idx, 0]
    sorted_agents[sorted_idx, 1] = original_agents[original_idx, 1]
    sorted_agents[sorted_idx, 2] = original_agents[original_idx, 2]
    sorted_agents[sorted_idx, 3] = original_agents[original_idx, 3]

fn boids_update_optimized_kernel[
    agents_layout: Layout,
    histogram_layout: Layout,
    cell_offsets_layout: Layout,
](
    agents_next: LayoutTensor[mut=True, dtype, agents_layout],
    sorted_agents: LayoutTensor[mut=False, dtype, agents_layout],
    cell_counts: LayoutTensor[mut=False, DType.int32, histogram_layout],
    cell_offsets: LayoutTensor[mut=False, DType.int32, cell_offsets_layout],
):
    var i = block_dim.x * block_idx.x + thread_idx.x
    if i >= N:
        return

    var px = sorted_agents[i, 0][0]
    var py = sorted_agents[i, 1][0]
    var vx = sorted_agents[i, 2][0]
    var vy = sorted_agents[i, 3][0]

    var cell_x = Int(floor(px / CELL_SIZE))
    var cell_y = Int(floor(py / CELL_SIZE))

    var avg_px: Float32 = 0.0
    var avg_py: Float32 = 0.0
    var avg_vx: Float32 = 0.0
    var avg_vy: Float32 = 0.0
    var sep_vx: Float32 = 0.0
    var sep_vy: Float32 = 0.0
    var neighbor_count: Int = 0

    for dy in range(-1, 2):
        for dx in range(-1, 2):
            var neighbor_cell_x = cell_x + dx
            var neighbor_cell_y = cell_y + dy

            if neighbor_cell_x >= 0 and neighbor_cell_x < GRID_COLS and \
               neighbor_cell_y >= 0 and neighbor_cell_y < GRID_ROWS:
                
                var hash = neighbor_cell_y * GRID_COLS + neighbor_cell_x
                var cell_start = Int(cell_offsets[hash][0])
                var cell_end = cell_start + Int(cell_counts[hash][0])

                for j in range(cell_start, cell_end):
                    if i == j:
                        continue
                    
                    var neighbor_px = sorted_agents[j, 0][0]
                    var neighbor_py = sorted_agents[j, 1][0]
                    var d_px = px - neighbor_px
                    var d_py = py - neighbor_py
                    var dist_sq = d_px * d_px + d_py * d_py

                    if dist_sq < R * R:
                        neighbor_count += 1
                        avg_px += neighbor_px
                        avg_py += neighbor_py
                        avg_vx += sorted_agents[j, 2][0]
                        avg_vy += sorted_agents[j, 3][0]
                        sep_vx += d_px / (dist_sq + 1e-6)
                        sep_vy += d_py / (dist_sq + 1e-6)

    var vx_next = vx
    var vy_next = vy

    if neighbor_count > 0:
        var count_f32 = Float32(neighbor_count)
        
        avg_px /= count_f32
        avg_py /= count_f32
        vx_next += (avg_px - px) * COHESION_STRENGTH
        vy_next += (avg_py - py) * COHESION_STRENGTH

        avg_vx /= count_f32
        avg_vy /= count_f32
        vx_next += (avg_vx - vx) * ALIGNMENT_STRENGTH

        vx_next += sep_vx * SEPARATION_STRENGTH
        vy_next += sep_vy * SEPARATION_STRENGTH

    var speed = sqrt(vx_next * vx_next + vy_next * vy_next)
    if speed > 1e-6:
        if speed > MAX_SPEED:
            var factor = MAX_SPEED / speed
            vx_next *= factor
            vy_next *= factor
        if speed < MIN_SPEED:
            var factor = MIN_SPEED / speed
            vx_next *= factor
            vy_next *= factor

    var px_next = (px + vx_next + WORLD_WIDTH) % WORLD_WIDTH
    var py_next = (py + vy_next + WORLD_HEIGHT) % WORLD_HEIGHT

    agents_next[i, 0] = px_next
    agents_next[i, 1] = py_next
    agents_next[i, 2] = vx_next
    agents_next[i, 3] = vy_next

def main():
    with DeviceContext() as ctx:
        print("Initializing", N, "boids...")
        
        var agents_buf_A = ctx.enqueue_create_buffer[dtype](agents_layout.size())
        var agents_buf_B = ctx.enqueue_create_buffer[dtype](agents_layout.size())
        var sorted_agents_buffer = ctx.enqueue_create_buffer[dtype](agents_layout.size())
        var hashes_buffer = ctx.enqueue_create_buffer[DType.int32](hash_layout.size())
        var indices_buffer = ctx.enqueue_create_buffer[DType.int32](index_layout.size())
        var histogram_buffer = ctx.enqueue_create_buffer[DType.int32](histogram_layout.size())
        var cell_offsets_buffer = ctx.enqueue_create_buffer[DType.int32](cell_offsets_layout.size())
        var cell_offsets_counter_buffer = ctx.enqueue_create_buffer[DType.int32](cell_offsets_layout.size())

        with agents_buf_A.map_to_host() as agents_host:
            var agents_tensor_host = LayoutTensor[dtype, agents_layout](agents_host.unsafe_ptr())
            for i in range(N):
                agents_tensor_host[i, 0] = Float32(random_float64(0.0, Float64(WORLD_WIDTH)))
                agents_tensor_host[i, 1] = Float32(random_float64(0.0, Float64(WORLD_HEIGHT)))
                agents_tensor_host[i, 2] = Float32(random_float64(-2.0, 2.0))
                agents_tensor_host[i, 3] = Float32(random_float64(-2.0, 2.0))

        var current_agents_buffer = agents_buf_A
        var next_agents_buffer = agents_buf_B

        var hashes_tensor = LayoutTensor[DType.int32, hash_layout](hashes_buffer)
        var indices_tensor = LayoutTensor[DType.int32, index_layout](indices_buffer)
        var histogram_tensor = LayoutTensor[DType.int32, histogram_layout](histogram_buffer)
        var cell_offsets_tensor = LayoutTensor[DType.int32, cell_offsets_layout](cell_offsets_buffer)
        var cell_offsets_counter_tensor = LayoutTensor[DType.int32, cell_offsets_layout](cell_offsets_counter_buffer)
        var sorted_agents_tensor = LayoutTensor[dtype, agents_layout](sorted_agents_buffer)

        var blocks_per_grid = ceildiv(N, TPB)
        
        print("Starting simulation for", STEPS, "steps...")
        var total_kernel_time_ns: Int = 0
        
        var sim_start_ns = perf_counter_ns()

        for _ in range(STEPS):
            var step_start_ns = perf_counter_ns()

            var current_agents_tensor = LayoutTensor[dtype, agents_layout](current_agents_buffer)
            var next_agents_tensor = LayoutTensor[dtype, agents_layout](next_agents_buffer)

            ctx.enqueue_memset(histogram_buffer, 0)

            ctx.enqueue_function[
                spatial_hash_kernel[agents_layout, hash_layout, index_layout]
            ](hashes_tensor, indices_tensor, current_agents_tensor, grid_dim=blocks_per_grid, block_dim=TPB)

            ctx.enqueue_function[
                histogram_kernel[hash_layout, histogram_layout]
            ](hashes_tensor, histogram_tensor, grid_dim=blocks_per_grid, block_dim=TPB)
            
            ctx.enqueue_function[
                parallel_prefix_sum_kernel[histogram_layout, cell_offsets_layout]
            ](histogram_tensor, cell_offsets_tensor, grid_dim=1, block_dim=PADDED_NUM_CELLS)

            ctx.enqueue_copy(cell_offsets_counter_buffer, cell_offsets_buffer)

            ctx.enqueue_function[
                reorder_kernel[agents_layout, hash_layout, index_layout, cell_offsets_layout]
            ](
                sorted_agents_tensor,
                current_agents_tensor,
                hashes_tensor,
                indices_tensor,
                cell_offsets_counter_tensor,
                grid_dim=blocks_per_grid,
                block_dim=TPB,
            )

            ctx.enqueue_function[
                boids_update_optimized_kernel[agents_layout, histogram_layout, cell_offsets_layout]
            ](
                next_agents_tensor,
                sorted_agents_tensor,
                histogram_tensor,
                cell_offsets_tensor,
                grid_dim=blocks_per_grid,
                block_dim=TPB,
            )

            ctx.synchronize()
            
            var step_end_ns = perf_counter_ns()
            total_kernel_time_ns += (step_end_ns - step_start_ns)

            var temp_buf = current_agents_buffer
            current_agents_buffer = next_agents_buffer
            next_agents_buffer = temp_buf

        var sim_end_ns = perf_counter_ns()
        var total_duration_s = (sim_end_ns - sim_start_ns) / 1_000_000_000
        var avg_kernel_ms = (total_kernel_time_ns / STEPS) / 1_000_000

        print("\nSimulation finished.")
        print("Total steps:", STEPS)
        print("Total simulation time:", total_duration_s, "seconds")
        print("Average time per kernel step:", avg_kernel_ms, "ms")

        print("\nFinal state of first 3 boids:")
        with current_agents_buffer.map_to_host() as result_host:
            var result_tensor = LayoutTensor[dtype, agents_layout](result_host.unsafe_ptr())
            for i in range(3):
                print(
                    "  Boid", i, ": pos=(", result_tensor[i, 0][0], ",", result_tensor[i, 1][0], 
                    ") vel=(", result_tensor[i, 2][0], ",", result_tensor[i, 3][0], ")"
                )