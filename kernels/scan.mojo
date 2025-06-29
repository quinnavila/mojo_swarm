from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from gpu import barrier
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from math import ceildiv, floor, log2
from random import random_float64
from os.atomic import Atomic

alias N = 4096
alias TPB = 256
alias WORLD_WIDTH: Float32 = 800.0
alias WORLD_HEIGHT: Float32 = 600.0
alias CELL_SIZE: Float32 = 50.0 

alias GRID_COLS = Int(floor(WORLD_WIDTH / CELL_SIZE))
alias GRID_ROWS = Int(floor(WORLD_HEIGHT / CELL_SIZE))
alias NUM_CELLS = GRID_COLS * GRID_ROWS

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
    var hash = cell_y * GRID_COLS + cell_x
    
    particle_hashes.store[1](i, 0, SIMD[DType.int32, 1](hash))
    particle_indices.store[1](i, 0, SIMD[DType.int32, 1](i))

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
    var shared_data = tb[DType.int32]().row_major[NUM_CELLS]().shared().alloc()
    var i = thread_idx.x

    if i < NUM_CELLS:
        shared_data[i] = cell_counts[i][0]
    barrier()

    var log2_num_cells = Int(log2(Float64(NUM_CELLS)))

    for d in range(log2_num_cells):
        var stride = 1 << (d + 1)
        if (i + 1) % stride == 0:
            shared_data[i] += shared_data[i - (stride // 2)]
        barrier()

    if i == NUM_CELLS - 1:
        shared_data[NUM_CELLS - 1] = 0
    barrier()

    for d in range(log2_num_cells - 1, -1, -1):
        var stride = 1 << (d + 1)
        if (i + 1) % stride == 0:
            var temp = shared_data[i - (stride // 2)]
            shared_data[i - (stride // 2)] = shared_data[i]
            shared_data[i] += temp
        barrier()

    if i < NUM_CELLS:
        cell_offsets.store[1](i, 0, SIMD[DType.int32, 1](shared_data[i][0]))

def main():
    with DeviceContext() as ctx:
        print("Initializing GPU buffers...")
        agents_buffer = ctx.enqueue_create_buffer[dtype](agents_layout.size())
        hashes_buffer = ctx.enqueue_create_buffer[DType.int32](hash_layout.size())
        indices_buffer = ctx.enqueue_create_buffer[DType.int32](index_layout.size())
        histogram_buffer = ctx.enqueue_create_buffer[DType.int32](histogram_layout.size())
        cell_offsets_buffer = ctx.enqueue_create_buffer[DType.int32](cell_offsets_layout.size())
        
        ctx.enqueue_memset(histogram_buffer, SIMD[DType.int32, 1](0))
        ctx.enqueue_memset(cell_offsets_buffer, SIMD[DType.int32, 1](0))

        with agents_buffer.map_to_host() as agents_host:
            var agents_tensor_host = LayoutTensor[dtype, agents_layout](agents_host)
            for i in range(N):
                agents_tensor_host[i, 0] = Float32(random_float64(0.0, Float64(WORLD_WIDTH)))
                agents_tensor_host[i, 1] = Float32(random_float64(0.0, Float64(WORLD_HEIGHT)))
                agents_tensor_host[i, 2] = Float32(random_float64(-2.0, 2.0))
                agents_tensor_host[i, 3] = Float32(random_float64(-2.0, 2.0))

        var agents_tensor = LayoutTensor[dtype, agents_layout](agents_buffer)
        var hashes_tensor = LayoutTensor[DType.int32, hash_layout](hashes_buffer)
        var indices_tensor = LayoutTensor[DType.int32, index_layout](indices_buffer)
        var histogram_tensor = LayoutTensor[DType.int32, histogram_layout](histogram_buffer)
        var cell_offsets_tensor = LayoutTensor[DType.int32, cell_offsets_layout](cell_offsets_buffer)

        var blocks_per_grid = ceildiv(N, TPB)

        print("Launching spatial hash kernel...")
        ctx.enqueue_function[
            spatial_hash_kernel[agents_layout, hash_layout, index_layout]
        ](
            hashes_tensor,
            indices_tensor,
            agents_tensor,
            grid_dim=blocks_per_grid,
            block_dim=TPB,
        )

        print("Launching histogram kernel...")
        ctx.enqueue_function[
            histogram_kernel[hash_layout, histogram_layout]
        ](
            hashes_tensor,
            histogram_tensor,
            grid_dim=blocks_per_grid,
            block_dim=TPB,
        )

        print("Launching parallel prefix sum kernel...")
        var prefix_sum_threads = 1
        while prefix_sum_threads < NUM_CELLS:
            prefix_sum_threads *= 2
        
        ctx.enqueue_function[
            parallel_prefix_sum_kernel[histogram_layout, cell_offsets_layout]
        ](
            histogram_tensor,
            cell_offsets_tensor,
            grid_dim=1,
            block_dim=prefix_sum_threads,
        )

        ctx.synchronize()
        print("Kernel execution finished.")

        with cell_offsets_buffer.map_to_host() as result_offsets:
            var offsets_tensor = LayoutTensor[DType.int32, cell_offsets_layout](result_offsets)
            print("Cell offsets (first 20):")
            for i in range(20):
                print("  Cell", i, "starts at index:", offsets_tensor[i][0])