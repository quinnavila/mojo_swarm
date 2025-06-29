from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from layout import Layout, LayoutTensor
from math import ceildiv, floor
from random import random_float64

alias N = 4096
alias TPB = 256
alias WORLD_WIDTH: Float32 = 800.0
alias WORLD_HEIGHT: Float32 = 600.0
alias CELL_SIZE: Float32 = 50.0 

alias GRID_COLS = Int(floor(WORLD_WIDTH / CELL_SIZE))
alias GRID_ROWS = Int(floor(WORLD_HEIGHT / CELL_SIZE))

alias dtype = DType.float32
alias agents_layout = Layout.row_major(N, 4)
alias hash_layout = Layout.row_major(N)
alias index_layout = Layout.row_major(N)

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

def main():
    with DeviceContext() as ctx:
        print("Initializing GPU buffers...")
        agents_buffer = ctx.enqueue_create_buffer[dtype](agents_layout.size())
        hashes_buffer = ctx.enqueue_create_buffer[DType.int32](hash_layout.size())
        indices_buffer = ctx.enqueue_create_buffer[DType.int32](index_layout.size())

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

        var blocks_per_grid = ceildiv(N, TPB)

        print("Launching spatial hash kernel on GPU...")
        ctx.enqueue_function[
            spatial_hash_kernel[agents_layout, hash_layout, index_layout]
        ](
            hashes_tensor,
            indices_tensor,
            agents_tensor,
            grid_dim=blocks_per_grid,
            block_dim=TPB,
        )

        ctx.synchronize()
        print("Kernel execution finished.")

        with hashes_buffer.map_to_host() as result_hashes, \
             agents_buffer.map_to_host() as result_agents:
            
            var hash_result_tensor = LayoutTensor[DType.int32, hash_layout](result_hashes)
            var agent_result_tensor = LayoutTensor[dtype, agents_layout](result_agents)

            print("First 10 boids and their calculated spatial hash:")
            for i in range(10):
                print(
                    "  Boid", i, 
                    ": pos=(", agent_result_tensor[i, 0], ",", agent_result_tensor[i, 1], ")",
                    "-> Hash:", hash_result_tensor[i][0]
                )