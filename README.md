# High-Performance Boids Simulation in Mojo

This project demonstrates a massively parallel implementation of the [Boids artificial life simulation](https://en.wikipedia.org/wiki/Boids) on the GPU using the Mojo programming language.

The goal was to explore and implement high-performance GPU computing techniques. We began with a simple, brute-force GPU kernel and evolved it into a highly efficient, algorithmically-optimized version to showcase how kernel design impacts performance.

## Key Accomplishment: 14.5x Performance Speedup on the GPU

By redesigning our GPU kernel from a naive, brute-force algorithm to a spatially-aware one, we achieved a **~14.5x performance increase**. This was accomplished by replacing the O(N²) neighbor search with a multi-stage pipeline that sorts boids into a grid, allowing each boid to only check its immediate neighbors.

### Performance Results
*(Both versions run entirely on an **NVIDIA A10G GPU** with 100,000 boids over 200 steps)*

| Version | GPU Algorithm | Time per Step | Speedup |
| :--- | :--- | :--- | :--- |
| `kernels/boids.mojo` | Naive O(N²) | ~52.2 ms | 1x |
| `kernels/boids_full_optimized.mojo` | Spatial Grid O(N) | ~3.6 ms | **~14.5x** |

## The Optimization Pipeline

The final, optimized simulation is not a single kernel but a pipeline of parallel algorithms that work together to prepare the data for the final physics calculation.

*   **`kernels/spatial_hash.mojo`**: Calculates a 1D grid cell ID for each boid based on its 2D position.
*   **`kernels/histogram.mojo`**: Uses atomic operations to count the number of boids in each grid cell.
*   **`kernels/scan.mojo`**: Performs a parallel prefix sum (scan) on the histogram to calculate the starting memory offset for each cell.
*   **`kernels/reorder.mojo`**: Uses the calculated offsets to sort the boids into a new buffer, grouping them by their grid cell.
*   **`kernels/boids_full_optimized.mojo`**: The final update kernel. It uses the sorted data to have each boid only check its immediate 8 neighboring cells (and its own) for interactions, drastically reducing computation.

## How to Run

You can run both the naive and the final optimized versions to see the performance difference for yourself.

**Run the Naive O(N²) GPU Version:**
```bash
mojo run kernels/boids.mojo
```

**Run the Final Optimized GPU Version:**
```bash
mojo run kernels/boids_full_optimized.mojo
```

## The Optimization Journey

The significant performance gain was the result of a step-by-step process of building and verifying each component of the spatial grid pipeline. For a detailed narrative of how we went from the simple version to the final optimized one, please see our detailed log.

➡️ **[Read the Full Optimization Journey](./OPTIMIZATION_JOURNEY.md)**