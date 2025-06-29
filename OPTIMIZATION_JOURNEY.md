# From O(N²) to O(N): A Boids Optimization Journey on the GPU

This document chronicles the step-by-step process of optimizing a Boids simulation in Mojo. The journey is not about comparing CPU to GPU, but about evolving a **naive GPU implementation** into a **highly efficient, algorithmically-optimized GPU implementation**.

The goal is to demonstrate how thoughtful kernel design and data structures are critical for unlocking the true potential of parallel hardware.

## Step 1: The Baseline - A Naive O(N²) GPU Kernel

The most straightforward way to implement the Boids algorithm is with a brute-force approach, which we can still accelerate by running it on the GPU.

**File:** [`kernels/boids.mojo`](./kernels/boids.mojo)

**Concept:** Each boid must compare its position to every other boid in the simulation to find its neighbors. On the GPU, we can assign one thread to each boid, parallelizing the main loop.

```mojo
// Inside the kernel, for each boid 'i' (run by a single thread):
for j in range(N):
    if i == j:
        continue
    
    // ... calculate distance and check if 'j' is a neighbor
```

**Limitation:** While the GPU parallelizes the work for all `N` boids, each individual thread is still performing an O(N) operation. This results in a total computational complexity of O(N²). For 100,000 boids, this is 10 billion comparisons per step. This approach is limited by computation, not by the parallelism of the hardware.

*   **Baseline Performance:** ~52.2 ms per step.

## Step 2: The Strategy - Spatial Grid Hashing

The key to optimization is to avoid the O(N²) check. A boid's behavior is only influenced by its nearby neighbors.

**Concept:** We can divide the 2D world into a grid of cells. To find its neighbors, a boid only needs to check for other boids within its own cell and the 8 immediately surrounding cells. This reduces the number of comparisons from `N` to a small, roughly constant number `k`, changing the algorithm's complexity from O(N²) to O(N * k), which is effectively O(N).

To achieve this, we must first sort the boids by their grid cell location. This requires a multi-stage pipeline of specialized kernels.

## Step 3: Building the Pipeline, Kernel by Kernel

### 3.1 Hashing: Assigning Boids to Cells

**File:** [`kernels/spatial_hash.mojo`](./kernels/spatial_hash.mojo)

**Goal:** Determine which grid cell each boid belongs to and assign it a unique "hash" (a 1D cell ID).

**Implementation:** A simple kernel where each thread takes one boid, calculates its `(cell_x, cell_y)` coordinate based on its position, and computes a 1D hash: `hash = cell_y * GRID_WIDTH + cell_x`. We also store the boid's original index to use later during reordering.

For a deep dive into how the spatial grid works, see our [Spatial Grid Explainer](./SPATIAL_GRID_EXPLAINER.md).

### 3.2 Counting: The Histogram

**File:** [`kernels/histogram.mojo`](./kernels/histogram.mojo)

**Goal:** Count the number of boids that fall into each grid cell.

**Implementation:** We launch a kernel where each thread reads the hash for one boid and increments a counter for that hash value. Because thousands of threads could be trying to increment the same counter simultaneously, this creates a race condition. We solve this by using `Atomic.fetch_add()`, which guarantees that these updates are handled safely in global memory.

### 3.3 Calculating Offsets: The Parallel Scan

**File:** [`kernels/scan.mojo`](./kernels/scan.mojo)

**Goal:** From the histogram of cell counts, calculate the starting index for each cell's data in the final, sorted array. For example, if Cell 0 has 5 boids and Cell 1 has 3, Cell 0 starts at index 0 and Cell 1 starts at index 5.

**Implementation:** This is a classic parallel **prefix sum** (or scan) algorithm. Our implementation is a work-efficient Blelloch scan that must operate on power-of-two-sized arrays. Since our `NUM_CELLS` (192) is not a power of two, we pad the input to the next power of two (256) to ensure the algorithm's correctness. This is a common technique in GPU programming.

### 3.4 Sorting: The Reorder Kernel

**File:** [`kernels/reorder.mojo`](./kernels/reorder.mojo)

**Goal:** Physically reorder the boids in memory according to their cell hash, creating a new, sorted buffer.

**Implementation:** This kernel uses all the data we've prepared. Each thread takes a boid and:
1.  Reads its hash and original index.
2.  Looks up the starting offset for its hash from the prefix sum result.
3.  Uses an atomic counter on that offset to get a unique position within its cell's group.
4.  Copies the boid data from its original position to its new, sorted position.

## Step 4: The Payoff - The Optimized Update Kernels

With the boids now sorted by location, the main update kernel no longer needs to check all N boids. It only needs to check the contents of its own cell and the 8 adjacent cells.

### 4.1 Optimized Alignment-Only Kernel

**File:** [`kernels/boids_optimized.mojo`](./kernels/boids_optimized.mojo)

**Concept:** First, we implement an optimized version of the boids update that only considers the **alignment** rule. This allows us to verify that the spatial grid pipeline is working correctly and to measure the performance gain from the algorithmic change alone, without the complexity of the other rules.

*   **Intermediate Performance:** ~3.0 ms per step.

### 4.2 Full Optimized Kernel (Separation, Alignment, Cohesion)

**File:** [`kernels/boids_full_optimized.mojo`](./kernels/boids_full_optimized.mojo)

**Concept:** Now we expand the optimized kernel to include all three classic Boids rules:
*   **Separation:** Steer to avoid crowding local flockmates.
*   **Alignment:** Steer towards the average heading of local flockmates.
*   **Cohesion:** Steer to move toward the average position of local flockmates.

This kernel still uses the efficient 3x3 neighbor cell check, but performs more calculations for each neighbor found.

*   **Final Performance:** ~3.6 ms per step.

## Conclusion

By chaining these kernels together in each step of the simulation (`hash -> histogram -> scan -> reorder -> update`), we fundamentally change the algorithm's complexity. This algorithmic improvement, executed in parallel on the GPU, is what provides the final **~14.5x speedup** over the naive GPU implementation, demonstrating that intelligent algorithm design is paramount for achieving high performance on parallel hardware.