# mojo_swarm
Boids artificial life in mojo


# Boids Simulation: A Parallel GPU Implementation Plan

Starting with a foundational Python implementation to establish a performance baseline, followed by a parallelized Mojo version.

## The Boids Algorithm

The Boids algorithm, developed by Craig Reynolds in 1986, is an artificial life program that simulates the flocking behavior of birds. Each "boid" (bird-oid object) follows a simple set of rules, and the complex, emergent behavior of the flock arises from the interaction of these rules among individual boids.

The simulation operates in discrete time steps. In each step, every boid independently calculates its next move based on the state of its neighbors.

### Core Concepts

-   **Boid:** An individual agent in the simulation. Each boid has a position and a velocity.
-   **Neighbors:** A boid's neighbors are all other boids within a defined `perception_radius`.
-   **Rules:** Each boid adjusts its velocity based on three fundamental steering behaviors related to its neighbors.

## Phase 1: Implementing the Alignment Rule

To begin, we will implement the simplest and most visually impactful of the three rules: **Alignment**. This will allow us to build and validate the core O(n²) simulation structure on both the CPU (Python) and GPU (Mojo) before introducing further complexity.

### Alignment

-   **Goal:** Steer in the same general direction as your neighbors.
-   **Effect:** This rule causes boids to form groups that move together, mimicking the coordinated movement of a flock of birds or a school of fish.

#### Algorithm for a Single Boid (`i`):

1.  **Initialization:** Create an empty list of neighbors. Initialize an `average_velocity` vector to (0, 0) and a `neighbor_count` to 0.

2.  **Neighbor Search (The O(n²) part):**
    -   Iterate through every other boid (`j`) in the simulation.
    -   Calculate the distance between boid `i` and boid `j`.
    -   If the distance is less than the `perception_radius`, then boid `j` is a neighbor.
        -   Add the velocity of boid `j` to the `average_velocity` vector.
        -   Increment the `neighbor_count`.

3.  **Calculate Average Velocity:**
    -   If `neighbor_count` is greater than 0, divide the `average_velocity` vector by `neighbor_count` to get the true average.

4.  **Update Velocity:**
    -   If there were neighbors, calculate a "steering vector" by finding the difference between the `average_velocity` and the boid's current velocity.
    -   Apply a small fraction (`ALPHA`) of this steering vector to the boid's current velocity. This creates a smooth turning motion rather than an instantaneous snap to the new direction.
    -   The new velocity is: `new_velocity = current_velocity + ALPHA * (average_velocity - current_velocity)`.
    -   If there were no neighbors, the boid's velocity remains unchanged.

5.  **Update Position:**
    -   Update the boid's position by adding its newly calculated velocity: `new_position = current_position + new_velocity`.

6.  **Handle Boundaries:**
    -   To keep the boids within the simulation space, implement a wrap-around (toroidal) boundary. If a boid moves off one edge of the screen, it reappears on the opposite edge.

### GPU Parallelization Strategy

The O(n²) algorithm is highly parallelizable. We can assign one GPU thread to each boid.

-   **Kernel Launch:** Launch a 1D grid of threads, where the total number of threads is equal to the number of boids (`N`).
-   **Thread Work:** Each thread `i` will be responsible for executing the algorithm described above for boid `i`.
    -   The thread will read the global state of all `N` boids from GPU memory.
    -   It will perform the neighbor search loop (`for j in range(N)`).
    -   It will calculate the new position and velocity for boid `i`.
    -   It will write the updated state for boid `i` to a new output array in GPU memory.
-   **Data Flow:** The simulation will use two GPU buffers (`agents` and `agents_next`) to avoid race conditions. In each step, the kernel reads from `agents` and writes to `agents_next`. After the kernel completes, the pointers are swapped for the next step.

## Future Phases

Once the Alignment-only simulation is successfully implemented and benchmarked in Mojo, we will incrementally add the remaining two rules:

-   **Phase 2: Cohesion:** Steer to move toward the average position of local flockmates.
-   **Phase 3: Separation:** Steer to avoid crowding local flockmates.

Finally, we will explore more advanced O(n log n) or O(n) spatial partitioning algorithms (e.g., grid-based methods) to optimize the neighbor search, which is the primary bottleneck of the O(n²) approach.