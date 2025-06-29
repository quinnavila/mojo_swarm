# Understanding the Spatial Grid: From Radius to Cells

The core of our optimization is replacing a slow, precise neighbor search with a fast, approximate one, followed by a precise check on a much smaller group. This document explains how the "cell" in our spatial grid is analogous to the "radius" in the naive version.

### The Core Problem with the Radius Search

In our naive `O(N²)` solution, every boid had to ask a simple but expensive question: "Which of the other 99,999 boids are within my perception radius?"

To answer this, it had to perform a distance calculation for every single other boid. This is computationally expensive but guarantees a perfect, circular neighborhood.



### The Solution: A Grid of Cells

The spatial grid changes the question. Instead of asking "who is near me?", a boid now asks a much faster, two-part question:

1.  "Which grid cell am I in?" (This is extremely fast to calculate).
2.  "Who else is in my cell, or in the 8 cells surrounding me?"

A **cell** is essentially a square "bucket" in our 2D world. By sorting all the boids into these buckets, we can instantly ignore the vast majority of boids that are too far away to possibly be neighbors.

A cell is therefore an **approximation of a neighborhood**.

### The Data Flow: A Step-by-Step Example

Let's trace 5 boids (A, B, C, D, E) through the entire process to see how the data is transformed.

**Scenario:**
*   A 3x2 world grid (`NUM_CELLS = 6`).
*   `CELL_SIZE` is equal to the boid perception radius `R`.



---

#### **Step 1: Spatial Hashing (Putting Boids in Buckets)**

**Goal:** Assign a unique cell ID (a "hash") to every boid.
*   Boid A is in cell 0 -> `hash = 0`
*   Boid B is in cell 5 -> `hash = 5`
*   Boid C is in cell 0 -> `hash = 0`
*   Boid D is in cell 2 -> `hash = 2`
*   Boid E is in cell 5 -> `hash = 5`

---

#### **Step 2: Histogram (Counting the Buckets)**

**Goal:** Count how many boids are in each cell. The `histogram` array is a block of memory where the index is the cell hash and the value is the count.

| Index (Hash) | 0 | 1 | 2 | 3 | 4 | 5 |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: |
| **Count** | 2 | 0 | 1 | 0 | 0 | 2 |

*This tells us: Cell 0 has 2 boids, Cell 2 has 1, Cell 5 has 2, and the others are empty.*

---

#### **Step 3: Prefix Sum / Scan (Creating the "Address Book")**

**Goal:** Calculate the starting memory address (offset) for each cell's group in the final sorted array. The offset for a cell is the sum of all counts *before* it.

| Index (Hash) | 0 | 1 | 2 | 3 | 4 | 5 |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: |
| **Count (Histogram)** | 2 | 0 | 1 | 0 | 0 | 2 |
| **Offset (Prefix Sum)** | 0 | 2 | 2 | 3 | 3 | 3 |

*This tells us: The boids for Cell 0 will start at index **0** in our final sorted array. The boids for Cell 2 will start at index **2**, and so on.*

---

#### **Step 4: Reordering (The Great Sort)**

**Goal:** Physically move the boids in memory so they are grouped by cell. The `reorder_kernel` uses the offsets to place each boid into its correct new slot.

The final `sorted_agents` array (our final block of memory) is now perfectly grouped by cell:

| Index | 0 | 1 | 2 | 3 | 4 |
| :--- | :-: | :-: | :-: | :-: | :-: |
| **Boid** | A | C | D | B | E |
| **Cell** | **Cell 0** | **Cell 2** | **Cell 5** |

---

### The Payoff: The Final, Optimized Search

Now we get to the `boids_update_optimized_kernel`. For any given boid, instead of checking all N boids, it does the following:

1.  Finds its own cell (e.g., Boid D is in Cell 2).
2.  Identifies its 3x3 neighborhood of cells (Cells 1, 2, 3, 4, 5, 6).
3.  For each of those 9 cells, it uses the `cell_offsets` and `histogram` arrays to find the exact slice of memory containing the boids in that cell.
4.  It loops *only* through those small slices of boids, performing the final, precise distance check (`dist_sq < R * R`).

Instead of checking all 100,000 boids, it might only check the 30-40 boids that are in its local 3x3 grid.

**Conclusion:** The cell is not the neighborhood itself, but an incredibly efficient data structure for **finding the potential neighborhood**. The combination of a coarse grid lookup followed by a fine-grained radius check on a tiny subset of the data is the source of the massive performance gain.