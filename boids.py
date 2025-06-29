from pathlib import Path
import time
import numpy as np
from numpy.typing import NDArray

from max.driver import CPU, Accelerator, Device, Tensor, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

def run_boids_graph(
    initial_agents: NDArray[np.float32],
    R: float,
    ALPHA: float,
    session: InferenceSession,
    device: Device,
) -> Tensor:
    
    dtype = DType.float32
    N = initial_agents.shape[0]

    agents_tensor = Tensor.from_numpy(initial_agents).to(device)
    
    mojo_op_path = Path(__file__).parent / "op"

    with Graph(
        "boids_graph",
        input_types=[
            TensorType(
                dtype,
                shape=agents_tensor.shape,
                device=DeviceRef.from_device(device),
            ),

        ],
        custom_extensions=[mojo_op_path],
    ) as graph:
        
        agents_in, = graph.inputs

        agents_out = ops.custom(
            name="boids",
            values=[
                agents_in,
                ops.constant(N, dtype=DType.int64, device=DeviceRef.CPU()),
                ops.constant(R, dtype=DType.float32, device=DeviceRef.CPU()),
                ops.constant(ALPHA, dtype=DType.float32, device=DeviceRef.CPU()),
            ],
            device=DeviceRef.from_device(device),
            out_types=[
                TensorType(
                    dtype=agents_in.tensor.dtype,
                    shape=agents_in.tensor.shape,
                    device=DeviceRef.from_device(device),
                )
            ],
        )[0].tensor
        graph.output(agents_out)

    print("Compiling Boids graph...")
    model = session.load(graph)

    print("Executing Boids step...")
    result_tensor = model.execute(agents_tensor)[0]

    assert isinstance(result_tensor, Tensor)
    return result_tensor.to(CPU())

# ... (The if __name__ == "__main__" block remains exactly the same) ...
if __name__ == "__main__":
    N = 1000
    R_param = 50.0
    ALPHA_param = 0.1

    device = CPU() if accelerator_count() == 0 else Accelerator()
    print(f"--- Accelerator count: {accelerator_count()} Running on device: {type(device)} ---")
    session = InferenceSession(devices=[device])

    np.random.seed(42)
    initial_agents_np = np.zeros((N, 4), dtype=np.float32)
    
    initial_agents_np[:, 0] = np.random.rand(N) * 800
    initial_agents_np[:, 1] = np.random.rand(N) * 600
    angles = np.random.rand(N) * 2 * np.pi
    initial_agents_np[:, 2] = np.cos(angles) * 2.0
    initial_agents_np[:, 3] = np.sin(angles) * 2.0

    # def boids_step_python(agents, R, ALPHA):
    #     N = len(agents)
    #     agents_next = np.zeros_like(agents)
    #     for i in range(N):
    #         px, py, vx, vy = agents[i]
    #         avg_vx, avg_vy = 0.0, 0.0
    #         neighbor_count = 0
    #         for j in range(N):
    #             if i == j: continue
    #             dist_sq = (px - agents[j, 0])**2 + (py - agents[j, 1])**2
    #             if dist_sq < R**2:
    #                 neighbor_count += 1
    #                 avg_vx += agents[j, 2]
    #                 avg_vy += agents[j, 3]
    #         if neighbor_count > 0:
    #             avg_vx /= neighbor_count
    #             avg_vy /= neighbor_count
    #             vx_next = vx + ALPHA * (avg_vx - vx)
    #             vy_next = vy + ALPHA * (avg_vy - vy)
    #         else:
    #             vx_next, vy_next = vx, vy
    #         px_next = px + vx_next
    #         py_next = py + vy_next
    #         agents_next[i] = [px_next, py_next, vx_next, vy_next]
    #     return agents_next

    # print("Calculating expected result with Python...")
    # expected_agents_np = boids_step_python(initial_agents_np, R_param, ALPHA_param)
    # print("Done.")

    result_from_mojo = run_boids_graph(
        initial_agents_np, R_param, ALPHA_param, session, device
    )
    
    result_np = result_from_mojo.to_numpy()

    # np.testing.assert_allclose(result_np, expected_agents_np, rtol=1e-5, atol=1e-5)
    # print("\n✅ Verification passed: Mojo kernel results match Python calculation!")