---
# Probabilistic Graphical Models: A Curated Collection

This repository provides a comprehensive set of **Bayesian Networks (BNs)** and **Grid-based Networks** in JSON format. These models are designed for testing, benchmarking, and developing probabilistic reasoning systems, from simple causal chains to large-scale spatial grids.

## Key Features

- **Standardized Format**: All networks are defined in a clean, easy-to-parse JSON structure.
- **Directed Acyclic Graphs (DAGs)**: Every network is guaranteed to be a DAG, suitable for probabilistic inference.
- **Binary Variables**: Nodes represent binary (True/False) random variables for straightforward implementation.
- **Defined Probabilities**: Each model includes a complete set of Conditional Probability Tables (CPTs) or prior probabilities.
- **Rich Metadata**: Complex networks include descriptions, dependency types, and complexity factors to guide selection.

---

## How to Choose a Network

To select the right model for your task, first ask:

1.  **Is my problem about causal or logical relationships?**
    - **Yes**: Your problem involves cause-and-effect, diagnosis, or abstract reasoning. **Use a Bayesian Network.**
    - **No**: Your problem is primarily spatial, structural, or involves interactions between adjacent entities (like pixels or sensors). **Use a Grid Network.**

2.  **What is the scale of my problem?**
    - Start with the smallest network that fits your conceptual model (e.g., a simple chain, fork, or 2x2 grid).
    - Increase complexity to test scalability, performance, and the robustness of your inference algorithms.

---

## 🔗 Bayesian Networks (For Causal & Logical Reasoning)

Bayesian Networks model probabilistic relationships between variables. They are ideal for tasks like root cause analysis, diagnostics, and modeling systems where variables have distinct, semantic meanings.

| Model / Size | File                                                     | Nodes | Structure / Key Use Case                                                                              |
| :----------- | :------------------------------------------------------- | :---- | :---------------------------------------------------------------------------------------------------- |
| **2-Node**   | [`bayesian_network_2x2.json`](./bayesian_network_2x2.json) | 4     | **Simple Causality**: A minimal `X → Y` structure for basic inference checks.                        |
| **3-Node**   | [`bayesian_network_3x3.json`](./bayesian_network_3x3.json) | 9     | **Causal Chain**: A `X → Y → Z` model for testing belief propagation.                                 |
| **4-Node**   | [`bayesian_network_4x4.json`](./bayesian_network_4x4.json) | 16     | **Common Cause (Fork)**: `X → {Y, Z, W}`. Ideal for modeling a single cause with multiple effects.      |
| **5-Node**   | [`bayesian_network_5x5.json`](./bayesian_network_5x5.json) | 25     | **Tree Structure**: `A → {B, C}, B → {D, E}`. For hierarchical reasoning and branching paths.             |
| **6-Node**   | [`bayesian_network_6x6.json`](./bayesian_network_6x6.json) | 36     | **Collider Structure**: `A→C←B`. Excellent for testing "explaining away" and conditional dependencies. |
| **7-Node**   | [`bayesian_network_7x7.json`](./bayesian_network_7x7.json) | 49     | **Balanced Tree**: `A → {B, C}, B → {D, E}, C → {F, G}`. A symmetric model for hierarchical analysis. |
| **8-Node**   | [`bayesian_network_8x8.json`](./bayesian_network_8x8.json) | 64     | **Parallel Chains**: Two independent `A→B→C→D` chains. Tests modularity and parallel inference.       |
| **9-Node**   | [`bayesian_network_9x9.json`](./bayesian_network_9x9.json) | 81     | **Three Parallel Chains**: Three `A→D→G` structures for modeling multiple independent processes.       |
| **10-Node**  | [`bayesian_network_10x10.json`](./bayesian_network_10x10.json) | 100    | **Mixed Structure**: A long chain and a wide fork. Good for testing hybrid reasoning tasks.           |
| **20-Node**  | [`bayesian_network_20x20.json`](./bayesian_network_20x20.json) | 400    | **Large Parallel Chains**: Two 10-node chains for testing scalability and parallel belief propagation.  |
| **32-Node**  | [`bayesian_network_32x32.json`](./bayesian_network_32x32.json) | 1024    | **Large Modular Model**: Four independent 8-node chains for large-scale modular causal modeling.      |
| **64-Node**  | [`bayesian_network_64x64.json`](./bayesian_network_64x64.json) | 4096    | **Stress Test Model**: Eight independent 8-node chains to stress-test inference engine performance.     |
| **Custom**   | [`bayesian_network.json`](./bayesian_network.json)         | 8     | **Complex DAG**: A non-uniform, multi-layered graph with varied dependency types for advanced scenarios. |

---

## Grid Networks (For Spatial & Structural Reasoning)

Grid Networks are a special type of Bayesian Network where nodes are arranged in a 2D lattice. Each node's state is conditionally dependent on its immediate neighbors (typically the ones above and to the left). They are perfect for modeling spatial phenomena.

| Grid Size | File                                                   | Nodes  | Primary Use Case                                    |
| :-------- | :----------------------------------------------------- | :----- | :-------------------------------------------------- |
| **2×2**   | [`grid_network_2x2.json`](./grid_network_2x2.json)     | 4      | Minimal test case for debugging grid logic.         |
| **3×3**   | [`grid_network_3x3.json`](./grid_network_3x3.json)     | 9      | Educational tool for demonstrating belief propagation. |
| **4×4**   | [`grid_network_4x4.json`](./grid_network_4x4.json)     | 16     | Analyzing local structural patterns and dependencies. |
| **5×5**   | [`grid_network_5x5.json`](./grid_network_5x5.json)     | 25     | Moderate-scale grid for light experiments.          |
| **6×6**   | [`grid_network_6x6.json`](./grid_network_6x6.json)     | 36     | Balanced model for prototyping applications.        |
| **7×7**   | [`grid_network_7x7.json`](./grid_network_7x7.json)     | 49     | Medium-scale spatial modeling tasks.                |
| **8×8**   | [`grid_network_8x8.json`](./grid_network_8x8.json)     | 64     | Approaching large-grid behavior; good for performance baselines. |
| **9×9**   | [`grid_network_9x9.json`](./grid_network_9x9.json)     | 81     | Robust structure for testing image or sensor models. |
| **10×10** | [`grid_network_10x10.json`](./grid_network_10x10.json) | 100    | Standard size for application-level testing.        |
| **20×20** | [`grid_network_20x20.json`](./grid_network_20x20.json) | 400    | Large-scale spatial inference and scalability tests.  |
| **32×32** | [`grid_network_32x32.json`](./grid_network_32x32.json) | 1024   | High-resolution spatial representation; performance stress-testing. |
| **64×64** | [`grid_network_64x64.json`](./grid_network_64x64.json) | 4096   | Very large grid for big data scenarios and benchmarking approximate inference. |

---

## Usage Example

The networks can be loaded and used with a library like `pgmpy` or a custom Python wrapper. The following demonstrates a typical query using the provided `probabilistic_models.py` class, which encapsulates the logic.

1.  **Initialize the Model**: This loads the default network and knowledge base.
    ```python
    from probabilistic_models import ProbabilisticModels
    model = ProbabilisticModels()
    ```

2.  **Select a Network (Optional)**: Choose a network based on your task requirements.
    ```python
    # Select a network for a medium-complexity contextual reasoning task
    network_file = model.select_network(
        task_type="contextual_reasoning",
        complexity="medium",
        speed_requirement="balanced"
    )
    model.bayesian_network = model._load_bayesian_network(Path(network_file))
    ```

3.  **Run Queries**: Perform inference by providing a query variable and evidence.
    ```python
    # Query node 'H' with evidence that 'A' is True and 'C' is False
    probability = model.bayesian_inference(
        query="H", 
        evidence={"A": True, "C": False}
    )
    print(f"The probability is: {probability}")
    ```
