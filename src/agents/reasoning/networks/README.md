# 🧠 Probabilistic Graphical Models: Bayesian and Grid Networks

This repository contains examples and utilities for working with **Bayesian Networks** and **Grid-based Bayesian Networks**, modeled as probabilistic graphical structures using `pgmpy`.

## 📌 What Are Bayesian Networks?

**Bayesian Networks (BNs)** are **directed acyclic graphs** (DAGs) where:
- **Nodes** represent random variables.
- **Edges** represent conditional dependencies.
- **Conditional Probability Tables (CPTs)** encode the probabilities associated with each node, given its parents.

They allow efficient reasoning about uncertainty using probabilistic inference.

### Bayesian Layouts Provided:
| Grid Size | File                          | Node Count | Use Case                           |
|-----------|-------------------------------|------------|------------------------------------|
| 2×2       | [`bayesian_network_2x2.json`](./bayesian_network_2x2.json) | 4          | A simple chain of influence: X → Y → Z |
| 3×3       | [`bayesian_network_3x3.json`](./bayesian_network_3x3.json) | 9          |  |
| 4x4       | [`bayesian_network_4x4.json`](./bayesian_network_4x4.json) | 16         |  |
| 5x5       | [`bayesian_network_5x5.json`](./bayesian_network_5x5.json) | 25         |  |
| 6x6       | [`bayesian_network_6x6.json`](./bayesian_network_6x6.json) | 36         |  |
| 7×7       | [`bayesian_network_7x7.json`](./bayesian_network_7x7.json) | 49         |  |
| 8x8       | [`bayesian_network_8x8.json`](./bayesian_network_8x8.json) | 64         |  |
| 9x9       | [`bayesian_network_9x9.json`](./bayesian_network_9x9.json) | 81         |  |
| 10×10     | [`bayesian_network_10x10.json`](./bayesian_network_10x10.json) | 100        |  |
| 20x20     | [`bayesian_network_20x20.json`](./bayesian_network_20x20.json) | 400         |  |
| 32x32     | [`bayesian_network_32x32.json`](./bayesian_network_32x32.json) | 1024         |  |
| 64x64     | [`bayesian_network_64x64.json`](./bayesian_network_64x64.json) | 4096         |  |


## 🔳 What Are Grid Networks?

**Grid Bayesian Networks** are structured as a 2D grid (like a lattice). Each node:
- Represents a local variable (e.g., pixel, cell, sensor).
- Depends on neighbors (e.g., left, top).

These networks are commonly used for:
- **Image denoising**
- **Sensor networks**
- **Terrain modeling**
- **Spatio-temporal inference**

### Grid Layouts Provided:
| Grid Size | File                          | Node Count | Use Case                           |
|-----------|-------------------------------|------------|------------------------------------|
| 2×2       | [`grid_network_2x2.json`](./grid_network_2x2.json) | 4          | Minimal test case/debugging       |
| 3×3       | [`grid_network_3x3.json`](./grid_network_3x3.json) | 9          | Education / propagation demo      |
| 4x4       | [`grid_network_4x4.json`](./grid_network_4x4.json) | 16         | Useful for structural pattern analysis |
| 5x5       | [`grid_network_5x5.json`](./grid_network_5x5.json) | 25         | Moderate-size grid                |
| 6x6       | [`grid_network_6x6.json`](./grid_network_6x6.json) | 36         | Light-weight experiments          |
| 7×7       | [`grid_network_7x7.json`](./grid_network_7x7.json) | 49         | Medium-scale modeling             |
| 8x8       | [`grid_network_8x8.json`](./grid_network_8x8.json) | 64         | Approaching large grid behavior
| 9x9       | [`grid_network_9x9.json`](./grid_network_9x9.json) | 81         |  |
| 10×10     | [`grid_network_10x10.json`](./grid_network_10x10.json) | 100        | Robust structure for applications |
| 20x20     | [`grid_network_20x20.json`](./grid_network_3x3.json) | 400         | Large-scale inference testing |
| 32x32     | [`grid_network_32x32.json`](./grid_network_3x3.json) | 1024         | High-resolution spatial representation |
| 64x64     | [`grid_network_64x64.json`](./grid_network_3x3.json) | 4096         | Very large grid; suitable for big data BNs |

Each of these networks has:
- Binary-valued nodes
- Priors or conditional CPTs based on 1 or 2 neighbors
- Strictly acyclic edges (e.g., top-down, left-right)

---

## 🗂️ File Overview

| File Name                   | Description                                        |
|----------------------------|----------------------------------------------------|
| `bayesian_network.json`     | Custom DAG with labeled nodes, CPTs, and metadata |
| `grid_network_2x2.json`     | Simple 2x2 test grid                              |
| `grid_network_3x3.json`     | 3x3 educational grid with uniform CPTs            |
| `grid_network_5x5.json`     | Balanced scale grid                               |
| `grid_network_7x7.json`     | Medium-sized grid                                 |
| `grid_network_10x10.json`   | Application-ready spatial structure               |

---

## 🧮 How Inference Works

1. **Load Network** from JSON using your wrapper class.
2. **Compile Model**: Define nodes, edges, and CPTs in `pgmpy`.
3. **Run Queries** using variable elimination:
   ```python
   pbn.query("H", {"A": True, "C": False})
