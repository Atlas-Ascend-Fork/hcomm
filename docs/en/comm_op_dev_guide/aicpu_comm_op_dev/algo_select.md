# Selecting Algorithms

<!-- md-trans-meta sourceCommit=8167da4e7ea711e122efa4d6174265e53a4c6bc8 translatedAt=2026-09-24T06:30:32.771Z pushedAt=2026-09-28T05:59:29.452Z -->

## Selection Policy

AI CPU communication operators should select a communication algorithm based on the actual physical topology. For example:

- For a single-server multi-device mesh topology, the Mesh algorithm is recommended to leverage direct inter-device links for parallel communication.

- For a topology with multiple servers and a single device per server, the NHR algorithm is recommended for communication along cross-server links.

The following figure shows the AI CPU communication algorithm selection schematic diagram.

![](./figures/algo_select_new.png)

> [!NOTE] Note
>
> 1. Algorithm selection should be based on the actual physical topology queried from the runtime environment, rather than determining the topology solely by the number of ranks.
> 2. If a custom operator supports only one physical topology and one algorithm implementation, you can skip the algorithm selection step and directly use that algorithm.

## Sample Code

```c
CommEngine engine;
if (engine == CommEngine::COMM_ENGINE_AICPU_TS) {
    algName = "AicpuMesh";  // Select the Mesh algorithm implemented by AI CPU.
} else {
    return HCCL_E_NOT_SUPPORT;
}
```
