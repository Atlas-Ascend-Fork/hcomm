# Selecting Algorithms

<!-- md-trans-meta sourceCommit=e169993072c8b4f8692fd6915a9e65337ecf402f translatedAt=2026-10-08T07:55:51.580Z pushedAt=2026-10-09T07:18:21.875Z -->

## Selection Strategy

Using the CCU engine saves on-chip memory bandwidth, reduces communication expansion time compared with the AI CPU mode, and does not occupy computing hardware (for example, AIV). However, the hardware resources of the CCU are limited. Therefore, it is recommended to use the CCU engine in key scenarios that require ultimate performance, such as TP and EP parallelism.

**Figure 1** Communication algorithm selection diagram
![](figures/ccu_algo_select.png "Communication algorithm selection diagram")

As shown in the preceding figure, the CCU communication operator supports multiple algorithm implementations. Developers can select the optimal communication algorithm based on topology information.

- Mesh algorithm implementation: suitable for scenarios where the intra-server physical topology is Mesh.
- NHR algorithm implementation: suitable for multi-server scenarios where one rank is selected from each server for communication.

> [!NOTE] Note
>
>1. If a communication operator has only one algorithm implementation, you can skip the algorithm selection step in this section.
>2. The algorithms described in this chapter are implemented by developers. The algorithms configured by the **HCCL\_ALGO** environment variable are HCCL built-in algorithms. For details about HCCL built-in algorithms, see collective communication algorithm introduction under relevant reference in the *[HCCL Collective Communication Library User Guide](https://gitcode.com/cann/hccl/blob/9.2.0/docs/en/user_guide/README.md)*.

## Sample Code

The following code snippet shows the algorithm selection logic based on the selection strategy described in [Selection Strategy](#selection-strategy):

```c
CommEngine engine;
if (engine == CommEngine::COMM_ENGINE_CCU) {
    algName = "CCUAllGatherMesh";  // Select the Mesh algorithm of the CCU engine.
} else {
    return HCCL_E_NOT_SUPPORT;
}
```
