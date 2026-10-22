# Selecting Algorithms

<!-- md-trans-meta sourceCommit=2b912e7d2b105735579276a63a51364890c56788 translatedAt=2026-09-24T07:07:29.286Z pushedAt=2026-10-08T08:31:03.159Z -->

## Selection Strategy

The CCU engine saves HBM bandwidth, reduces communication expansion time compared with the AI CPU mode, and does not occupy computing hardware (for example, AIV). However, CCU hardware resources are limited. Therefore, it is recommended to use the CCU engine in key scenarios that require ultimate performance, such as TP and EP parallelism.

**Figure 1** Communication algorithm selection diagram
![](figures/ccu_algo_select.png "Communication algorithm selection diagram")

As shown in the preceding figure, the CCU communication operator supports multiple algorithm implementations. Developers can select the optimal communication algorithm based on topology information.

- Mesh algorithm implementation: suitable for scenarios where the intra-server physical topology is Mesh.
- NHR algorithm implementation: suitable for multi-server scenarios where one rank is selected from each server for communication.

> [!NOTE] Note
>
>1. If a communication operator has only one algorithm implementation, you can skip the algorithm selection step in this section.
>2. The algorithms described in this chapter are implemented by developers. The algorithms configured by the **HCCL\_ALGO** environment variable are built-in HCCL algorithms. For details about the built-in HCCL algorithms, see the collective communication algorithm introduction under the relevant reference in *[HCCL Collective Communication Library User Guide](https://gitcode.com/cann/hccl/blob/9.3.0-beta.1/docs/en/user_guide/README.md)*.

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
