# Selecting Algorithms

<!-- md-trans-meta sourceCommit=1ed0535ba46025e54fb68f42726e7319b84ac5ca translatedAt=2026-08-11T07:08:10.874Z pushedAt=2026-08-20T11:39:14.563Z -->

## Selection Strategy

When the communication engine is AIV, communication operators are expanded on the Vector Core. The paths for delivery and expansion are short with low latency overhead, making it suitable for inference scenarios with small communication data volumes and high requirements on latency.

A communication operator often has multiple algorithm implementations. However, for AIV communication operators, the Mesh algorithm is recommended because it requires very few communication steps, further reducing communication latency.

The following figure shows the communication algorithm selection diagram of the AIV engine:

![AIV engine communication algorithm selection diagram](figures/aiv_algo_select.png)

> [!NOTE] Note
>
> 1. If a communication operator supports only one communication engine and algorithm implementation, you can skip the algorithm selection step described in this section.
> 2. The algorithms described in this chapter are those implemented by developers. The algorithms configured through the HCCL_ALGO environment variable are built-in HCCL algorithms. For details about built-in HCCL algorithms, see "References > Introduction to Collective Communication Algorithms" in *[HCCL User Guide](https://gitcode.com/cann/hccl/blob/master/docs/en/user_guide/README.md)*.

## Sample Code

Taking the algorithm selection strategy described in [Selection Strategy](#selection-strategy) as an example, the corresponding algorithm selection code snippet is as follows:

```c
CommEngine engine;
if (engine == CommEngine::COMM_ENGINE_AIV) {
    algName = "AivMesh";  // Select the Mesh algorithm expanded by AIV.
} else {
    return HCCL_E_NOT_SUPPORT;
}
```
