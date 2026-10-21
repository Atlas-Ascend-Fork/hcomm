# Selecting Algorithms

<!-- md-trans-meta sourceCommit=b91b5fc610070a3c9e821377fe6b4d1c01750eae translatedAt=2026-10-08T07:55:39.780Z pushedAt=2026-10-09T07:18:21.875Z -->

## Selection Strategy

When the communication engine is AIV, communication operators are expanded on the Vector Core. The paths for delivery and expansion are short with low latency overhead, making it suitable for inference scenarios with small communication data volumes and high requirements on latency.

A communication operator often has multiple algorithm implementations. However, for AIV communication operators, the Mesh algorithm is recommended because it requires very few communication steps, further reducing communication latency.

The following figure shows the communication algorithm selection diagram of the AIV engine:

![AIV engine communication algorithm selection diagram](figures/aiv_algo_select.png)

> [!NOTE] Note
>
> 1. If a communication operator supports only one communication engine and algorithm implementation, you can skip the algorithm selection step in this section.
> 2. The algorithms described in this chapter are implemented by developers, while the algorithms configured through the **HCCL_ALGO** environment variable are built-in HCCL algorithms. For details about the built-in HCCL algorithms, see collection communication algorithm introduction under relevant reference in the *[HCCL Collective Communication Library User Guide](https://gitcode.com/cann/hccl/blob/9.2.0/docs/en/user_guide/README.md)*.

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
