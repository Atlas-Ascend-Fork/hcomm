# CcuLocalAddrHandle

<!-- md-trans-meta sourceCommit=9a28f2fdc3bd8008a5fa6d178f4a7efe8b2d3eb9 translatedAt=2026-10-08T12:18:32.443Z pushedAt=2026-10-12T06:12:35.665Z -->

## Description

Handle type of the CCU LocalAddr (local address), used to identify the local on-chip memory address resource created in the CCU kernel through the [LocalAddr](../data_plane_api/ccu/resource_allocation_operation/LocalAddr.md) resource. This handle is used as the local address parameter in CCU data plane APIs (such as **LocalCopy**, **LocalReduce**, **Read**, and **Write**).

## Prototype

```c
typedef uint64_t CcuLocalAddrHandle;
```
