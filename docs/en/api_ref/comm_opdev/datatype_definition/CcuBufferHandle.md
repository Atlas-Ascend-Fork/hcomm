# CcuBufferHandle

<!-- md-trans-meta sourceCommit=9a28f2fdc3bd8008a5fa6d178f4a7efe8b2d3eb9 translatedAt=2026-10-08T12:18:11.806Z pushedAt=2026-10-12T06:12:35.665Z -->

## Description

Handle type of CcuBuffer, used to identify the CcuBuffer created through [CcuBuffer](../data_plane_api/ccu/resource_allocation_operation/CcuBuffer.md) within the CCU kernel. CcuBuffer is an on-chip high-speed scratchpad area in the CCU die, used to transfer data between on-chip memory and the remote end.

## Prototype

```c
typedef uint64_t CcuBufferHandle;
```
