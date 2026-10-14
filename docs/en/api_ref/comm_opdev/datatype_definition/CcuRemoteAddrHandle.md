# CcuRemoteAddrHandle

<!-- md-trans-meta sourceCommit=2b912e7d2b105735579276a63a51364890c56788 translatedAt=2026-09-28T08:50:50.318Z pushedAt=2026-10-08T03:50:22.308Z -->

## Description

Handle type of the CCU RemoteAddr (remote address), used to identify the remote on-chip memory address resource created through the [RemoteAddr](../data_plane_api/ccu/resource_allocation_operation/RemoteAddr.md) resource in a CCU kernel. This handle is used as the remote address parameter in CCU data plane cross-rank APIs (such as **Read**, **ReadReduce**, **Write**, and **WriteReduce**).

## Prototype

```c
typedef uint64_t CcuRemoteAddrHandle;
```
