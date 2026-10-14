# HcommThreadAlloc

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:02:04.543Z pushedAt=2026-09-29T07:03:01.499Z -->

## Supported Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Supported
<!-- end id3 -->
<!-- npu="910" id4 -->
- Atlas training products: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas inference products: Not supported
<!-- end id5 -->

## Description

Allocates communication threads. Currently, the AI CPU, AI CPU+TS, HOST CPU, and HOST CPU+TS communication engines are supported. Note: If the communication engine is AI CPU+TS, an additional kernel delivery must be performed before the AI CPU side can use this communication thread.

## Function Prototype

```c
HcommResult HcommThreadAlloc(CommEngine engine, uint32_t threadNum, const uint32_t *notifyNumPerThread, ThreadHandle* threads)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| engine | Input | Communication engine type.<br>For the definition of the CommEngine type, see [CommEngine](../../datatype_definition/CommEngine.md). |
| threadNum | Input | Number of communication threads. The maximum number of threads that can be allocated per call to this API is 200. |
| notifyNumPerThread | Input | Number of synchronization resources (Notify) in each communication thread. The maximum number of Notify resources that can be allocated per call to this API for each communication thread is 64. |
| threads | Output | Returned communication thread handles. Pass in a **ThreadHandle** array of size **threadNum**.<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../datatype_definition/ThreadHandle.md). |

## Return Value

**HcommResult**: The API returns **0** on success and other values on failure.

## Constraints

1. The threads allocated by calling this API must be released by calling [HcommThreadFree](HcommThreadFree.md) later. Before calling **HcommThreadAlloc** to allocate threads of communication engines such as AICPU_TS or CPU_TS, you must first call **aclrtSetdevice** on the same thread to specify **deviceId**.

2. This API does not support the **COMM_ENGINE_AIV** and **COMM_ENGINE_CCU** communication engines.

## Example

```c
ThreadHandle thread[2];
// Allocate two streams, each with 3 Notify resources.
const uint32_t notifyNumPerThread[2] = {3, 3};
HcommResult ret =  HcommThreadAlloc(COMM_ENGINE_AICPU_TS, 2, notifyNumPerThread, thread);
```
