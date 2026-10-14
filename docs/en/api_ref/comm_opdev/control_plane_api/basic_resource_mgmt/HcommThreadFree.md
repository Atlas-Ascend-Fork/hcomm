# HcommThreadFree

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:03:07.472Z pushedAt=2026-09-29T07:05:25.259Z -->

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

Releases threads allocated by calling the **HcommThreadAlloc** API.

## Function Prototype

```c
HcommResult HcommThreadFree(const ThreadHandle* threads, uint32_t threadNum)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| threads | Input | Communication thread handle.<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../datatype_definition/ThreadHandle.md). |
| threadNum | Input | Number of communication threads. |

## Return Value

**HcommResult**: The API returns **0** on success and other values on failure.

## Constraints

Only threads allocated by the **HcommThreadAlloc** API can be released.

## Example

```c
ThreadHandle thread[2];
const uint32_t notifyNumPerThread[2] = {3, 3};
HcommResult ret = HcommThreadAlloc(COMM_ENGINE_AICPU_TS, 2, notifyNumPerThread, thread);
ret = HcommThreadFree(thread, 2);
```
