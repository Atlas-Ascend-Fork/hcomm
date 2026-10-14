# HcommThreadResGetInfo

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:03:33.238Z pushedAt=2026-09-29T07:08:21.692Z -->

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

Obtains the underlying resources of a thread, such as streams.

## Function Prototype

```c
HcommResult HcommThreadResGetInfo(ThreadHandle thread, ThreadResType resType, uint32_t infoLen, void **info)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| thread | Input | Thread handle.<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../datatype_definition/ThreadHandle.md). A communication thread can be created through the [HcommThreadAlloc](./HcommThreadAlloc.md) API. |
| resType | Input | Underlying resource type (such as stream).<br>For the definition of the ThreadResType type, see [ThreadResType](../../datatype_definition/ThreadResType.md). |
| infoLen | Input | Size of the target resource information, which must be equal to the size of the corresponding resource type. |
| info | Output | Buffer for outputting resource information. The returned type is the obtained corresponding resource type. |

## Return Value

[HcommResult](../../datatype_definition/HcommResult.md): The API returns **0** on success and other values on failure.

## Constraints

1. This API supports only obtaining the underlying stream resource of a communication thread (type: [ThreadResTypeStream](../../datatype_definition/ThreadResTypeStream.md)).
2. The **infoLen** parameter must be equal to **sizeof(ThreadResTypeStream)**; otherwise, a parameter validation failure is returned.
3. The **info** parameter cannot be null, and the **thread** parameter cannot be **0**; otherwise, a null pointer error is returned.

## Example

```c
ThreadHandle thread;          // Handle of the thread created by HcommThreadAlloc.
const uint32_t notifyNumPerThread = 3;
HcommResult ret = HcommThreadAlloc(COMM_ENGINE_AICPU_TS, 1, &notifyNumPerThread, &thread);
if (ret != 0) {
    // Error handling.
}

ThreadResTypeStream stream;   // The info buffer must be aligned by resource type and writable.
uint32_t size = sizeof(ThreadResTypeStream);  // Must be equal to the target type size.
ret = HcommThreadResGetInfo(thread, THREAD_RES_TYPE_STREAM, size, (void**)&stream);
if (ret != 0) {
    // Error handling.
}
// Use the stream resource.
// ...

HcommThreadFree(&thread, 1);
```
