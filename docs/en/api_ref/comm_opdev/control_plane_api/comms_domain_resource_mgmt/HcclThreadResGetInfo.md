# HcclThreadResGetInfo

<!-- md-trans-meta sourceCommit=306b11da159831ad0e3dbb73fec73f1ad849a0a2 translatedAt=2026-10-08T11:53:18.725Z pushedAt=2026-10-12T06:12:35.609Z -->

## Supported Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Not supported
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
HcclResult HcclThreadResGetInfo(HcclComm comm, ThreadHandle thread, ThreadResType resType, uint32_t infoLen, void **info)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator handle.<br>For the definition of the HcclComm type, see [HcclComm](../../../comm_mgr_c/data_type_definition/HcclComm.md). |
| thread | Input | Thread handle.<br>For the definition of the ThreadHandle type, see [ThreadHandle](../../datatype_definition/ThreadHandle.md). A communication thread can be created by calling APIs such as [HcclThreadAcquire](./HcclThreadAcquire.md), [HcclThreadAcquireWithConfig](./HcclThreadAcquireWithConfig.md), and [HcclThreadAcquireWithStream](./HcclThreadAcquireWithStream.md). |
| resType | Input | Underlying resource type (such as stream).<br>For the definition of the ThreadResType type, see [ThreadResType](../../datatype_definition/ThreadResType.md). |
| infoLen | Input | Size of the target resource information. |
| info | Output | Buffer for outputting resource information. The return type is the corresponding resource type obtained. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

This API supports only obtaining the underlying stream resource of a communication thread (type: [ThreadResTypeStream](../../datatype_definition/ThreadResTypeStream.md)).

- **comm** must be a valid communicator handle (obtained through the communicator creation API and kept valid during the call). Do not pass a null pointer, wild pointer, or destroyed handle.

## Example

```c
// Communicator handle.
HcclComm comm;
ThreadHandle thread;          // Thread handle created by HcclThreadAcquire.
ThreadResTypeStream stream;   // The info buffer must be aligned to the resource type and writable.
uint32_t size = sizeof(ThreadResTypeStream);  // Must equal the target type size.
HcclResult ret = HcclThreadResGetInfo(comm, thread, THREAD_RES_TYPE_STREAM, size, &stream);
if (ret != HCCL_SUCCESS) {
    // Error handling.
}
// Use the stream resource.
// ...
```
