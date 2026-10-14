# HcclEngineCtxCopy

<!-- md-trans-meta sourceCommit=306b11da159831ad0e3dbb73fec73f1ad849a0a2 translatedAt=2026-10-08T11:48:36.210Z pushedAt=2026-10-12T06:12:35.598Z -->

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

Specifies the communicator, communication engine, and communication engine context tag to copy host-side memory data to the corresponding communication engine context.

## Function Prototype

```c
HcclResult HcclEngineCtxCopy(HcclComm comm, CommEngine engine, const char *ctxTag, const void *srcCtx, uint64_t size, uint64_t dstCtxOffset)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator handle.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| engine | Input | Communication engine type. |
| ctxTag | Input | Communication engine context tag (maximum character length: **HCCL_RES_TAG_MAX_LEN**). |
| srcCtx | Input | Source memory address. |
| size | Input | Source memory size. |
| dstCtxOffset | Input | Address offset in the communication engine context to which data is copied. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

- **comm** must be a valid communicator handle (obtained through the communicator creation API and kept valid during the call). Do not pass a null pointer, wild pointer, or destroyed handle.

## Example

```c
// Communicator handle.
HcclComm comm;
CommEngine engine = COMM_ENGINE_AICPU_TS;
const char *ctxTag = "ctxTag";
void *resCtx = nullptr;  // Valid source ctx, which must point to allocated memory.
uint64_t size = 16; // Actual size to be copied.
uint64_t dstCtxOffset = 0; // When copying all data, pass 0 as the offset.
HcclResult ret = HcclEngineCtxCopy(comm, engine, ctxTag, resCtx, size, dstCtxOffset);
if (ret != HCCL_SUCCESS) {
    // Error handling
}
```
