# HcclGetHcclBufferCleared

<!-- md-trans-meta sourceCommit=306b11da159831ad0e3dbb73fec73f1ad849a0a2 translatedAt=2026-10-08T11:50:46.336Z pushedAt=2026-10-12T06:12:35.601Z -->

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

Obtains the HCCL communication memory of the local rank in a communicator. The obtained communication memory has been cleared. On the first call, the memory is initialized and allocated on the device side. Subsequent calls reuse the allocated memory without re-initialization.

> [!NOTE] Note
> The returned HCCL communication memory is managed internally by HCOMM. The caller must not release it.

## Function Prototype

```c
HcclResult HcclGetHcclBufferCleared(HcclComm comm, void **buffer, uint64_t *size)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| buffer | Output | HCCL communication memory address. |
| size | Output | Size of the HCCL communication memory. The memory size is twice the value configured during communicator initialization or the value configured by the **HCCL_BUFFSIZE** environment variable, and defaults to 400 MB. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

If the communicator contains only one rank, no HCCL communication memory is allocated, and the output parameter `buffer` is a null pointer with `size` set to 0.

- **comm** must be a valid communicator handle (obtained through the communicator creation API and kept valid during the call). Do not pass a null pointer, wild pointer, or destroyed handle.

## Example

```c
HcclComm comm; // Initialized.
void *hcclBuffer = nullptr;
uint64_t hcclBufferSize = 1 * 1024;   // 1KB
HcclResult result = HcclGetHcclBufferCleared(comm, &hcclBuffer, &hcclBufferSize);
```
