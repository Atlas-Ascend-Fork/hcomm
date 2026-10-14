# HcclCommUnsetMemoryRange

<!-- md-trans-meta sourceCommit=306b11da159831ad0e3dbb73fec73f1ad849a0a2 translatedAt=2026-10-08T11:40:15.669Z pushedAt=2026-10-12T06:12:35.576Z -->

> [!NOTE] Note
> This API is for trial use and may be changed later. It cannot be used in production environments.

## Supported Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Not supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Not supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- Atlas inference products: Not supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas training products: Not supported
<!-- end id5 -->

## Description

Notifies the HCCL communicator to stop using the reserved virtual memory.

## Function Prototype

```c
HcclResult HcclCommUnsetMemoryRange(HcclComm comm, void *baseVirPtr)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | HCCL communicator. It is recommended to use the largest communicator in the server, that is, the communicator covering the maximum number of devices. |
| baseVirPtr | Input | Base address of the reserved virtual memory.<br>The specified base address must have been successfully set by [HcclCommSetMemoryRange](HcclCommSetMemoryRange.md); otherwise, an error is reported. |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

If activated memory still exists in this virtual address space, this API fails to execute.

- **comm** must be a valid communicator handle (obtained through the communicator creation API and kept valid during the call). Do not pass a null pointer, a wild pointer, or a destroyed handle.

## Example

```c
// Assume that the virtual memory range has been set by HcclCommSetMemoryRange.
// baseVirPtr is the base address of the virtual memory passed to HcclCommSetMemoryRange.

// Ensure that all activated memory within this address range has been deactivated.
// HcclCommDeactivateCommMemory(hcclComm, activatedVirPtr);

// Cancel the reserved virtual memory.
HcclCommUnsetMemoryRange(hcclComm, baseVirPtr);
```
