# HcclCommActivateCommMemory

<!-- md-trans-meta sourceCommit=306b11da159831ad0e3dbb73fec73f1ad849a0a2 translatedAt=2026-10-08T11:36:35.928Z pushedAt=2026-10-12T06:12:35.567Z -->

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

Activates the reserved virtual memory. The zero-copy feature can be enabled only when the activated memory is used as the input and output of communication operators.

## Function Prototype

```c
HcclResult HcclCommActivateCommMemory(HcclComm comm, void *virPtr, size_t size, size_t offset, aclrtDrvMemHandle handle, uint64_t flags)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | HCCL communicator. It is recommended to use the largest communicator in the server, that is, the communicator covering the maximum number of devices. |
| virPtr | Input | Virtual memory address to be activated, that is, the to-be-mapped virtual memory address passed in when the user calls the **aclrtMapMem** API to map physical memory to virtual memory. |
| size | Input | Size of the memory to be activated, in bytes. |
| offset | Input | Reserved field.<br>Currently, only **0** is supported. |
| handle | Input | Handle of the allocated physical memory information, that is, the handle of the device physical memory information allocated by the user by calling the **aclrtMallocPhysical** API. |
| flags | Input | Reserved field.<br>Currently, only **0** is supported. |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

- The virtual memory address to be activated must be within the address range set by [HcclCommSetMemoryRange](HcclCommSetMemoryRange.md).
- The virtual memory address must not overlap or intersect with any already activated virtual memory address.
- **comm** must be a valid communicator handle (obtained through the communicator creation API and kept valid during the call). Do not pass a null pointer, a wild pointer, or a destroyed handle.

## Example

```c
// Assume that the virtual memory range has been set through HcclCommSetMemoryRange.
// baseVirPtr is the base address of the virtual memory.

// Allocate physical memory.
aclrtDrvMemHandle handle;
aclrtMallocPhysical(&handle, size, NULL, 0);

// Map the physical memory to the virtual memory.
void *virPtr = baseVirPtr;
aclrtMapMem(virPtr, size, 0, handle, 0);

// Activate the reserved virtual memory.
HcclCommActivateCommMemory(hcclComm, virPtr, size, 0, handle, 0);

// The activated memory can then be used for zero-copy communication.
// ...

// Deactivate the memory.
HcclCommDeactivateCommMemory(hcclComm, virPtr);
```
