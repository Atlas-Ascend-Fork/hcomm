# HcclCommMemReg

<!-- md-trans-meta sourceCommit=306b11da159831ad0e3dbb73fec73f1ad849a0a2 translatedAt=2026-10-08T11:47:35.632Z pushedAt=2026-10-12T06:12:35.595Z -->

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

Registers the allocated memory with a communicator and obtains the corresponding registration handle.

## Function Prototype

```c
HcclResult HcclCommMemReg(HcclComm comm, const char *memTag, const CommMem *mem, HcclMemHandle *memHandle)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator handle.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| memTag | Input | Memory string tag. The maximum character length is **HCCL_RES_TAG_MAX_LEN**.<br>const uint32_t HCCL_RES_TAG_MAX_LEN = 255; |
| mem | Input | Memory information. For details about the CommMem type, see [CommMem](../../datatype_definition/CommMem.md). |
| memHandle | Output | Memory handle.<br>The HcclMemHandle type is defined as follows:<br>typedef void *HcclMemHandle; |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

- Within a communicator, only one block of memory can be registered for the same **memTag**.
- Within a communicator, registering the same **memTag** repeatedly returns the **HCCL_E_PARA** error and does not reuse the existing registered memory handle.
- Within a communicator, different **memTags** can be mapped to overlapping or identical memory regions.
- **comm** must be a valid communicator handle (obtained through the communicator creation API and kept valid during the call). Do not pass a null pointer, wild pointer, or destroyed handle.

## Example

```c
HcclComm comm; // Communicator handle. Omitted here.
const char* memTag = "memTag"; // Memory tag.
void *deviceBuffer = nullptr; // Address of the allocated device memory (allocated through APIs such as aclrtMalloc).
CommMem memInfo; // Memory information.
memInfo.addr = deviceBuffer; // Configure the address of the allocated memory.
memInfo.size = 1024; // Configure the size of the allocated memory.
memInfo.type = COMM_MEM_TYPE_DEVICE; // Configure the type of the allocated memory.
HcclMemHandle memHandle; // Memory handle.
HcclCommMemReg(comm, memTag, &memInfo, &memHandle);
```
