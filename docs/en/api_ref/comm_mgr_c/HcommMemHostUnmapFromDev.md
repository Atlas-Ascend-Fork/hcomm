# HcommMemHostUnmapFromDev

## Product Support

<!-- npu="950" id1 -->
- Ascend 950PR&950DT series products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 series products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 series products: Supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- Atlas inference series products: Not supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas training series products: Not supported
<!-- end id5 -->

## Description

Unmaps the mapping created by `HcommMemHostMap2Dev` and releases the device virtual address (dVA) that corresponds to the host DDR memory.

## Prototype

```c
HcommResult HcommMemHostUnmapFromDev(const CommMem* hostMem)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| hostMem | Input | Host memory description. It must be the same as the input of `HcommMemHostMap2Dev`. `type` must be `COMM_MEM_TYPE_HOST` and `addr` is the host virtual address (hVA). |

## Return Value

[HcommResult](../comm_opdev/datatype_definition/HcommResult.md): `HCCL_SUCCESS` indicates success; other values indicate failure.

- `HCCL_E_PTR`: `hostMem` is a null pointer.
- `HCCL_E_PARA`: `hostMem` is invalid (the `type` is not `COMM_MEM_TYPE_HOST` or `addr` is null).
- `HCCL_E_RUNTIME`: Failed to obtain the current device, obtain the device type, obtain the host-to-device (H2D) protocol, or call the host memory unregistration API.
- `HCCL_E_NOT_SUPPORT`: The current runtime does not support `rtHostUnRegisterByPciAtu`, so host memory unregistration cannot be completed when H2D is PCIe and the product is an Ascend 950PR&950DT series product.

## Constraints

- Call `aclrtSetDevice` before this API. The API obtains the device logic ID through `aclrtGetDevice`.
- `HcommMemHostMap2Dev` and `HcommMemHostUnmapFromDev` must be called in the same device context (the same current device).
- This API must be called in pairs with `HcommMemHostMap2Dev`, and `hostMem` must be the same as the input of the mapping.
- After the API succeeds, the mapped dVA becomes invalid and cannot be accessed by the device.
- Unregistering the same mapping twice or unregistering an unmapped address causes the runtime to return an error.
- `rtHostUnRegisterByPciAtu` is used when H2D is PCIe and the product is an Ascend 950PR&950DT series product; other products use the ACL host memory unregistration API (`aclrtHostUnregister`). Both take the host address (hVA) as the input.
- When the runtime does not support querying the H2D connection type (for example, an earlier runtime version), the API treats the link as non-PCIe and falls back to `aclrtHostUnregister`.

## Example

```c
// Return value check macros
#define HCOMMCHECK(cmd) do { HcommResult ret = (cmd); if (ret != HCCL_SUCCESS) { return ret; } } while (0)
#define ACLCHECK(cmd) do { aclError ret = (cmd); if (ret != ACL_SUCCESS) { return (HcclResult)ret; } } while (0)

size_t memSize = 4096;
void *rawHostPtr = nullptr;
ACLCHECK(aclrtMallocHost(&rawHostPtr, memSize + 4096));
uintptr_t aligned = (reinterpret_cast<uintptr_t>(rawHostPtr) + 4095) & ~uintptr_t(4095);
void *hostPtr = reinterpret_cast<void *>(aligned);

CommMem hostMem;
hostMem.type = COMM_MEM_TYPE_HOST;
hostMem.addr = hostPtr;
hostMem.size = memSize;

CommMem devMem;
devMem.type = COMM_MEM_TYPE_INVALID;
devMem.addr = nullptr;
devMem.size = 0;
HCOMMCHECK(HcommMemHostMap2Dev(&hostMem, &devMem, HCOMM_MEM_MAP_DEFAULT));

// Unmap when the mapping is no longer needed
HCOMMCHECK(HcommMemHostUnmapFromDev(&hostMem));

ACLCHECK(aclrtFreeHost(rawHostPtr));
```
