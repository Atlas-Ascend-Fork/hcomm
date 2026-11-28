# HcommMemHostMap2Dev

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

Maps Host DDR memory to a virtual address (dVA) accessible from the device, so that the device can directly read and write the host memory.

## Prototype

```c
HcommResult HcommMemHostMap2Dev(const CommMem* hostMem, CommMem* devMem, HcommMemMapType mapType)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| hostMem | Input | Host memory description. `type` must be `COMM_MEM_TYPE_HOST`, `addr` is the host virtual address (hVA), and `size` is the memory size in bytes and must not be 0. |
| devMem | Output | Mapping result. On success, `addr` is the dVA accessible from the device, `type` is `COMM_MEM_TYPE_DEVICE`, and `size` equals `hostMem->size`. |
| mapType | Input | Host memory mapping type. For the values, see `HcommMemMapType`:<br>`HCOMM_MEM_MAP_DEFAULT`: maps and registers the host memory as a device-accessible memory address (default).<br>`HCOMM_MEM_MAP_PINNED`: converts host non-page-locked memory allocated by C/C++ standard library functions (for example, `malloc`) into page-locked memory before mapping. Only the non-ATU (ACL) path supports this value; the ATU path returns `HCCL_E_NOT_SUPPORT`. |

## Return Value

[HcommResult](../comm_opdev/datatype_definition/HcommResult.md): `HCCL_SUCCESS` indicates success; other values indicate failure.

- `HCCL_E_PTR`: `hostMem` or `devMem` is a null pointer.
- `HCCL_E_PARA`: `hostMem` is invalid (`type` is not `COMM_MEM_TYPE_HOST`, `addr` is null, or `size` is 0), or `mapType` is invalid.
- `HCCL_E_RUNTIME`: Failed to obtain the current device, obtain the device type, obtain the H2D protocol, call the host memory registration API, or call `aclrtHostGetDevicePointer`.
- `HCCL_E_NOT_SUPPORT`: `mapType` is `HCOMM_MEM_MAP_PINNED` but the ATU path is used, or the current runtime does not support `rtHostRegisterByPciAtu`/`rtHostUnRegisterByPciAtu`, so host memory registration cannot be completed when H2D is PCIe and the product is an Ascend 950PR&950DT series product.

## Constraints

- Call `aclrtSetDevice` before this API. The API obtains the device logic ID through `aclrtGetDevice`.
- `HcommMemHostMap2Dev` and `HcommMemHostUnmapFromDev` must be called in the same device context (the same current device).
- `hostMem->addr` must be 4 KB page aligned.
- When the OS kernel version is 5.10 or earlier, `aclrtMallocHost` must be used to allocate page-locked host memory; otherwise the mapping fails. When the OS kernel version is later than 5.10, you can use `aclrtMallocHost` to allocate page-locked memory or use `malloc` to allocate non-page-locked memory.
- `mapType` is converted by HCOMM into the underlying registration flag: `rtHostRegisterByPciAtu` is used when H2D is PCIe and the product is an Ascend 950PR&950DT series product, and only `HCOMM_MEM_MAP_DEFAULT` is supported; other products use `aclrtHostRegisterV2`, which supports `HCOMM_MEM_MAP_DEFAULT` and `HCOMM_MEM_MAP_PINNED`. In non-ATU scenarios, the dVA is obtained through `aclrtHostGetDevicePointer` after registration.
- When the runtime does not support querying the H2D connection type (for example, an earlier runtime version), the API treats the link as non-PCIe and falls back to `aclrtHostRegisterV2`.
- The mapped dVA is used by the device to access host memory and cannot be used for memory copy operations.
- The mapped dVA is valid only within the process and device context that performed the mapping. Do not pass it across processes or devices.
- The caller must keep the host memory valid and must not release or reuse it while it is mapped. Call [HcommMemHostUnmapFromDev](./HcommMemHostUnmapFromDev.md) when the mapping is no longer needed.
- `devMem` is not modified when the API fails.

## Example

```c
// Return value check macros
#define HCOMMCHECK(cmd) do { HcommResult ret = (cmd); if (ret != HCCL_SUCCESS) { return ret; } } while (0)
#define ACLCHECK(cmd) do { aclError ret = (cmd); if (ret != ACL_SUCCESS) { return (HcclResult)ret; } } while (0)

// Allocate host memory and align it up to 4 KB (aclrtMallocHost guarantees only 64-byte alignment)
size_t memSize = 4096;
void *rawHostPtr = nullptr;
ACLCHECK(aclrtMallocHost(&rawHostPtr, memSize + 4096));
uintptr_t aligned = (reinterpret_cast<uintptr_t>(rawHostPtr) + 4095) & ~uintptr_t(4095);
void *hostPtr = reinterpret_cast<void *>(aligned);

// Map the host memory to a dVA accessible from the device
CommMem hostMem;
hostMem.type = COMM_MEM_TYPE_HOST;
hostMem.addr = hostPtr;
hostMem.size = memSize;

CommMem devMem;
devMem.type = COMM_MEM_TYPE_INVALID;
devMem.addr = nullptr;
devMem.size = 0;
HCOMMCHECK(HcommMemHostMap2Dev(&hostMem, &devMem, HCOMM_MEM_MAP_DEFAULT));

// After successful mapping, devMem.addr is the dVA accessible from the device

// Unmap when the mapping is no longer needed
HCOMMCHECK(HcommMemHostUnmapFromDev(&hostMem));

ACLCHECK(aclrtFreeHost(rawHostPtr));
```
