# HcommCcuGetMemToken

<!-- md-trans-meta sourceCommit=9a28f2fdc3bd8008a5fa6d178f4a7efe8b2d3eb9 translatedAt=2026-10-08T11:44:57.681Z pushedAt=2026-10-12T06:12:35.589Z -->

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

Converts the virtual address (VA) of a host process into a memory token accessible by CCU hardware. CCU hardware does not directly use process virtual addresses to access on-chip memory. You must first use this API to convert the memory region corresponding to `(srcVa, size)` into a 64-bit token issued by the driver, then pass the token to the kernel function through `kernelArg` or `taskArgs`, and assign it to `LocalAddr.token` or `RemoteAddr.token` inside the kernel.

This API has no dependency on the main lifecycle flow of a CCU instance. It can be called at any time after an instance is created. Typically, a token is obtained and filled into `kernelArg` before calling [HcommCcuKernelRegister](HcommCcuKernelRegister.md).

## Function Prototype

```c
CcuResult HcommCcuGetMemToken(uint64_t srcVa, uint64_t size, uint64_t *tokenInfo)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| srcVa | Input | Starting virtual address of the memory region. It must be a valid address registered with the driver and cannot be 0. |
| size | Input | Length of the memory region, in bytes. It must be greater than 0 and cannot exceed the actual memory size mapped to the virtual address. |
| tokenInfo | Output | Output parameter, a pointer to a single `uint64_t`. On success, the encoded token value is written to it. It cannot be a null pointer. |

## Return Value

[CcuResult](../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | Operation succeeded. |
| `CCU_E_PTR` | Null pointer error. `tokenInfo` is a null pointer. |
| `CCU_E_PARA` | Parameter error. `srcVa` is 0 or `size` is 0. |
| `CCU_E_RUNTIME` | Runtime error. `srcVa` is not allocated, or the `size` parameter is greater than the allocated size. |
| `CCU_E_DRV_*` | Driver-layer error. Currently, 4097 (`CCU_E_DRV_INIT_FAILED`) or 4098 (`CCU_E_DRV_BUSY`) may be returned. The driver-layer error code range is defined as [`CCU_E_DRV_START`=4096, `CCU_E_DRV_END`=4224]. For details, see [CcuResult](../../datatype_definition/CcuResult.md). |

## Constraints

- A token is security-sensitive information. Do not print the token value to host logs or device logs.
- `(srcVa, size)` must fall within the same registered contiguous memory region. Crossing regions or covering unregistered memory regions causes the driver to return an error.
- The valid lifecycle of a token is bound to the registration lifecycle of the corresponding memory. After the memory is deregistered, the token becomes invalid.
- In cross-rank communication scenarios, the local token must be passed to the peer through an out-of-band channel (not the CCU data plane), and the peer end assembles `RemoteAddr.token`. The token value must not be transmitted in plaintext over the CCU data plane.
- This API can be called only on the host side, not inside a kernel function body.

## Example

```c
// srcVa is the start address of the registered on-chip memory, and size is the byte size of the memory region.
uint64_t srcVa = /* Virtual address (VA) of the registered memory. */;
uint64_t size  = /* Byte size of the memory region. */;

uint64_t tokenInfo = 0;
CcuResult ret = HcommCcuGetMemToken(srcVa, size, &tokenInfo);
if (ret != CCU_SUCCESS) {
    printf("HcommCcuGetMemToken failed, ret = %d\n", ret);
    return ret;
}

// Pass tokenInfo to the kernel through kernelArg or taskArgs.
// Inside the kernel, assign it to LocalAddr.token or RemoteAddr.token.
```
