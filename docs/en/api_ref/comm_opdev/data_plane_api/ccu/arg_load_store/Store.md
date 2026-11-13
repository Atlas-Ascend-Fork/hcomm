# Store

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:50:03.562Z pushedAt=2026-10-08T09:29:37.498Z -->

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

Writes the `uint64_t` value of one or more variables to an HBM address in a CCU kernel.

It is the reverse operation of `Load` and supports the same two address types, which are automatically selected based on the type of the first parameter:

| Overload Group | Address Type | When the Address Is Determined |
| --- | --- | --- |
| Overload 1/2 | Immediate address (`uint64_t`) | Determined in the registration phase |
| Overload 3/4 | Variable address (`Variable`) | Read from a hardware register at runtime |

Within each group, the overloads are further divided by the number of stored elements into a single variable (num=1) and a batch `Array<Variable>` (num>1).

## Function Prototype

```cpp
namespace hcomm {
namespace ccu {
// Overload 1: Store one variable to an immediate address.
CcuResult Store(uint64_t addr, Variable v);
// Overload 2: Store num variables in batch to an immediate address.
CcuResult Store(uint64_t addr, Array<Variable>& vArr, uint32_t num);
// Overload 3: Store 1 variable to a variable address (indirect addressing)
CcuResult Store(Variable addrVar, Variable v);
// Overload 4: Store num variables in batch to a variable address (indirect addressing).
CcuResult Store(Variable addrVar, Array<Variable>& vArr, uint32_t num);
} // namespace ccu
} // namespace hcomm
```

## Parameters

### Overload 1/2 Parameters (Immediate Address)

| Parameter | Input/Output | Description |
| --- | --- | --- |
| addr | Input | Immediate HBM destination address (`uint64_t`). It must be a physical address accessible to the CCU or a tokenized VA, determined in the registration phase and immutable at runtime. |
| v | Input | Source variable (overload 1, num=1). At runtime, the value of this variable is written to `HBM[addr]`. |
| vArr | Input | First element of the source variable array (overload 2, num>1). It must be allocated through `ccu::Array<Variable>` to ensure physical contiguity. |
| num | Input | Number of `uint64_t` elements to store (overload 2), which must be greater than 0. When `num>1`, `vArr[0]`, `vArr[1]`, ..., `vArr[num-1]` are written to `HBM[addr]`, `HBM[addr+8]`, ..., `HBM[addr+(num-1)*8]`, respectively. |

### Overload 3/4 Parameters (Variable Address, Indirect Addressing)

| Parameter | Input/Output | Description |
| --- | --- | --- |
| addrVar | Input | Address variable (`Variable`). At runtime, the value stored in this variable is used as the HBM destination address, and it must have been assigned a valid address value. |
| v | Input | Source variable (overload 3, num=1). |
| vArr | Input | First element of the source variable array (overload 4, num>1). It must be allocated through `ccu::Array<Variable>` to ensure physical contiguity. |
| num | Input | Number of `uint64_t` elements to store (overload 4), which must be greater than 0. The semantics are the same as overload 2, with the address taken from the runtime value of `addrVar`. |

## Return Value

[CcuResult](../../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success, and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | The operation succeeds. |
| `CCU_E_PARA` | Parameter error: `num` is 0; or when `num>1`, the `vArr` elements are not physically contiguous. |
| `CCU_E_PTR` | No kernel is currently in the registration phase (the interface is called outside the kernel registration phase). |
| `CCU_E_NOT_FOUND` | The passed `v`/`vArr` handle (or the adjacent handle of `vArr[1..num-1]`) is not registered in the current kernel; or `num` is greater than the actual length of `vArr`, causing access to a non-existent adjacent handle. |

## Constraints

- `addr` must be a physical address accessible to the CCU or a tokenized VA. Directly passing a non-tokenized process virtual address will trigger a driver error.
- `num` must be greater than 0. Passing 0 returns `CCU_E_PARA`.
- When `num>1` (overloads 2/4), `vArr` must point to a physically contiguous array of `Variable` objects, which must be allocated through `ccu::Array<Variable>`. Declaring multiple `Variable` objects separately does not guarantee physical contiguity. If this requirement is violated, `CCU_E_PARA` is returned at the `Store(...)` call site.
- `num` must be less than or equal to the actual length of `vArr`. This API does not check `num <= vArr.size()`. If `num` is out of range, adjacent handles that do not belong to `vArr` may be accessed, or `CCU_E_NOT_FOUND` is returned. Ensure that `num` does not exceed the allocated length.
- `addrVar` (overloads 3/4) must have been assigned a valid address value (through `LoadArg`, immediate assignment, or arithmetic operations) before this API is called.
- The immediate address (overloads 1/2) is determined during the registration phase and cannot be changed at runtime. Use overloads 3/4 when a runtime dynamic address is required.

## Example

```cpp
using namespace hcomm::ccu;

// Scenario 1: Store one variable to an immediate address (overload 1).
CcuResult MyKernel(CcuKernelArg arg) {
    Variable result;
    // ... Compute result ...
    Store(0x20000000ULL, result);
    return CCU_SUCCESS;
}

// Scenario 2: Batch-store 4 variables to an immediate address (overload 2, num > 1)
CcuResult MyKernel2(CcuKernelArg arg) {
    Array<Variable> vArr(4);    // Four physically contiguous variables.
    // ... Compute vArr ...
    Store(0x80000000ULL, vArr, 4);
    return CCU_SUCCESS;
}

// Scenario 3: Store the variable to the address specified by a variable (overload 3).
CcuResult MyKernel3(CcuKernelArg arg) {
    Variable dstAddr, result;
    LoadArg(dstAddr, 0);    // Pass the target address from the host.
    // ... Compute result ...
    Store(dstAddr, result); // Write via indirect addressing at runtime.
    return CCU_SUCCESS;
}
```
