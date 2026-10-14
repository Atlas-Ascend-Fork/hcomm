# Load

<!-- md-trans-meta sourceCommit=60a73b342d2c766f787196d1081a922b09ee0fd7 translatedAt=2026-10-08T11:57:40.994Z pushedAt=2026-10-12T06:12:35.622Z -->

## Supported Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: not supported
<!-- end id3 -->
<!-- npu="910" id4 -->
- Atlas training products: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas inference products: Not supported
<!-- end id5 -->

## Description

Reads one or more `uint64_t` values from an on-chip memory address and writes them to a variable within a CCU kernel.

Two address types are supported, and the type is automatically selected based on the type of the first parameter:

| Overload Group | Address Type | When the Address Is Determined |
| --- | --- | --- |
| Overload 1/2 | Immediate address (`uint64_t`) | Determined in the registration phase |
| Overload 3/4 | Variable address (`Variable`) | Read from hardware registers at runtime |

Within each group, the overloads are further divided by the target quantity into a single-variable (num=1) overload and a batch `Array<Variable>` (num>1) overload.

## Function Prototype

```cpp
namespace AscendC {
namespace ccu {
// Overload 1: Load one variable from an immediate address.
CcuResult Load(uint64_t addr, Variable v);
// Overload 2: Load num variables in batch from an immediate address.
CcuResult Load(uint64_t addr, Array<Variable>& vArr, uint32_t num);
// Overload 3: Load a variable from the variable address (indirect addressing).
CcuResult Load(Variable addrVar, Variable v);
// Overload 4: Load num variables in batch from a variable address (indirect addressing).
CcuResult Load(Variable addrVar, Array<Variable>& vArr, uint32_t num);
} // namespace ccu
} // namespace AscendC
```

## Parameters

### Overload 1/2 Parameters (Immediate Address)

| Parameter | Input/Output | Description |
| --- | --- | --- |
| addr | Input | Immediate on-chip memory address (`uint64_t`). It must be a physical address accessible to the CCU or a tokenized VA, determined in the registration phase and immutable at runtime. |
| v | Input/Output | Target variable (overload 1, num=1). At runtime, 8 bytes are read from the on-chip memory address `addr` and written to this variable. |
| vArr | Input/Output | First element of the target variable array (overload 2, num>1). It must be allocated through `ccu::Array<Variable>` to ensure physical contiguity. |
| num | Input | Number of `uint64_t` elements to load (overload 2). It must satisfy `1 ≤ num ≤ vArr.size()`. When `num>1`, 8 bytes are read from the on-chip memory addresses `addr`, `addr+8`, ..., `addr+(num-1)*8` in sequence and written to `vArr[0], vArr[1], ..., vArr[num-1]`, respectively. |

### Overload 3/4 Parameters (Variable Address, Indirect Addressing)

| Parameter | Input/Output | Description |
| --- | --- | --- |
| addrVar | Input | Address variable (`Variable`). At runtime, the value stored in this variable is used as the on-chip memory address, and it must have been assigned a valid address value. |
| v | Input/Output | Target variable (overload 3, num=1). |
| vArr | Input/Output | First element of the target variable array (overload 4, num>1). It must be allocated through `ccu::Array<Variable>` to ensure physical contiguity. |
| num | Input | Number of `uint64_t` elements to load (overload 4). Must satisfy `1 ≤ num ≤ vArr.size()`. The semantics are the same as overload 2, with the address taken from the runtime value of `addrVar`. |

## Return Value

[CcuResult](../../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success, and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | The operation succeeds. |
| `CCU_E_PARA` | Parameter error: `num` is 0 or greater than `vArr.size()`; or when `num>1`, the elements of `vArr` are not physically contiguous. |
| `CCU_E_PTR` | No kernel is currently in registration (the interface is called outside the kernel registration phase). |
| `CCU_E_NOT_FOUND` | The passed `v`/`vArr` handle (or the adjacent handle of `vArr[1..num-1]`) is not registered in the current kernel. |

## Constraints

- `addr` must be a physical address accessible to the CCU or a tokenized VA. Directly passing a non-tokenized process virtual address will trigger a driver error.
- `num` must satisfy `1 ≤ num ≤ vArr.size()`. This API performs this validation before accessing `vArr[0]`; passing 0 or a value exceeding `vArr.size()` returns `CCU_E_PARA` without reaching the underlying API. Passing any `num` to an empty `Array` (where `vArr.size()` is 0) also returns `CCU_E_PARA`.
- When `num>1` (overloads 2/4), `vArr` must point to a physically contiguous variable array, which must be allocated through `ccu::Array<Variable>`. Declaring multiple `Variable` objects separately does not guarantee physical contiguity. If this requirement is violated, `CCU_E_PARA` is returned at the `Load(...)` call site.
- `addrVar` (overloads 3/4) must have been assigned a valid address value (through `LoadArg`, immediate assignment, or arithmetic operations) before this API is called.
- The immediate address (overloads 1/2) is determined during the registration phase and cannot be changed at runtime. Use overloads 3/4 when a runtime dynamic address is required.

## Example

```cpp
using namespace AscendC::ccu;

// Scenario 1: Load one variable from an immediate address (overload 1).
CcuResult MyKernel(CcuKernelArg arg) {
    Variable v;
    Load(0x10000000ULL, v);
    return CCU_SUCCESS;
}

// Scenario 2: Load 4 variables in batch from an immediate address (overload 2, num>1).
CcuResult MyKernel2(CcuKernelArg arg) {
    Array<Variable> vArr(4);    // Four physically contiguous variables.
    Load(0x80000000ULL, vArr, 4);
    return CCU_SUCCESS;
}

// Scenario 3: Indirect load from a variable address (overload 3, the address is computed in the previous step).
CcuResult MyKernel3(CcuKernelArg arg) {
    Variable srcAddr, v;
    LoadArg(srcAddr, 0);    // Address passed by the host.
    Load(srcAddr, v);       // Indirect addressing at runtime.
    return CCU_SUCCESS;
}
```
