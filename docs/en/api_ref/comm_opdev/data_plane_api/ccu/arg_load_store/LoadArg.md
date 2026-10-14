# LoadArg

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:49:17.306Z pushedAt=2026-09-30T03:30:34.607Z -->

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

During the CCU kernel registration phase, declares that "`taskArgs[argId]` of `HcommCcuKernelLaunch` is loaded into the specified variable at runtime". At registration, only this declaration is recorded; the actual loading is performed by the hardware at runtime.

This API is the core mechanism that makes a single registered kernel configurable at runtime: only template logic is written at registration, and `LoadArg` declares which variable values are injected by the host on each launch, thereby avoiding repeated registration.

> [!NOTE] Note
> `LoadArg` does not perform the actual loading during the registration phase. The actual loading occurs after `HcommCcuKernelLaunch` and before the CCU kernel starts execution, where the hardware automatically reads the value from `taskArgs[argId]` and writes it to the hardware register corresponding to the variable.

## Function Prototype

```cpp
namespace AscendC {
namespace ccu {
CcuResult LoadArg(Variable v, uint32_t argId);
} // namespace ccu
} // namespace AscendC
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| v | Input | Target variable object. At runtime, the hardware writes `taskArgs[argId]` (`uint64_t`) to the hardware register corresponding to this variable. |
| argId | Input | Index of the `taskArgs` array (starting from 0). Within the same kernel, the `argId` of all `LoadArg` calls must be globally unique and numbered consecutively from 0. |

## Return Value

[CcuResult](../../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success, and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | The operation is successful. |
| `CCU_E_PTR` | No kernel is currently being registered (the API is called outside the kernel registration phase). |
| `CCU_E_NOT_FOUND` | The `v` handle passed in is not registered in the current kernel. |

> [!NOTE] Note
> This API does not return `CCU_E_PARA`. All constraint errors related to argId (duplication, skipped numbers, mismatch with `argNum`) are not returned by this API. Instead, they are uniformly validated during the `HcommCcuKernelLaunch` phase and `CCU_E_INTERNAL` is returned after validation. For details, see the next section "Constraints".

## Constraints

> [!CAUTION] Caution
> The total number of distinct `argId` values used by all `LoadArg` calls within the same kernel must be exactly equal to the `argNum` parameter of `HcommCcuKernelLaunch`, and `argId` must cover all values in `0..argNum-1`. Both are uniformly validated during the `HcommCcuKernelLaunch` phase, and `CCU_E_INTERNAL` is returned on violation.

- `argId` must be numbered consecutively starting from 0 (0, 1, 2, ...), and must not skip or be out of order. This is strictly validated during the launch phase.
- The `argId` values of different `LoadArg` calls within the same kernel must not be duplicated. This API does not intercept duplicates, but duplication reduces the total number of valid `argId` values below `argNum`, causing `CCU_E_INTERNAL` to be reported during the `HcommCcuKernelLaunch` phase. Binding two variables to the same `argId` is a dangerous error and should be avoided.
- The unit of `argNum` is the number of `uint64_t` elements, not bytes. For example, if there are 3 `LoadArg` calls (with argId 0, 1, and 2), `argNum` of `HcommCcuKernelLaunch` must be `3`, and `taskArgs` must contain at least 3 elements.
- `LoadArg` does not take effect at the call site: during translation, all `LoadArg` calls are hoisted to the very beginning of the kernel and executed first, and they only set the "initial value upon entering the body" for `v`. Therefore, if the body later assigns an immediate value (`v = 1024;`) or uses `Load` to assign a value to the same `v`, these assignments are placed after `LoadArg` and will overwrite the injected initial value. When mixing the same `v`, you must first read out the injected value and then reassign it; otherwise the injected value is overwritten before being used, and `LoadArg` effectively has no effect (operating on different variables, such as `LoadArg(addrVar, 0); Load(addrVar, v);`, is not affected).
- The `argId` of `LoadArg` must be less than 13, that is, 0, 1, ..., 12, because the CCU can carry at most 13 parameters when dispatching an operator.

## Example

```cpp
using namespace AscendC::ccu;

// Scenario: Declare at registration that the loop count n and the start offset are passed in by the host at launch.
CcuResult MyKernel(CcuKernelArg arg) {
    Variable n, offset;

    LoadArg(n, 0);        // At runtime, taskArgs[0] -> n.
    LoadArg(offset, 1);   // At runtime, taskArgs[1] -> offset.

    // Use n and offset for subsequent loop and address calculation.
    // ...
    return CCU_SUCCESS;
}

// Corresponding Launch call on the host side (illustrative):
// uint64_t taskArgs[] = {100, 4096};   // n=100 iterations, offset=4096 bytes.
// HcommCcuKernelLaunch(..., taskArgs, /*argNum=*/2);
```
