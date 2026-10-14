# CallFunc

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:59:33.355Z pushedAt=2026-09-30T06:13:48.712Z -->

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

`ccu::CallFunc<Obj>(args...)` is the call-site API of the FuncBlock mechanism and is used together with [Func](Func.md): on the first call, it synthesizes the FuncBlock segment corresponding to `Obj` and appends a `Call` instruction; subsequent calls to the same `Obj` directly append a `Call` instruction and reuse the existing FuncBlock segment. In this way, a piece of repeatedly used logic generates only one copy of instructions, and all call sites jump to it through `Call` instructions, saving SRAM.

For details about the FuncBlock definition (`ccu::Func`), see [Func](Func.md).

## Template Function

```cpp
namespace AscendC {
namespace ccu {

// Obj must be a ccu::Func object with global or static storage (required by C++ NTTP).
// Each parameter in Args must be ccu::Variable.
template <Func &Obj, typename... Args>
CcuResult CallFunc(Args... args);

} // namespace ccu
} // namespace AscendC
```

## Parameters

| Parameter | Description |
| --- | --- |
| Obj | Template non-type parameter (NTTP), which must be a reference to a `ccu::Func` object with `global` or `static` storage. |
| args... | Argument list. The number of arguments must be the same as the number of lambda parameters of `Obj`, and each argument type must be `ccu::Variable`. |

## Return Value

[CcuResult](../../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success, and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | The operation is successful. |
| `CCU_E_PARA` | The number of arguments is inconsistent with `Obj.NumIn()` (this check is performed first and returns directly). |
| `CCU_E_PTR` | No kernel is currently being registered (the API is called outside the kernel registration phase). |

> [!NOTE] Note
> Other failure cases (such as being called inside a hardware loop body and nested **CallFunc**) throw an exception (carrying an error code) instead of being reported through the return value. For details, see [Exceptions](#exceptions).

## Exceptions

When `CallFunc` fails, an exception is thrown (carrying a [CcuResult](../../../datatype_definition/CcuResult.md) error code). Common scenarios:

| Scenario | Error Code |
| --- | --- |
| `CallFunc` is called inside a hardware loop body, or `CallFunc` is nested inside the lambda body of `Func` (only top-level calls are allowed) | `CCU_E_INTERNAL` |
| Other underlying failures during **FuncBlock** synthesis (for example, failure to allocate resources for the formal parameter `Variable`) | Corresponding error code |

> [!NOTE] Note
> The exception thrown by `CallFunc` does not propagate beyond the user kernel: the kernel is executed during the `HcommCcuKernelRegister` phase, and this entry point applies a unified `try/catch` to the kernel body. After the exception is caught, it is normalized along the standard exception path, and the registration API ultimately returns `CCU_E_INTERNAL` (whose enumeration value, together with `CCU_SUCCESS`/`CCU_E_PARA`, belongs to `CcuResult` and equals `HCCL_E_INTERNAL`). Therefore, the caller does not need to perform its own `try/catch`; it only needs to check the return value of `HcommCcuKernelRegister` to detect the failures described above.

## Constraints

- The `Obj` of `CallFunc<Obj>` must be a `ccu::Func` object with `global` or `static` storage (it cannot be a function-local variable). This is a C++ language requirement for the reference-type NTTP `template <Func& Obj, ...>`, and a compile-time error is reported if violated. For details on the linkage constraints of the `Func` object, see the constraints in [Func](Func.md).
- `CallFunc<F>` cannot be called inside a hardware loop body; otherwise, an exception is thrown (error code `CCU_E_INTERNAL`).
- `CallFunc<F>` cannot be nested: the lambda of a `Func` cannot call another `CallFunc`. In this case, the inner `CallFunc` detects at the very first `FuncBlockLookup` stage that it is currently inside a `Func` body (`inFuncBody_`) and throws an exception (error code `CCU_E_INTERNAL`); this exception directly propagates through the outer `CallFunc` (the outer `RunBody` is not wrapped in try/catch), so the outer `FuncBlockEnd` is not executed.
- On the first `CallFunc`, a FuncBlock is synthesized. Subsequent `CallFunc` calls on the same `Obj` only append a `Call` instruction and reuse the existing FuncBlock. Different `ccu::Func` objects synthesize independent FuncBlocks even if their lambda contents are identical (the key is `&Obj`, not the hash of the lambda content).

## Example

```cpp
using namespace AscendC::ccu;

// Declare as static to meet the CallFunc NTTP requirement.
static Func MyAdd([](Variable x) {
    Variable tmp;
    tmp = x + x;   // Double x.
});

CcuResult MyKernel(CcuKernelArg arg) {
    Variable a, b;
    a = 2;
    b = 5;

    // First call: synthesize the FuncBlock segment and append a call instruction.
    CcuResult ret = CallFunc<MyAdd>(a);
    if (ret != CCU_SUCCESS) { return ret; }

    // Second call: append only a call instruction and reuse the existing FuncBlock.
    return CallFunc<MyAdd>(b);
}
```
