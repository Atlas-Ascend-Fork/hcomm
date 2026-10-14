# Variable

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T08:14:47.902Z pushedAt=2026-09-30T08:45:54.784Z -->

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

`ccu::Variable` is a C++ wrapper class for scalar registers within a CCU kernel.

- Allocation on construction: The default constructor automatically allocates one scalar register virtual handle.
- No release on destruction: The destructor does not release hardware resources; the virtual handle becomes invalid after translation completes, and the physical resource is managed and reclaimed uniformly over the lifecycle of the CCU instance.
- Operators are device operations: The assignment and arithmetic operators on `Variable` describe operations executed on the device side (hardware), operating on the corresponding scalar register at runtime, rather than being computed immediately on the host side.

> [!NOTE] Note
> CCU resource allocation adopts a two-phase "virtual first, physical second" model: the `Variable()` construction in the registration phase only produces a virtual handle, and the actual physical scalar register allocation is completed in the `HcommCcuKernelRegister` phase (after the kernel function finishes execution).

## Class Declaration

```cpp
namespace AscendC {
namespace ccu {
class Variable final {
public:
    Variable();                                                   // Allocation on construction.
    // Bind the index-th scalar register reserved on the host side without allocating a new scalar register.
    explicit Variable(CcuVariableHandle varHandle, uint32_t index = 0);
    void operator=(uint64_t immediate) const;                    // Assign an immediate.
    void operator=(const Variable& other) const;                 // Assign between Variable objects.
    void operator+=(const Variable& other) const;               // In-place addition.
    /* Internal expression object. */ operator+(const Variable& that) const;    // Addition (expression template).
    CondExpr operator==(uint64_t immediate);                     // Produce a CondExpr.
    CondExpr operator!=(uint64_t immediate);                     // Produce a CondExpr.
    CcuVariableHandle handle{0};                                 // Virtual handle.
};
} // namespace ccu
} // namespace AscendC
```

## Constructor Description

| Construction Form | Description |
| --- | --- |
| `Variable v;` | Allocates a virtual handle for one scalar register. It can only be called during the kernel registration phase (inside the kernel function body executed by `HcommCcuKernelRegister`). |
| `Variable v(varHandle, index);` | Binds the `index`-th scalar register reserved on the host side through [HcommCcuVariableAlloc](../../../control_plane_api/ccu_resource_mgmt/HcommCcuVariableAlloc.md), without allocating a new scalar register. The default value of `index` is `0`. |

### Constructor for Binding Host-Side Reserved Resources

The `varHandle` in `explicit Variable(CcuVariableHandle varHandle, uint32_t index = 0)` is the **reservation handle** returned by [HcommCcuVariableAlloc](../../../control_plane_api/ccu_resource_mgmt/HcommCcuVariableAlloc.md) on the host side, not the `handle` of another `Variable`. A typical usage is to pass the reservation handle into the kernel through `kernelArgs`, and then bind one of its scalar registers inside the kernel.

Calling this constructor multiple times with the same `index` of the same reservation handle yields a new in-kernel `handle` each time, all pointing to the same physical scalar register.

If construction fails, an exception is thrown, which is uniformly caught by [HcommCcuKernelRegister](../../../control_plane_api/ccu_kernel_launch_execution/HcommCcuKernelRegister.md). `CCU_E_INTERNAL` is returned to the caller and the current registration is aborted.

> [!CAUTION] Caution
> The copy/move constructors only copy the `handle` field and do not allocate a new scalar register. After `Variable v2 = v1;`, the two objects are the same in-kernel handle. If an independent `Variable` is required, use the default constructor or [`Array<Variable>`](Array.md).

## Operator Description

### Assignment Operators

| Expression Syntax | Hardware Semantics |
| --- | --- |
| `v = imm;` (`imm` is `uint64_t`) | `v ← imm`. The immediate is determined during the registration phase and is immutable at runtime. |
| `d = s;` (`s` is a `Variable`) | `d ← s`. A device-side register assignment, not a host-side handle copy. |

### Arithmetic Operator

| Expression Syntax | Hardware Semantics |
| --- | --- |
| `r = a + b;` | `r ← a + b` (a single dual-source addition instruction). `operator+` returns an expression template object, which is consumed by `operator=` to generate one device-side addition without creating a temporary `Variable`. |
| `r += b;` | `r ← r + b`. Semantically identical to `r = r + b`, with no temporary object. |

> [!CAUTION] Caution
> `r = a + b` uses an expression template (internal type) to avoid creating a temporary `Variable` that would consume additional scalar registers. Do not store the result of `a + b` in an ordinary C++ variable; otherwise, the corresponding device-side operation will not be generated.

### Conditional Operator

| Expression Syntax | Return Type | Description |
| --- | --- | --- |
| `n == imm` | `CondExpr` | Produces a condition expression object without generating any device-side operation, intended exclusively for consumption by the `CCU_IF`/`CCU_WHILE`/`CCU_DO...CCU_WHILE()` macros. |
| `n != imm` | `CondExpr` | Same as above. |

> [!CAUTION] Caution
> `CondExpr` can only be consumed by control flow macros and cannot be used as an ordinary C++ Boolean expression (for example, `if (n == 0)`). Placing it in an ordinary `if` statement does not generate any CCU control flow, and the expression is discarded directly.

## Constraints

- A variable can only be constructed during the kernel registration phase.
- The destructor does not release hardware resources. Do not save or compare the `handle` value outside the kernel; the handle becomes invalid once translation is complete.
- Currently, only addition is supported. Subtraction, multiplication, and division are not supported.
- An immediate cannot directly participate in arithmetic (`v + 1` is invalid). It must first be assigned to a variable (`one = 1;`) before participating in an operation (`v = v + one;`).
- The default constructor only allocates a virtual handle (without consuming a physical scalar register), and **construction within the kernel registration phase always succeeds**. If construction occurs outside the registration phase (not within the kernel function body called by `HcommCcuKernelRegister`), the underlying `CcuVariableAlloc` cannot find the current kernel and throws a `CcuException` carrying `CCU_E_PTR`. Insufficient physical scalar register resources are reported by `HcommCcuKernelRegister` as `CCU_E_UNAVAIL`, and are not triggered at construction time.
- Reservation handle construction `Variable v(varHandle, index);` can likewise be called only during the kernel registration phase. It validates the reservation handle and `index`, and throws an exception when the parameters are invalid, so it is no longer guaranteed to always succeed.

## Example

```cpp
using namespace AscendC::ccu;

CcuResult MyKernel(CcuKernelArg arg) {
    Variable n, i, one, step;

    // Assign an immediate (determined during the registration phase).
    n = 100;
    i = 0;
    one = 1;
    step = 8;

    // Assign between variables.
    Variable cursor;
    cursor = i;           // cursor ← i

    // Arithmetic: expression template syntax (r = a + b), which generates only one device addition.
    Variable sum;
    sum = i + step;       // sum ← i + step

    // In-place addition.
    i += one;             // i ← i + one

    // Conditional operation (consumed by the CCU_WHILE macro).
    CCU_WHILE(n != 0) {
        // ...
    }

    return CCU_SUCCESS;
}
```
