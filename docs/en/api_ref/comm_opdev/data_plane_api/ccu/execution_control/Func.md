# Func

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T08:02:13.546Z pushedAt=2026-09-30T06:26:37.510Z -->

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

`ccu::Func` is used to encapsulate a lambda into a reusable FuncBlock definition. Together with the call-site API [CallFunc](CallFunc.md), it forms the FuncBlock mechanism within a CCU kernel, which generates only one copy of instructions for a piece of repeatedly used logic inside the kernel. All call sites jump to it through the `Call` instruction, saving SRAM.

The lambda of `ccu::Func` takes zero or more `ccu::Variable` parameters and returns `void`. A defined `Func` object is called within the kernel through [CallFunc](CallFunc.md).

## Class Definition

```cpp
namespace AscendC {
namespace ccu {

class Func {
public:
    // The lambda must return void, and all parameters must be ccu::Variable.
    template <typename Lambda>
    explicit Func(Lambda body);

    Func(const Func &) = delete;             // Not copyable.
    Func &operator=(const Func &) = delete;  // Not assignable.
};

} // namespace ccu
} // namespace AscendC
```

## Parameters

| Parameter | Description |
| --- | --- |
| body | Lambda expression whose parameters are zero or more `ccu::Variable` objects and whose return type is `void`. The number of parameters is deduced automatically from the lambda signature at compilation time. |

## Constraints

- A `ccu::Func` object must have "linkage": it can only be defined at namespace scope (including `static`) or as a static data member of a class, and cannot be a function-local variable (not even with `static`). This is a C++ language requirement for reference-type NTTPs of the form `template <Func& Obj, ...>`; a violation causes a compile-time error directly, unrelated to runtime. [CallFunc](CallFunc.md) also uses `&Obj` internally as an identifier to reuse the same FuncBlock, which is a side benefit of this constraint, not the cause of it.
- `ccu::Func` is non-copyable (the copy constructor and assignment operator are `= delete`), so each Func object has a unique identity.
- The lambda parameter type of `ccu::Func` must be passable from `ccu::Variable&`. In practice it can be written as one of `ccu::Variable` / `ccu::Variable&` / `const ccu::Variable&`; other types (such as `int` or `ccu::Address`) cause a compile-time error due to template substitution failure (note: currently no `static_assert` is applied to the parameter type, so the error appears as a deep template diagnostic and is less readable than the one for the return value type).
- The lambda of `ccu::Func` must return `void`; otherwise a compile-time error is reported (with a readable diagnostic provided by an explicit `static_assert` in the implementation).
- Different `ccu::Func` objects synthesize independent FuncBlocks even if their lambda bodies are identical (the key is `&Obj`, not a hash of the lambda body).

## Example

```cpp
using namespace AscendC::ccu;

// Declare it as static to meet the CallFunc NTTP requirement.
static Func MyAdd([](Variable x) {
    Variable tmp;
    tmp = x + x;   // Double x.
});

CcuResult MyKernel(CcuKernelArg arg) {
    Variable a, b;
    a = 2;
    b = 5;

    // Call the defined Func in the kernel through CallFunc.
    CcuResult ret = CallFunc<MyAdd>(a);
    if (ret != CCU_SUCCESS) { return ret; }

    return CallFunc<MyAdd>(b);
}
```

For details about the call semantics, return values, and exceptions of `CallFunc`, see [CallFunc](CallFunc.md).
