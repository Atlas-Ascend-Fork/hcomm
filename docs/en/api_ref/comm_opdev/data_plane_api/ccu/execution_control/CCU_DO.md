# CCU_DO

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:59:50.575Z pushedAt=2026-09-30T06:15:34.161Z -->

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

Starts a do-while loop block in a CCU kernel: the body is executed at least once, and then the accompanying `CCU_WHILE` determines whether to continue.

`CCU_DO` is a preprocessor macro that must be terminated with [CCU_WHILE](CCU_WHILE.md) to form a complete do-while loop:

```cpp
CCU_DO { body } CCU_WHILE(condExpr);
```

> [!CAUTION] Caution
> `CCU_DO` must be terminated with `CCU_WHILE(cond);`. Omitting the termination does not cause a compilation error, but leads to incorrect runtime behavior: the body is executed only once and then execution proceeds sequentially downward, and it affects subsequent `CCU_WHILE` statements, causing disordered loop jumps that are extremely difficult to debug.

## Macro Syntax

```cpp
CCU_DO {
    // body (executed at least once).
} CCU_WHILE(condExpr);
```

## Parameters

`CCU_DO` takes no parameters. For the definition and generation of `condExpr` in `CCU_WHILE(condExpr)`, see [CCU_IF](CCU_IF.md#parameters).

## Return Value

`CCU_DO` is a preprocessor macro and does not return [CcuResult](../../../datatype_definition/CcuResult.md) by itself; it does not fail under normal usage.

## Constraints

- `CCU_DO` must be immediately terminated by `CCU_WHILE(cond);`. If the termination is missing, the runtime behavior is incorrect, and no compile-time or registration-time error is generated.
- No other CCU API calls are allowed between `CCU_DO { body }` and its matching `CCU_WHILE(cond);`; otherwise, the do-while loop cannot be paired correctly.
- `CCU_DO ... CCU_WHILE` cannot be used inside the body lambda of `ccu::Loop`, because software loops are not supported inside a hardware loop body.
- The immediate value (`imm`) compared in `condExpr` must be of the `uint64_t` type.

## Example

```cpp
using namespace AscendC::ccu;

// Scenario: do-while loop, where the body is executed at least once, and then whether to continue is determined by the condition.
CcuResult MyKernel(CcuKernelArg arg) {
    Variable inner, innerLimit, one;
    inner = 0;
    innerLimit = 5;
    one = 1;

    // do-while: the body is executed once first, and then the condition is evaluated.
    CCU_DO {
        inner = inner + one;
    } CCU_WHILE(inner != innerLimit);

    return CCU_SUCCESS;
}
```
