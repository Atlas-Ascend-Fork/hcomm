# CCU_ELSE

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:59:48.106Z pushedAt=2026-09-30T06:18:08.046Z -->

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

Starts the else branch of `CCU_IF` in a CCU kernel: this branch is executed when the `CCU_IF` condition is not met.

`CCU_ELSE` is a preprocessor macro that must be used immediately after `CCU_IF { body }`. It cannot appear independently, and together with `CCU_IF` it forms a complete if-else structure.

> [!NOTE] Note
> `CCU_ELSE` is optional. `CCU_IF` can be used alone, and it is also valid without `CCU_ELSE`.

## Macro Syntax

```cpp
CCU_IF(condExpr) {
    // then branch body
} CCU_ELSE {
    // else branch body
}
```

For the complete usage, see [CCU_IF](CCU_IF.md).

## Parameters

`CCU_ELSE` takes no parameters and is automatically paired with the preceding `CCU_IF`, so you do not need to pass any parameters.

## Return Value

`CCU_ELSE` is a preprocessor macro and does not return [CcuResult](../../../datatype_definition/CcuResult.md); it does not fail under normal usage.

## Constraints

- `CCU_IF` supports matching only one `CCU_ELSE`. A single `if` with multiple `else` branches is not supported.
- `CCU_ELSE` must immediately follow `CCU_IF { body }`. It cannot appear independently or in any other position.
- No other CCU API calls are allowed between `CCU_IF { body }` and `CCU_ELSE { else-body }`.

> [!CAUTION] Caution
> Once any CCU API call (such as data movement or synchronization) is inserted in between, the framework automatically closes the `CCU_IF` in advance. The subsequent `CCU_ELSE` then cannot find a matching `CCU_IF`, so its body is skipped while registration still succeeds. No error value is returned at runtime, making this extremely difficult to debug. Ensure that no CCU API call exists between `CCU_IF { ... }` and `CCU_ELSE { ... }`.

- The `else if` syntax is not supported. For multiple branches, nest `CCU_IF`:

```cpp
CCU_IF(v == 0) {
    // case 0
} CCU_ELSE {
    CCU_IF(v == 1) {
        // case 1
    } CCU_ELSE {
        // other case
    }
}
```

## Example

```cpp
using namespace AscendC::ccu;

CcuResult MyKernel(CcuKernelArg arg) {
    Variable mode;
    LoadArg(mode, 0);

    CCU_IF(mode == 0) {
        // Handle mode 0.
        Variable result;
        result = 100;
    } CCU_ELSE {
        // Handle non-mode 0.
        Variable result;
        result = 200;
    }

    return CCU_SUCCESS;
}
```
